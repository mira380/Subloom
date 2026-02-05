from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from difflib import SequenceMatcher
import wave
import array
import math

# audioop is faster
try:
    import audioop  # type: ignore
except Exception:
    audioop = None  # type: ignore

from merge_subs import (
    SrtItem,
    parse_srt,
    write_srt_items,
    srt_time_to_seconds,
    seconds_to_srt_time,
    normalize_jp,
    run,
)

# ------------------------------------------------------------
# Pass 3: Rescue
# ------------------------------------------------------------


@dataclass
class RescueConfig:
    # ---------- Gap windows ----------
    min_gap_s: float = 2.5
    pad_s: float = 0.35
    max_window_s: float = 9.0
    max_windows: int = 18

    # ---------- Energy windows ----------
    enable_energy_windows: bool = True
    energy_frame_ms: int = 30
    energy_min_region_s: float = 0.35
    energy_pad_s: float = 0.20

    # dBFS threshold. Higher (e.g. -30) = more sensitive, more false positives.
    # Lower (e.g. -38) = stricter, fewer windows.
    energy_dbfs_thresh: float = -34.0

    # If total energy windows explode, cap how much audio we’re willing to re-check.
    max_total_energy_window_s: float = 90.0

    # ---------- Stereo rescue ----------
    enable_stereo_split: bool = True

    # Only run L/R split if the base slice produced "too few" lines.
    # 0 = only if base produced nothing
    # 1 = try L/R if base produced <= 1 line
    stereo_fallback_min_lines: int = 0

    # ---------- Duplicate guardrails ----------
    near_s: float = 0.9
    sim_thresh: float = 0.78

    # ---------- Output behavior ----------
    # Always written:
    #   rescue_report.txt
    #   rescue_added.srt (if any)
    #   rescue_merged_preview.srt
    # New:
    #   low_confidence.txt
    #
    # If set, we also write merged result here
    out_merged_srt: Optional[Path] = None

    # ---------- Rescue-mode hook ----------
    rescue_decode_profile: str = "default"  # "default" | "aggressive"


def _norm_for_dupe_check(s: str) -> str:
    return normalize_jp((s or "").strip())


def _similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out: List[Tuple[float, float]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def _clamp_window(ws: float, we: float, cfg: RescueConfig) -> Tuple[float, float]:
    ws = max(0.0, ws)
    we = max(ws + 0.05, we)
    if (we - ws) > cfg.max_window_s:
        center = (ws + we) / 2.0
        half = cfg.max_window_s / 2.0
        ws = max(0.0, center - half)
        we = center + half
    return ws, we


def _find_gap_windows(
    items: List[SrtItem], cfg: RescueConfig
) -> List[Tuple[float, float, str]]:
    if len(items) < 2:
        return []

    windows: List[Tuple[float, float, str]] = []
    for prev, cur in zip(items, items[1:]):
        prev_end = srt_time_to_seconds(prev.end)
        cur_start = srt_time_to_seconds(cur.start)
        gap = cur_start - prev_end

        if gap < cfg.min_gap_s:
            continue

        ws = prev_end - cfg.pad_s
        we = cur_start + cfg.pad_s
        ws, we = _clamp_window(ws, we, cfg)
        windows.append((ws, we, f"gap={gap:.2f}s"))

    return windows[: cfg.max_windows]


def _slice_wav_ffmpeg(
    in_wav: Path, out_wav: Path, start_s: float, end_s: float
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(in_wav),
        "-t",
        f"{dur:.3f}",
        "-ar",
        "16000",
        str(out_wav),
    ]
    run(cmd, check=True)


def _slice_wav_ffmpeg_channel(
    in_wav: Path, out_wav: Path, start_s: float, end_s: float, which: str
) -> None:
    """
    which: "L" or "R"
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, end_s - start_s)

    if which == "L":
        pan = "pan=mono|c0=c0"
    else:
        pan = "pan=mono|c0=c1"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(in_wav),
        "-t",
        f"{dur:.3f}",
        "-af",
        pan,
        "-ar",
        "16000",
        str(out_wav),
    ]
    run(cmd, check=True)


def _shift_items(items: List[SrtItem], delta_s: float) -> List[SrtItem]:
    out: List[SrtItem] = []
    for it in items:
        a = srt_time_to_seconds(it.start) + delta_s
        b = srt_time_to_seconds(it.end) + delta_s
        out.append(
            SrtItem(
                idx=0,
                start=seconds_to_srt_time(a),
                end=seconds_to_srt_time(b),
                text=(it.text or "").strip(),
                whisper_text=(getattr(it, "whisper_text", "") or ""),
                kotoba_text=(getattr(it, "kotoba_text", "") or ""),
            )
        )
    return out


def _iter_nearby(base: List[SrtItem], t_s: float, near_s: float) -> Iterable[SrtItem]:
    lo = t_s - near_s
    hi = t_s + near_s
    for b in base:
        bs = srt_time_to_seconds(b.start)
        be = srt_time_to_seconds(b.end)
        if (lo <= bs <= hi) or (lo <= be <= hi):
            yield b


def _additive_merge(
    base: List[SrtItem], rescue: List[SrtItem], cfg: RescueConfig
) -> Tuple[List[SrtItem], List[SrtItem]]:
    added: List[SrtItem] = []

    for r in rescue:
        r_text = (r.text or "").strip()
        if not r_text:
            continue

        r_norm = _norm_for_dupe_check(r_text)
        if not r_norm:
            continue

        r_t = srt_time_to_seconds(r.start)
        candidates = list(_iter_nearby(base, r_t, cfg.near_s))

        is_dup = False
        for c in candidates:
            c_text = (c.text or "").strip()
            if not c_text:
                continue

            if _norm_for_dupe_check(c_text) == r_norm:
                is_dup = True
                break

            if _similarity(r_text, c_text) >= cfg.sim_thresh:
                is_dup = True
                break

        if not is_dup:
            added.append(r)

    merged = sorted(base + added, key=lambda it: srt_time_to_seconds(it.start))
    return merged, added


# ----------------------------
# Energy detection
# ----------------------------


def _wav_info(path: Path) -> Tuple[int, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getframerate()


def _read_wav_mono_i16(path: Path) -> Tuple[int, array.array]:
    """
    Returns (sr, samples) where samples are int16 mono.
    If file has 2ch, we downmix (L+R)/2 in Python.
    """
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()

        if sampwidth != 2:
            raise RuntimeError(
                f"Expected 16-bit PCM wav, got sampwidth={sampwidth} for {path}"
            )

        raw = wf.readframes(nframes)
        data = array.array("h")
        data.frombytes(raw)

        if ch == 1:
            return sr, data

        if ch == 2:
            mono = array.array("h")
            mono_extend = mono.append
            # data is interleaved L R L R...
            for i in range(0, len(data), 2):
                mono_extend(int((data[i] + data[i + 1]) / 2))
            return sr, mono

        raise RuntimeError(f"Unsupported channel count: {ch} for {path}")


def _rms_dbfs(frame: array.array) -> float:
    if not frame:
        return -120.0
    # int16 full-scale is 32768
    acc = 0.0
    for x in frame:
        acc += float(x) * float(x)
    mean = acc / float(len(frame))
    rms = math.sqrt(mean)
    if rms <= 0.0:
        return -120.0
    db = 20.0 * math.log10(rms / 32768.0)
    return db


def _speech_regions_from_energy(
    wav_path: Path,
    *,
    frame_ms: int,
    dbfs_thresh: float,
    min_region_s: float,
) -> List[Tuple[float, float]]:
    # Fast path: stream frames and compute RMS in C via audioop
    if audioop is not None:
        with wave.open(str(wav_path), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()

            if sampwidth != 2:
                # fallback to old code for odd formats
                sr2, samples = _read_wav_mono_i16(wav_path)
                hop = max(1, int(sr2 * (frame_ms / 1000.0)))
                regions: List[Tuple[float, float]] = []
                in_region = False
                start_i = 0
                for i in range(0, len(samples), hop):
                    frame = samples[i : i + hop]
                    db = _rms_dbfs(frame)
                    if db >= dbfs_thresh and not in_region:
                        in_region = True
                        start_i = i
                    elif db < dbfs_thresh and in_region:
                        in_region = False
                        end_i = i
                        s = start_i / sr2
                        e = end_i / sr2
                        if (e - s) >= min_region_s:
                            regions.append((s, e))
                if in_region:
                    s = start_i / sr2
                    e = len(samples) / sr2
                    if (e - s) >= min_region_s:
                        regions.append((s, e))
                return _merge_intervals(regions)

            hop_frames = max(1, int(sr * (frame_ms / 1000.0)))
            regions: List[Tuple[float, float]] = []
            in_region = False
            start_t = 0.0

            frame_index = 0
            while True:
                raw = wf.readframes(hop_frames)
                if not raw:
                    break

                # downmix stereo to mono bytes if needed
                if ch == 2:
                    raw_mono = audioop.tomono(raw, 2, 0.5, 0.5)
                elif ch == 1:
                    raw_mono = raw
                else:
                    # weird multi-channel: fallback to old method
                    sr2, samples = _read_wav_mono_i16(wav_path)
                    hop = max(1, int(sr2 * (frame_ms / 1000.0)))
                    regions2: List[Tuple[float, float]] = []
                    in_region2 = False
                    start_i2 = 0
                    for i in range(0, len(samples), hop):
                        frame = samples[i : i + hop]
                        db = _rms_dbfs(frame)
                        if db >= dbfs_thresh and not in_region2:
                            in_region2 = True
                            start_i2 = i
                        elif db < dbfs_thresh and in_region2:
                            in_region2 = False
                            end_i2 = i
                            s = start_i2 / sr2
                            e = end_i2 / sr2
                            if (e - s) >= min_region_s:
                                regions2.append((s, e))
                    if in_region2:
                        s = start_i2 / sr2
                        e = len(samples) / sr2
                        if (e - s) >= min_region_s:
                            regions2.append((s, e))
                    return _merge_intervals(regions2)

                rms = audioop.rms(raw_mono, 2)  # int16 RMS
                if rms <= 0:
                    db = -120.0
                else:
                    db = 20.0 * math.log10(rms / 32768.0)

                t0 = (frame_index * hop_frames) / sr

                if db >= dbfs_thresh and not in_region:
                    in_region = True
                    start_t = t0
                elif db < dbfs_thresh and in_region:
                    in_region = False
                    end_t = t0
                    if (end_t - start_t) >= min_region_s:
                        regions.append((start_t, end_t))

                frame_index += 1

            if in_region:
                end_t = (frame_index * hop_frames) / sr
                if (end_t - start_t) >= min_region_s:
                    regions.append((start_t, end_t))

        return _merge_intervals(regions)

    # Fallback: array-based scanner
    sr, samples = _read_wav_mono_i16(wav_path)
    hop = max(1, int(sr * (frame_ms / 1000.0)))

    regions: List[Tuple[float, float]] = []
    in_region = False
    start_i = 0

    # We step by hop, using hop-sized frames
    for i in range(0, len(samples), hop):
        frame = samples[i : i + hop]
        db = _rms_dbfs(frame)

        if db >= dbfs_thresh and not in_region:
            in_region = True
            start_i = i
        elif db < dbfs_thresh and in_region:
            in_region = False
            end_i = i
            s = start_i / sr
            e = end_i / sr
            if (e - s) >= min_region_s:
                regions.append((s, e))

    if in_region:
        s = start_i / sr
        e = len(samples) / sr
        if (e - s) >= min_region_s:
            regions.append((s, e))

    return _merge_intervals(regions)


def _subtitle_coverage(items: List[SrtItem]) -> List[Tuple[float, float]]:
    cov: List[Tuple[float, float]] = []
    for it in items:
        s = srt_time_to_seconds(it.start)
        e = srt_time_to_seconds(it.end)
        if e > s:
            cov.append((s, e))
    return _merge_intervals(cov)


def _subtract_intervals(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Return portions of A not covered by B.
    Assumes both are merged/sorted.
    """
    out: List[Tuple[float, float]] = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return _merge_intervals(out)


def _energy_windows(
    items: List[SrtItem], clean_wav: Path, cfg: RescueConfig
) -> List[Tuple[float, float, str]]:
    if not cfg.enable_energy_windows:
        return []

    speech = _speech_regions_from_energy(
        clean_wav,
        frame_ms=cfg.energy_frame_ms,
        dbfs_thresh=cfg.energy_dbfs_thresh,
        min_region_s=cfg.energy_min_region_s,
    )
    if not speech:
        return []

    coverage = _subtitle_coverage(items)
    missing = _subtract_intervals(speech, coverage)
    if not missing:
        return []

    # pad and clamp, then cap total duration
    padded: List[Tuple[float, float, str]] = []
    total = 0.0
    for s, e in missing:
        ws = s - cfg.energy_pad_s
        we = e + cfg.energy_pad_s
        ws, we = _clamp_window(ws, we, cfg)
        dur = we - ws
        if total + dur > cfg.max_total_energy_window_s:
            break
        total += dur
        padded.append((ws, we, "energy_missing_subs"))

    return padded


# ----------------------------
# Confidence flags
# ----------------------------


def _write_confidence_flags(
    out_path: Path,
    *,
    base_items: List[SrtItem],
    gap_windows: List[Tuple[float, float, str]],
    energy_windows: List[Tuple[float, float, str]],
    added_items: List[SrtItem],
) -> None:
    """
    Flags places you might want to take a look at.
    """
    lines: List[str] = []
    lines.append("These are spots Subloom thinks are worth a second look.\n")

    if gap_windows:
        lines.append("## Gap-triggered windows")
        for ws, we, r in gap_windows:
            lines.append(
                f"- {seconds_to_srt_time(ws)} → {seconds_to_srt_time(we)} ({r})"
            )
        lines.append("")

    if energy_windows:
        lines.append(
            "## Energy-triggered windows (speech detected but subs were quiet)"
        )
        for ws, we, r in energy_windows:
            lines.append(
                f"- {seconds_to_srt_time(ws)} → {seconds_to_srt_time(we)} ({r})"
            )
        lines.append("")

    if added_items:
        lines.append("## Lines added by rescue")
        for it in added_items:
            lines.append(f"- {it.start} → {it.end}  {it.text.replace('\\n', ' / ')}")
        lines.append("")

    if not (gap_windows or energy_windows or added_items):
        lines.append("Nothing flagged. (Nice.)\n")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ----------------------------
# Main
# ----------------------------


def run_rescue_pass(
    *,
    clean_wav: Path,
    base_srt: Path,
    out_dir: Path,
    run_whispercpp_fn,
    whisper_bin: Path,
    whisper_model: Path,
    max_srt_line_dur: float,
    cfg: RescueConfig = RescueConfig(),
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Returns:
      (rescue_added_srt_path, rescue_report_path)

    Always writes:
      - rescue_report.txt
      - rescue_merged_preview.srt
      - low_confidence.txt

    Writes if needed:
      - rescue_added.srt
      - cfg.out_merged_srt (if provided)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base_items = parse_srt(base_srt)

    gap_windows = _find_gap_windows(base_items, cfg)
    energy_windows = _energy_windows(base_items, clean_wav, cfg)

    windows = gap_windows + energy_windows
    if not windows:
        return None, None

    # Deduplicate overlapping windows so we don't spam work
    merged_iv = _merge_intervals([(ws, we) for ws, we, _ in windows])
    windows = [(ws, we, "merged_window") for ws, we in merged_iv]
    windows = windows[: cfg.max_windows]

    report: List[str] = []
    all_rescue_shifted: List[SrtItem] = []

    # Check if stereo split is even possible
    stereo_possible = False
    if cfg.enable_stereo_split:
        try:
            ch, _sr = _wav_info(clean_wav)
            stereo_possible = ch >= 2
        except Exception:
            stereo_possible = False

    for i, (ws, we, reason) in enumerate(windows, start=1):
        # Base slice (whatever channels it already is)
        slice_wav = out_dir / f"rescue_win_{i:02d}.wav"
        _slice_wav_ffmpeg(clean_wav, slice_wav, ws, we)

        # Helper that actually runs whisper.cpp for this slice
        def _run_one(wav_path: Path, tag_suffix: str) -> Optional[List[SrtItem]]:
            outbase = out_dir / f"rescue_win_{i:02d}{tag_suffix}"

            kwargs = dict(
                whisper_bin=whisper_bin,
                whisper_model=whisper_model,
                whisper_tag=f"rescue_{i:02d}{tag_suffix}",
                max_srt_line_dur=max_srt_line_dur,
            )

            # Optional hook: only pass rescue_decode_profile if runner accepts it
            try:
                w = run_whispercpp_fn(
                    wav_path,
                    outbase,
                    **kwargs,
                    decode_profile=cfg.rescue_decode_profile,
                )
            except TypeError:
                w = run_whispercpp_fn(wav_path, outbase, **kwargs)

            slice_srt = Path(w.srt_path)
            if not slice_srt.exists():
                return None

            raw_items = parse_srt(slice_srt)
            return _shift_items(raw_items, ws)

        shifted_main = _run_one(slice_wav, "")
        n_main = len(shifted_main) if shifted_main else 0
        if shifted_main:
            all_rescue_shifted.extend(shifted_main)

        n_l = n_r = 0

        # stereo split is a fallback
        do_stereo = (
            stereo_possible
            and cfg.enable_stereo_split
            and (n_main <= cfg.stereo_fallback_min_lines)
        )

        if do_stereo:
            # Left
            slice_l = out_dir / f"rescue_win_{i:02d}.L.wav"
            _slice_wav_ffmpeg_channel(clean_wav, slice_l, ws, we, "L")
            shifted_l = _run_one(slice_l, "_L")
            if shifted_l:
                n_l = len(shifted_l)
                all_rescue_shifted.extend(shifted_l)

            # Right
            slice_r = out_dir / f"rescue_win_{i:02d}.R.wav"
            _slice_wav_ffmpeg_channel(clean_wav, slice_r, ws, we, "R")
            shifted_r = _run_one(slice_r, "_R")
            if shifted_r:
                n_r = len(shifted_r)
                all_rescue_shifted.extend(shifted_r)

        else:
            if cfg.enable_stereo_split and not stereo_possible:
                report.append(
                    f"[{i:02d}] {reason} {ws:.2f}-{we:.2f}s -> stereo_skip (mono input)"
                )
            elif cfg.enable_stereo_split and stereo_possible:
                report.append(
                    f"[{i:02d}] {reason} {ws:.2f}-{we:.2f}s -> stereo_skip (base_ok base={n_main})"
                )

        report.append(
            f"[{i:02d}] {reason} {ws:.2f}-{we:.2f}s -> base={n_main} L={n_l} R={n_r}"
        )

    merged, added = _additive_merge(base_items, all_rescue_shifted, cfg)

    rep_path = out_dir / "rescue_report.txt"
    rep_path.write_text(
        "\n".join(report) + f"\nadded_lines={len(added)}\n",
        encoding="utf-8",
    )

    preview = out_dir / "rescue_merged_preview.srt"
    write_srt_items(merged, preview)

    # Confidence flags
    low_conf = out_dir / "low_confidence.txt"
    _write_confidence_flags(
        low_conf,
        base_items=base_items,
        gap_windows=gap_windows,
        energy_windows=energy_windows,
        added_items=added,
    )

    # Optional: write merged output path directly
    if cfg.out_merged_srt:
        cfg.out_merged_srt.parent.mkdir(parents=True, exist_ok=True)
        write_srt_items(merged, cfg.out_merged_srt)

    if not added:
        return None, rep_path

    added_srt = out_dir / "rescue_added.srt"
    write_srt_items(added, added_srt)

    return added_srt, rep_path
