from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run
from settings import AUDIO_PRESETS, DEFAULT_AUDIO_PRESET, AUDIO_DEFAULTS


@dataclass
class AudioConfig:
    # Stream selection
    audio_stream: Optional[str] = None
    audio_auto: bool = True

    # Health checks
    min_audio_sec: float = AUDIO_DEFAULTS.min_audio_sec

    # Cleaning
    no_clean: bool = False
    preset: str = AUDIO_DEFAULTS.preset
    gain_db: float = AUDIO_DEFAULTS.gain_db

    # Output format
    target_sr: int = AUDIO_DEFAULTS.target_sr
    target_ch: int = AUDIO_DEFAULTS.target_ch

    # Stability knobs (anti-gap)
    use_dynaudnorm_in_extract: bool = AUDIO_DEFAULTS.use_dynaudnorm_in_extract
    extract_dynaudnorm: str = AUDIO_DEFAULTS.extract_dynaudnorm
    extract_resample: str = AUDIO_DEFAULTS.extract_resample

    # Fallback probe settings
    fallback_probesize: str = AUDIO_DEFAULTS.fallback_probesize
    fallback_analyzeduration: str = AUDIO_DEFAULTS.fallback_analyzeduration


def build_audio_filter(preset: str, gain_db: float = 0.0) -> str:
    p = (preset or DEFAULT_AUDIO_PRESET).strip().lower()
    chain = AUDIO_PRESETS.get(p, AUDIO_PRESETS[DEFAULT_AUDIO_PRESET])
    if abs(gain_db) > 0.01:
        chain = f"volume={gain_db}dB,{chain}"
    return chain


def _ffprobe_json(input_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    res = run(cmd, check=True)
    return json.loads(res.stdout or "{}")


def _get_duration_sec(meta: dict) -> Optional[float]:
    fmt = meta.get("format") or {}
    dur = fmt.get("duration")
    try:
        return float(dur) if dur is not None else None
    except Exception:
        return None


def pick_best_audio_map(input_path: Path) -> str:
    """
    Return mapping like '0:a:0' or '0:a:1'.
    Heuristic: highest bitrate; break ties with channels, then sample rate.
    """
    meta = _ffprobe_json(input_path)
    streams = meta.get("streams") or []

    audio_ord: list[tuple[int, dict]] = []
    ordinal = 0
    for s in streams:
        if s.get("codec_type") == "audio":
            audio_ord.append((ordinal, s))
            ordinal += 1

    if not audio_ord:
        return "0:a:0"

    def score(s: dict) -> tuple[int, int, int]:
        br = s.get("bit_rate")
        ch = s.get("channels")
        sr = s.get("sample_rate")
        try:
            br_i = int(br) if br is not None else 0
        except Exception:
            br_i = 0
        try:
            ch_i = int(ch) if ch is not None else 0
        except Exception:
            ch_i = 0
        try:
            sr_i = int(sr) if sr is not None else 0
        except Exception:
            sr_i = 0
        return (br_i, ch_i, sr_i)

    best_ordinal, _ = max(audio_ord, key=lambda t: score(t[1]))
    return f"0:a:{best_ordinal}"


def wav_preflight_ok(
    wav_path: Path,
    *,
    min_sec: float,
    expected_sec: Optional[float],
    expected_sr: int,
    expected_ch: int,
) -> tuple[bool, str]:
    if not wav_path.exists():
        return False, "wav missing"

    try:
        if wav_path.stat().st_size < 200_000:
            return False, "wav too small (likely wrong/empty track)"
    except Exception:
        pass

    meta = _ffprobe_json(wav_path)
    dur = _get_duration_sec(meta)
    if dur is None or dur <= 0.1:
        return False, "wav duration missing/zero"

    if dur < min_sec:
        return False, f"wav too short ({dur:.1f}s < {min_sec:.1f}s)"

    if expected_sec and expected_sec > 60 and dur < expected_sec * 0.60:
        return (
            False,
            f"wav much shorter than input ({dur:.1f}s vs ~{expected_sec:.1f}s)",
        )

    streams = meta.get("streams") or []
    a0 = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a0:
        sr = str(a0.get("sample_rate") or "")
        ch = a0.get("channels")
        if sr and int(sr) != int(expected_sr):
            return False, f"wav sample rate is {sr}, expected {expected_sr}"
        if ch is not None and int(ch) != int(expected_ch):
            return False, f"wav channels is {ch}, expected {expected_ch}"

    return True, "ok"


def extract_audio_stable(
    input_path: Path,
    out_wav: Path,
    *,
    audio_stream: Optional[str],
    audio_auto: bool,
    min_audio_sec: float,
    target_sr: int,
    target_ch: int,
    use_dynaudnorm: bool,
    extract_resample: str,
    extract_dynaudnorm: str,
    fallback_probesize: str,
    fallback_analyzeduration: str,
) -> Path:
    """
    Extracts a 'raw' WAV with anti-gap timestamp handling.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    expected_sec: Optional[float]
    try:
        expected_sec = _get_duration_sec(_ffprobe_json(input_path))
    except Exception:
        expected_sec = None

    if audio_stream:
        map_sel = audio_stream
    elif audio_auto:
        map_sel = pick_best_audio_map(input_path)
    else:
        map_sel = "0:a:0"

    af = extract_resample
    if use_dynaudnorm:
        af = f"{af},{extract_dynaudnorm}"

    cmd_extract = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-thread_queue_size",
        "4096",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-i",
        str(input_path),
        "-map",
        map_sel,
        "-vn",
        "-af",
        af,
        "-ac",
        str(target_ch),
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd_extract, check=True)

    ok, reason = wav_preflight_ok(
        out_wav,
        min_sec=min_audio_sec,
        expected_sec=expected_sec,
        expected_sr=target_sr,
        expected_ch=target_ch,
    )
    if ok:
        return out_wav

    bad = out_wav.with_suffix(".bad.wav")
    try:
        if bad.exists():
            bad.unlink()
        out_wav.replace(bad)
    except Exception:
        pass

    print(
        f"[subloom] audio preflight failed ({reason}) — retrying extraction (map {map_sel})"
    )

    cmd_extract_fallback = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-thread_queue_size",
        "8192",
        "-analyzeduration",
        fallback_analyzeduration,
        "-probesize",
        fallback_probesize,
        "-fflags",
        "+genpts+igndts",
        "-avoid_negative_ts",
        "make_zero",
        "-i",
        str(input_path),
        "-map",
        map_sel,
        "-vn",
        "-af",
        af,
        "-ac",
        str(target_ch),
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd_extract_fallback, check=True)

    ok2, reason2 = wav_preflight_ok(
        out_wav,
        min_sec=min_audio_sec,
        expected_sec=expected_sec,
        expected_sr=target_sr,
        expected_ch=target_ch,
    )
    if not ok2:
        raise RuntimeError(
            "Audio extraction still looks broken after fallback.\n"
            f"  input:  {input_path}\n"
            f"  map:    {map_sel}\n"
            f"  reason: {reason2}\n"
            "Try: --audio-stream 0:a:1 (or another track)\n"
        )

    return out_wav


def clean_audio(
    wav_path: Path,
    *,
    preset: str,
    gain_db: float,
    target_sr: int,
    target_ch: int,
) -> Path:
    clean_wav = wav_path.with_suffix(".clean.wav")
    af = build_audio_filter(preset=preset, gain_db=gain_db)

    cmd_clean = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-af",
        af,
        "-ac",
        str(target_ch),
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(clean_wav),
    ]
    run(cmd_clean, check=True)
    return clean_wav


def extract_and_clean_audio(
    input_path: Path,
    out_wav: Path,
    cfg: AudioConfig,
    *,
    out_wav_stereo: Path | None = None,
) -> Path:
    """
    Compatibility wrapper used by subloom.py.
    """
    # Avoid double loudness normalization for anime / chaotic dialogue presets
    use_extract_dyn = cfg.use_dynaudnorm_in_extract
    if (cfg.preset or "").strip().lower() == "anime":
        use_extract_dyn = False

    raw = extract_audio_stable(
        input_path,
        out_wav,
        audio_stream=cfg.audio_stream,
        audio_auto=cfg.audio_auto,
        min_audio_sec=cfg.min_audio_sec,
        target_sr=cfg.target_sr,
        target_ch=cfg.target_ch,
        use_dynaudnorm=use_extract_dyn,
        extract_resample=cfg.extract_resample,
        extract_dynaudnorm=cfg.extract_dynaudnorm,
        fallback_probesize=cfg.fallback_probesize,
        fallback_analyzeduration=cfg.fallback_analyzeduration,
    )

    # Optional stereo clean wav (used only by rescue pass)
    if out_wav_stereo is not None:
        out_wav_stereo.parent.mkdir(parents=True, exist_ok=True)
        extract_audio_stable(
            input_path,
            out_wav_stereo,
            audio_stream=cfg.audio_stream,
            audio_auto=cfg.audio_auto,
            min_audio_sec=cfg.min_audio_sec,
            target_sr=cfg.target_sr,
            target_ch=2,
            use_dynaudnorm=use_extract_dyn,
            extract_resample=cfg.extract_resample,
            extract_dynaudnorm=cfg.extract_dynaudnorm,
            fallback_probesize=cfg.fallback_probesize,
            fallback_analyzeduration=cfg.fallback_analyzeduration,
        )

    if cfg.no_clean:
        return raw

    return clean_audio(
        raw,
        preset=cfg.preset,
        gain_db=cfg.gain_db,
        target_sr=cfg.target_sr,
        target_ch=cfg.target_ch,
    )
