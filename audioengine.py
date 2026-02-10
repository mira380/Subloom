from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run
from settings import AUDIO_PRESETS, DEFAULT_AUDIO_PRESET, AUDIO_DEFAULTS


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------


@dataclass
class AudioConfig:
    audio_stream: Optional[str] = None
    audio_auto: bool = True
    min_audio_sec: float = AUDIO_DEFAULTS.min_audio_sec
    no_clean: bool = False
    preset: str = AUDIO_DEFAULTS.preset
    gain_db: float = AUDIO_DEFAULTS.gain_db
    target_sr: int = AUDIO_DEFAULTS.target_sr
    target_ch: int = AUDIO_DEFAULTS.target_ch
    use_dynaudnorm_in_extract: bool = AUDIO_DEFAULTS.use_dynaudnorm_in_extract
    extract_dynaudnorm: str = AUDIO_DEFAULTS.extract_dynaudnorm
    extract_resample: str = AUDIO_DEFAULTS.extract_resample
    fallback_probesize: str = AUDIO_DEFAULTS.fallback_probesize
    fallback_analyzeduration: str = AUDIO_DEFAULTS.fallback_analyzeduration

    # RNNoise-style denoise via ffmpeg's arnndn filter
    use_rnnoise: bool = True
    rnnoise_model: Optional[str] = (
        ""  # path to model file
)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def ffmpeg_has_filter(name: str) -> bool:
    try:
        rr = run(["ffmpeg", "-hide_banner", "-filters"], check=True)
        txt = (rr.stdout or "") + "\n" + (rr.stderr or "")
        return name in txt
    except Exception:
        return False


def build_audio_filter(
    preset: str,
    gain_db: float = 0.0,
    *,
    use_rnnoise: bool = False,
    rnnoise_model: Optional[str] = None,
) -> str:
    p = (preset or DEFAULT_AUDIO_PRESET).strip().lower()
    chain = AUDIO_PRESETS.get(p, AUDIO_PRESETS[DEFAULT_AUDIO_PRESET])

    # Keep existing behavior: gain first
    if abs(gain_db) > 0.01:
        chain = f"volume={gain_db}dB,{chain}"

    # RNNoise denoise first, then rest of chain
    if use_rnnoise:
        if rnnoise_model:
            chain = f"arnndn=m='{rnnoise_model}':mix=0.55,{chain}"
        else:
            chain = f"arnndn=mix=0.55,{chain}"

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


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _norm_lang(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t in ("jp", "jpn", "ja-jp"):
        return "ja"
    return t


# ------------------------------------------------------------
# Audio track selection
# ------------------------------------------------------------


def pick_best_audio_map(input_path: Path) -> str:
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

    def get_tags(s: dict) -> dict:
        t = s.get("tags")
        return t if isinstance(t, dict) else {}

    def lang_of(s: dict) -> str:
        tags = get_tags(s)
        return _norm_lang(str(tags.get("language") or ""))

    def title_of(s: dict) -> str:
        tags = get_tags(s)
        for k in ("title", "handler_name", "NAME", "name"):
            v = tags.get(k)
            if v:
                return str(v)
        return ""

    def is_commentary(s: dict) -> bool:
        t = (title_of(s) or "").lower()
        return any(x in t for x in ("commentary", "director", "descriptive"))

    def ch_score(ch: int) -> int:
        if ch == 2:
            return 30
        if ch == 1:
            return 25
        if ch == 6:
            return 5
        return max(0, 20 - abs(ch - 2) * 3)

    def score(s: dict) -> tuple[int, int, int, int]:
        br_i = _safe_int(s.get("bit_rate"), 0)
        sr_i = _safe_int(s.get("sample_rate"), 0)
        ch_i = _safe_int(s.get("channels"), 0)

        lang = lang_of(s)
        lang_boost = 50 if lang == "ja" else 0
        comm_penalty = -25 if is_commentary(s) else 0
        ch_boost = ch_score(ch_i)

        return (lang_boost, comm_penalty, ch_boost, br_i + sr_i)

    best_ordinal, _ = max(audio_ord, key=lambda t: score(t[1]))
    return f"0:a:{best_ordinal}"


# ------------------------------------------------------------
# Preflight check
# ------------------------------------------------------------


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

    meta = _ffprobe_json(wav_path)
    dur = _get_duration_sec(meta)
    if dur is None or dur < min_sec:
        return False, "wav too short"

    # Silence check
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            "25",
            "-i",
            str(wav_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        rr = run(cmd, check=True)
        st = rr.stderr or ""
        m_mean = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", st)
        m_max = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", st)

        if m_mean and m_max:
            mean_db = float(m_mean.group(1))
            max_db = float(m_max.group(1))
            if mean_db <= -55.0 and max_db <= -40.0:
                return False, "wav near-silent"
    except Exception:
        pass

    return True, "ok"


# ------------------------------------------------------------
# Extraction + cleaning
# ------------------------------------------------------------


def extract_audio_stable(input_path: Path, out_wav: Path, cfg: AudioConfig) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if cfg.audio_stream:
        map_sel = cfg.audio_stream
    elif cfg.audio_auto:
        map_sel = pick_best_audio_map(input_path)
    else:
        map_sel = "0:a:0"

    af = cfg.extract_resample
    if cfg.use_dynaudnorm_in_extract:
        af = f"{af},{cfg.extract_dynaudnorm}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        map_sel,
        "-vn",
        "-af",
        af,
        "-ac",
        str(cfg.target_ch),
        "-ar",
        str(cfg.target_sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd, check=True)

    ok, reason = wav_preflight_ok(
        out_wav,
        min_sec=cfg.min_audio_sec,
        expected_sec=None,
        expected_sr=cfg.target_sr,
        expected_ch=cfg.target_ch,
    )
    if not ok:
        raise RuntimeError(f"Audio extraction failed: {reason}")

    return out_wav


def extract_and_clean_audio(
    input_path: Path,
    out_wav: Path,
    cfg: AudioConfig,
    *,
    out_wav_stereo: Optional[Path] = None,
) -> Path:
    raw_wav = extract_audio_stable(input_path, out_wav, cfg)

    if cfg.no_clean:
        return raw_wav

    clean_wav = out_wav.with_name(out_wav.stem + ".clean.wav")

    use_rn = bool(cfg.use_rnnoise) and ffmpeg_has_filter("arnndn")
    af = build_audio_filter(
        cfg.preset,
        cfg.gain_db,
        use_rnnoise=use_rn,
        rnnoise_model=cfg.rnnoise_model,
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_wav),
        "-af",
        af,
        "-ac",
        str(cfg.target_ch),
        "-ar",
        str(cfg.target_sr),
        str(clean_wav),
    ]
    run(cmd, check=True)

    if out_wav_stereo:
        cmd_stereo = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_wav),
            "-ac",
            "2",
            "-ar",
            str(cfg.target_sr),
            str(out_wav_stereo),
        ]
        run(cmd_stereo, check=True)

    return clean_wav
