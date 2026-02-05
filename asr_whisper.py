from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run, seconds_to_srt_time


@dataclass
class WhisperPaths:
    json_path: Path
    srt_path: Path
    txt_path: Optional[Path] = None


def export_srt_from_whisper_json(
    json_path: Path, srt_path: Path, max_line_dur: float
) -> None:
    """
    whisper.cpp outputs a JSON 'transcription' list with millisecond offsets.
    This writes a normal SRT while capping line durations.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    transcription = data.get("transcription")
    if not isinstance(transcription, list):
        keys = (
            ", ".join(list(data.keys())[:30])
            if isinstance(data, dict)
            else "<not a dict>"
        )
        raise RuntimeError(
            f"Expected 'transcription' list in whisper JSON. Top-level keys: {keys}"
        )

    out_lines: list[str] = []
    idx = 1

    for item in transcription:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue

        offsets = item.get("offsets") or {}
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if start_ms is None or end_ms is None:
            continue

        start_s = float(start_ms) / 1000.0
        end_s = float(end_ms) / 1000.0
        if end_s <= start_s:
            end_s = start_s + 1.2

        # cap long lines
        if (end_s - start_s) > max_line_dur:
            end_s = start_s + max_line_dur

        out_lines.append(str(idx))
        out_lines.append(
            f"{seconds_to_srt_time(start_s)} --> {seconds_to_srt_time(end_s)}"
        )
        out_lines.append(text)
        out_lines.append("")
        idx += 1

    srt_path.write_text("\n".join(out_lines), encoding="utf-8")


def run_whispercpp(
    clean_wav: Path,
    outbase: Path,
    *,
    whisper_bin: Path,
    whisper_model: Path,
    whisper_tag: str,
    max_srt_line_dur: float,
    threads: int = 8,
    beam_size: int = 2,
    best_of: int = 2,
    device: str | None = "0",
) -> WhisperPaths:
    """
    Pass 1: Whisper (timing source).
    Runs whisper.cpp, writes JSON + TXT, then generates the timing SRT from JSON.
    Outputs are written next to outbase using names like:
      <outbase>.<whisper_tag>.json/.txt/.srt
    """
    out_prefix = str(outbase.parent / f"{outbase.name}.{whisper_tag}")

    cmd: list[str] = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(clean_wav),
        "-l",
        "ja",
        "-t",
        str(max(1, int(threads))),
        "-bs",
        str(max(1, int(beam_size))),
        "-bo",
        str(max(1, int(best_of))),
        "-mc",
        "3",
    ]

    # GPU (optional)
    if device is not None:
        cmd += ["-dev", str(device), "-fa"]

    cmd += ["-otxt", "-oj", "-of", out_prefix]

    print("[subloom] whisper cmd:", " ".join(cmd))
    run(cmd, check=True)

    json_path = Path(out_prefix + ".json")
    txt_path = Path(out_prefix + ".txt")
    srt_path = Path(out_prefix + ".srt")

    if not json_path.exists():
        raise RuntimeError(f"Whisper JSON not found after run: {json_path}")

    # generate SRT from JSON
    export_srt_from_whisper_json(json_path, srt_path, max_srt_line_dur)

    if not srt_path.exists():
        raise RuntimeError(f"Whisper SRT not found after export: {srt_path}")

    return WhisperPaths(
        json_path=json_path,
        srt_path=srt_path,
        txt_path=txt_path if txt_path.exists() else None,
    )
