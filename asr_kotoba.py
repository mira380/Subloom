from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run


@dataclass
class KotobaPaths:
    json_path: Path
    txt_path: Path


def find_kotoba_ggml_model(models_dir: Path) -> Optional[Path]:
    if not models_dir.exists():
        return None
    candidates = [p for p in models_dir.glob("*.bin") if p.is_file()]
    if not candidates:
        return None

    for p in candidates:
        if "q5_0" in p.name:
            return p

    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def run_kotoba_whispercpp(
    clean_wav: Path,
    outbase: Path,
    kotoba_model_path: Path,
    *,
    whisper_bin: Path,
    kotoba_tag: str,
    threads: int = 8,
    beam_size: int = 2,
    best_of: int = 2,
    device: str | None = "0",
) -> KotobaPaths:
    """
    Pass 2: Kotoba (accuracy checker).
    Runs Kotoba via whisper.cpp binary, writes JSON + TXT.
    We intentionally do NOT generate a Kotoba SRT — Whisper owns timing.
    Outputs:
      <outbase>.<kotoba_tag>.json/.txt
    """
    if not kotoba_model_path.exists():
        raise RuntimeError(f"Kotoba GGML model not found: {kotoba_model_path}")

    out_prefix = str(outbase.parent / f"{outbase.name}.{kotoba_tag}")

    cmd: list[str] = [
        str(whisper_bin),
        "-m",
        str(kotoba_model_path),
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

    if device is not None:
        cmd += ["-dev", str(device), "-fa"]

    cmd += ["-otxt", "-oj", "-of", out_prefix]

    print("[subloom] kotoba(ggml) cmd:", " ".join(cmd))
    run(cmd, check=True)

    kotoba_txt = Path(out_prefix + ".txt")
    kotoba_json = Path(out_prefix + ".json")

    if not kotoba_txt.exists():
        raise RuntimeError(f"Kotoba TXT not found after run: {kotoba_txt}")
    if not kotoba_json.exists():
        raise RuntimeError(f"Kotoba JSON not found after run: {kotoba_json}")

    return KotobaPaths(json_path=kotoba_json, txt_path=kotoba_txt)
