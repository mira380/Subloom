from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _progress_bar(prefix: str, i: int, n: int) -> None:
    """Simple stderr progress bar."""
    if n <= 0:
        return
    width = 24
    done = int(width * (i / n))
    bar = "█" * done + "·" * (width - done)
    pct = int(round(100 * (i / n)))
    sys.stderr.write(f"\r{prefix} [{bar}] {i}/{n} ({pct}%)")
    sys.stderr.flush()


# =========================================================
# Settings that control how the Ollama proofreading behaves
# (model choice, chunk size, safety rules, etc.)
# =========================================================


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b-instruct"
    url: str = "http://127.0.0.1:11434/api/generate"

    # Generation behavior (keep conservative)
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 4096
    timeout_sec: int = 300

    # Chunking
    window_sec: float = 45.0
    max_chars: int = 1000

    # Safety / behavior
    max_retries: int = 1
    skip_music_lines: bool = True
    skip_bracket_tags: bool = True

    # Style hint: "neutral" | "anime" | "formal"
    style: str = "neutral"

    # UX
    show_progress: bool = True

    # Context chooser behavior
    context_lines: int = 1  # prev/next
    only_suspicious: bool = True

    # Nuance protection
    max_change_ratio: float = 0.40
    max_abs_edit_chars: int = 28


# =========================================================
# Talking to Ollama
# Handles the HTTP request that sends text to the LLM and
# gets corrected subtitle lines back.
# =========================================================


def ollama_generate(cfg: OllamaConfig, prompt: str) -> str:
    """
    Uses Ollama /api/chat endpoint and returns assistant message content as plain text.
    """
    url = cfg.url.rstrip("/")
    # Allow cfg.url to be either base host or full /api/chat
    if not url.endswith("/api/chat"):
        url = url + "/api/chat"

    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_ctx": cfg.num_ctx,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    # Ollama chat response shape:
    # {"message":{"role":"assistant","content":"..."}, ...}
    msg = out.get("message") or {}
    return msg.get("content", "")


# =========================================================
# How we explain the task to the AI
# Builds the strict “proofread subtitles” instructions that
# get sent along with each chunk.
# =========================================================


def _style_hint(style: str) -> str:
    s = (style or "neutral").lower().strip()
    if s == "anime":
        return "Style: natural spoken Japanese typical of anime/TV dialogue. Keep casual tone if present.\n"
    if s == "formal":
        return "Style: keep polite/formal tone if present. Do not make casual lines formal.\n"
    return "Style: keep the original tone/register. Do not rewrite.\n"


def build_choice_prompt(items: List[Dict[str, Any]], style: str) -> str:
    rules = (
        "You are a Japanese subtitle proofreader.\n"
        "Goal: preserve meaning/nuance and choose the most correct candidate.\n"
        + _style_hint(style)
        + "\n"
        "STRICT RULES:\n"
        "1) Do NOT rewrite or paraphrase.\n"
        "2) ONLY choose from the provided candidates A/B/C.\n"
        "3) Use context (prev/next lines) to decide.\n"
        "4) If uncertain, pick A.\n"
        "5) Return ONLY valid JSON in the EXACT schema:\n"
        '   {"lines": [{"i": <int>, "pick": "A"|"B"|"C"}, ...]}\n'
        "6) Do NOT include explanations, markdown, or extra keys.\n"
    )

    input_json = json.dumps({"items": items}, ensure_ascii=False)
    return f"{rules}\nINPUT JSON:\n{input_json}\nOUTPUT JSON ONLY:\n"


# =========================================================
# Making sure the AI didn't go rogue
# Extracts JSON from the model response and checks that it
# didn’t change line counts, indices, or bloat the text.
# =========================================================


def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def parse_choice_json(s: str) -> Dict[int, str]:
    s = _extract_json_object(s)
    obj = json.loads(s)
    lines = obj.get("lines")
    if not isinstance(lines, list):
        raise ValueError("Missing or invalid lines[]")

    picks: Dict[int, str] = {}
    for it in lines:
        if not isinstance(it, dict):
            continue
        i = it.get("i")
        pick = it.get("pick")
        if isinstance(i, int) and pick in ("A", "B", "C"):
            picks[i] = pick

    return picks


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def too_big_a_change(original: str, corrected: str, cfg: OllamaConfig) -> bool:
    o = (original or "").strip()
    c = (corrected or "").strip()
    if not o or not c:
        return True

    sim = _similarity(o, c)
    abs_diff = abs(len(o) - len(c))
    approx_changed = int(round((1.0 - sim) * max(len(o), len(c))))

    if sim < (1.0 - cfg.max_change_ratio):
        return True
    if abs_diff > cfg.max_abs_edit_chars:
        return True
    if approx_changed > cfg.max_abs_edit_chars:
        return True
    return False


# =========================================================
# Breaking subtitles into safe-sized pieces
# Sends short time windows instead of the whole file so
# context stays manageable and outputs stay consistent.
# =========================================================


def chunk_by_time(
    subs: List[Dict[str, Any]], window_sec: float, max_chars: int
) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    if not subs:
        return chunks

    cur: List[Dict[str, Any]] = []
    start_t = float(subs[0]["start"])
    char_count = 0

    for s in subs:
        s_start = float(s["start"])
        s_text = str(s.get("text", ""))

        if (s_start - start_t) > window_sec or (char_count + len(s_text)) > max_chars:
            if cur:
                chunks.append(cur)
            cur = []
            start_t = s_start
            char_count = 0

        cur.append(s)
        char_count += len(s_text)

    if cur:
        chunks.append(cur)

    return chunks


# =========================================================
# Lines we don't bother sending to the AI
# Skips music notes, sound tags, or tiny bracket lines that
# don't need proofreading.
# =========================================================


def should_skip_line(text: str, cfg: OllamaConfig) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    if cfg.skip_music_lines:
        if "♪" in t or "♫" in t:
            return True
        if t.lower() in ("[music]", "[bgm]", "[applause]", "[laughs]"):
            return True

    if cfg.skip_bracket_tags:
        if (t.startswith("[") and t.endswith("]")) or (
            t.startswith("（") and t.endswith("）")
        ):
            if len(t) <= 12:
                return True

    return False


def looks_suspicious(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"[A-Za-z]{4,}", t):
        return True
    if "  " in t:
        return True
    return False


def _tiny_fix_candidate(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"\s+", " ", t)
    t = t.replace("...", "…").replace("・・", "…")
    t = t.replace("．", "。").replace("，", "、")
    t = t.replace("?", "？").replace("!", "！")
    return t


def build_candidates(sub: Dict[str, Any], cfg: OllamaConfig) -> List[Tuple[str, str]]:
    """
    Candidate sets for the chooser model.
    A: Whisper text (timing source)
    B: Current merged text with tiny safe normalization
    C: Kotoba text (alt hypothesis)

    Notes:
    - This does NOT ask the model to write new text.
    - If kotoba_text is missing, C falls back to merged.
    """
    whisper = str(sub.get("whisper_text") or sub.get("text") or "").strip()
    merged = str(sub.get("text") or "").strip()
    kotoba = str(sub.get("kotoba_text") or "").strip()

    b = _tiny_fix_candidate(merged)

    if not kotoba:
        kotoba = merged
    if not whisper:
        whisper = merged

    return [("A", whisper), ("B", b), ("C", kotoba)]


def _gather_context(lines: List[str], idx: int, k: int) -> Tuple[List[str], List[str]]:
    prev = []
    nxt = []
    for j in range(1, k + 1):
        if idx - j >= 0:
            prev.append(lines[idx - j])
        if idx + j < len(lines):
            nxt.append(lines[idx + j])
    prev.reverse()
    return prev, nxt


def choose_with_context(
    cfg: OllamaConfig,
    chunk: List[Dict[str, Any]],
) -> Dict[int, str]:
    chunk_sorted = sorted(chunk, key=lambda d: int(d["i"]))
    texts = [str(d.get("text", "")) for d in chunk_sorted]
    ids = [int(d["i"]) for d in chunk_sorted]

    items: List[Dict[str, Any]] = []
    for local_idx, (i, cur_text) in enumerate(zip(ids, texts)):
        if should_skip_line(cur_text, cfg):
            continue
        if cfg.only_suspicious and not looks_suspicious(cur_text):
            continue

        prev, nxt = _gather_context(texts, local_idx, max(0, int(cfg.context_lines)))

        cands = build_candidates(chunk_sorted[local_idx], cfg)

        items.append(
            {
                "i": i,
                "prev": prev,
                "cur": cur_text,
                "next": nxt,
                "candidates": [{"k": k, "text": t} for (k, t) in cands],
            }
        )

    if not items:
        return {}

    prompt = build_choice_prompt(items, cfg.style)

    for attempt in range(cfg.max_retries + 1):
        resp = ollama_generate(cfg, prompt)
        try:
            picks = parse_choice_json(resp)

            out: Dict[int, str] = {}
            for it in items:
                i = int(it["i"])
                cur_text = str(it["cur"])
                cand_map = {c["k"]: str(c["text"]) for c in it["candidates"]}

                pick = picks.get(i, "A")
                chosen = cand_map.get(pick, cur_text)

                # leash: if it picked something that changes too much, revert
                if too_big_a_change(cur_text, chosen, cfg):
                    chosen = cur_text

                out[i] = chosen

            return out
        except Exception:
            prompt = (
                build_choice_prompt(items, cfg.style)
                + "\nREMINDER: OUTPUT JSON ONLY. NO EXTRA TEXT. Picks must be A/B/C. If unsure pick A.\n"
            )
            time.sleep(0.12)

    return {}


# =========================================================
# What the rest of Subloom calls
# Entry point that runs proofreading over the subtitle list,
# chunk by chunk.
# =========================================================


def ollama_proofread_subtitles(
    cfg: OllamaConfig, merged_subs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not merged_subs:
        return merged_subs

    chunks = chunk_by_time(merged_subs, cfg.window_sec, cfg.max_chars)

    total_chunks = len(chunks)
    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] context-check", 0, total_chunks)

    for chunk_idx, ch in enumerate(chunks, start=1):
        chosen_map = choose_with_context(cfg, ch)

        if chosen_map:
            for s in ch:
                i = int(s["i"])
                if i in chosen_map:
                    s["text"] = chosen_map[i]

        if cfg.show_progress and total_chunks:
            _progress_bar("[ollama] context-check", chunk_idx, total_chunks)

    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] context-check", total_chunks, total_chunks)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return merged_subs
