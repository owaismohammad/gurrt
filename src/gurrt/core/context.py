"""Assemble retrieved frames and transcript into one budgeted timeline.

The job here is not to hand the LLM everything that matched. It is to hand it
a short, coherent, chronological excerpt of the lecture that actually answers
the question. Retrieval returns isolated high-scoring fragments scattered
across the video; on their own they read as disconnected noise and bury the
query. So each hit is expanded into the seconds around it, overlapping
expansions are merged, repeats are dropped, and the whole thing is capped.
"""
from math import ceil

# Progressively tighter context around the best hit, tried in order, so a
# large top window shrinks instead of being dropped for a weaker one.
_PAD_FALLBACKS = (1.0, 0.5, 0.25, 0.0)


def estimate_tokens(text: str) -> int:
    """Rough token count without pulling in a tokenizer.

    ~3.5 chars/token is deliberately pessimistic: equations, code and symbols
    tokenize far worse than prose, and under-filling the window costs nothing
    while overflowing it costs the whole request.
    """
    if not text:
        return 0
    return max(1, ceil(len(text) / 3.5))


def _fmt_ts(sec) -> str:
    if sec is None:
        return "??:??"
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def compress_caption(caption: str) -> str:
    """Drop the schema's empty fields.

    The captioner emits a fixed six-field record, so a speaker-only frame
    still carries four lines of 'NONE'. Across dozens of frames that is a
    meaningful share of the budget spent saying nothing.
    """
    if not caption:
        return ""
    kept = []
    for line in caption.splitlines():
        line = line.strip()
        if not line:
            continue
        _, _, value = line.partition(":")
        if value.strip().upper() in {"NONE", "NONE.", ""}:
            continue
        kept.append(line)
    return " | ".join(kept) if kept else caption.strip()


def merge_intervals(intervals):
    """Collapse overlapping spans so the same seconds are never paid for twice."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(m) for m in merged]


def fetch_pool(db, lo: float, hi: float):
    """Read every candidate event once, covering the whole anchor range.

    Budget-fitting tries many window combinations, so this must not hit the
    DB per attempt; one read per collection, then filter in memory.
    """
    events = []
    seen = set()

    frames = db.frames_in_range(lo, hi)
    for meta in frames.get("metadatas") or []:
        caption = compress_caption(meta.get("caption", ""))
        start = meta.get("start_sec")
        key = ("SHOWN", start, caption)
        if not caption or key in seen:
            continue
        seen.add(key)
        events.append({"kind": "SHOWN", "start": start or 0.0,
                       "end": meta.get("end_sec"), "text": caption})

    audio = db.audio_in_range(lo, hi)
    for doc, meta in zip(audio.get("documents") or [],
                         audio.get("metadatas") or []):
        text = (doc or "").strip()
        start = meta.get("start_sec")
        key = ("SAID", start, text)
        if not text or key in seen:
            continue
        seen.add(key)
        events.append({"kind": "SAID", "start": start or 0.0,
                       "end": meta.get("end_sec"), "text": text})

    events.sort(key=lambda e: (e["start"], e["kind"] != "SHOWN"))
    return events


def build_timeline(anchors, db, token_budget: int, pad_sec: float = 30.0):
    """Expand anchors into windows, then fill the budget best-first.

    `anchors` are dicts with start_sec/end_sec/score, already reranked.
    Windows are admitted whole and in score order, so what survives the budget
    is always the most relevant material and never a sentence cut in half.
    """
    windows = []
    for a in anchors:
        start = a.get("start_sec")
        if start is None:
            continue
        windows.append({
            "score": a.get("score", 0.0),
            "start": start,
            "end": a.get("end_sec") or start,
        })
    if not windows:
        return "", 0

    windows.sort(key=lambda w: w["score"], reverse=True)
    pool = fetch_pool(db,
                      min(w["start"] for w in windows) - pad_sec,
                      max(w["end"] for w in windows) + pad_sec)

    # The best match is always represented. Greedily taking whatever fits lets
    # a weak hit crowd out the strongest one when the strongest is large, which
    # is exactly backwards; shrink its window instead.
    top, accepted = windows[0], None
    for factor in _PAD_FALLBACKS:
        span = _span(top, pad_sec * factor)
        if estimate_tokens(render_spans([span], pool)) <= token_budget:
            accepted = [span]
            break
    if accepted is None:
        # Even the bare anchor overflows. This is the only place trimming is
        # acceptable, because the alternative is returning nothing at all.
        text = _trim_to_budget(render_spans([_span(top, 0.0)], pool),
                               token_budget)
        return text, estimate_tokens(text)

    for w in windows[1:]:
        candidate = merge_intervals(accepted + [_span(w, pad_sec)])
        if estimate_tokens(render_spans(candidate, pool)) <= token_budget:
            accepted = candidate

    text = render_spans(accepted, pool)
    return text, estimate_tokens(text)


def _span(window, pad: float):
    return (max(0.0, window["start"] - pad), window["end"] + pad)


def _trim_to_budget(text: str, token_budget: int) -> str:
    lines, kept, used = text.splitlines(), [], 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > token_budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)


def render_spans(spans, pool) -> str:
    """Lay out every pooled event inside the given spans, chronologically."""
    if not spans:
        return ""

    lines = []
    last_shown = None
    for e in pool:
        start = e["start"]
        if not any(lo <= start <= hi for lo, hi in spans):
            continue
        if e["kind"] == "SHOWN":
            # Consecutive keyframes of one unchanged slide add nothing.
            if e["text"] == last_shown:
                continue
            last_shown = e["text"]
            stamp = _fmt_ts(start)
            end = e.get("end")
            if end and end - start > 1:
                stamp = f"{stamp}-{_fmt_ts(end)}"
            lines.append(f"[{stamp}] SHOWN: {e['text']}")
        else:
            lines.append(f"[{_fmt_ts(start)}] SAID:  {e['text']}")
    return "\n".join(lines)


def format_prior_chat(chat_context, token_budget: int) -> str:
    """Render prior turns as text, newest first, capped.

    The memory client returns a response object; interpolating it straight
    into a prompt dumps an SDK repr of ids and scores, which is pure noise.
    """
    if not chat_context:
        return "None."

    docs = getattr(chat_context, "results", None) or getattr(chat_context, "documents", None)
    if docs is None and isinstance(chat_context, dict):
        docs = chat_context.get("results") or chat_context.get("documents")
    if docs is None:
        docs = chat_context if isinstance(chat_context, list) else []

    texts = []
    for d in docs:
        content = (getattr(d, "content", None)
                   or getattr(d, "memory", None)
                   or (d.get("content") if isinstance(d, dict) else None)
                   or (d.get("memory") if isinstance(d, dict) else None))
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())

    if not texts:
        return "None."

    kept, used = [], 0
    for t in texts:
        cost = estimate_tokens(t)
        if used + cost > token_budget:
            break
        kept.append(t)
        used += cost
    return "\n\n".join(kept) if kept else "None."
