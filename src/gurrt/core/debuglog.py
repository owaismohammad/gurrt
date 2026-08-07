"""Write what the pipeline produced to disk, for inspection.

Caption and retrieval quality are hard to judge from an answer alone: a weak
answer could be bad captions, bad retrieval, or a bad prompt, and there is no
way to tell them apart after the fact. These files make each stage readable on
its own.

Nothing here is load-bearing. Every writer swallows its own errors, because a
logging failure must never take down an index or a query.
"""
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from gurrt.core.context import estimate_tokens
from gurrt.cli import ui


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _video_dir(settings, video_path) -> Path:
    """One directory per video. The path hash keeps same-named files apart."""
    p = Path(str(video_path))
    digest = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", p.stem)[:60].strip("-") or "video"
    d = settings.LOGS_DIR / f"{slug}-{digest}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def log_captions(settings, video_path, metadatas, ids) -> None:
    """Every indexed frame, in time order, with its caption."""
    try:
        rows = []
        for i, meta in enumerate(metadatas):
            caption = meta.get("caption", "")
            rows.append({
                "id": ids[i] if i < len(ids) else None,
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "duration_sec": (
                    round(meta["end_sec"] - meta["start_sec"], 2)
                    if meta.get("end_sec") is not None
                    and meta.get("start_sec") is not None else None),
                "caption": caption,
                "caption_chars": len(caption),
                "est_tokens": estimate_tokens(caption),
            })
        rows.sort(key=lambda r: r["start_sec"] if r["start_sec"] is not None else 0)

        out = _video_dir(settings, video_path) / "captions.json"
        _write(out, {
            "video": str(video_path),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(rows),
            "total_est_tokens": sum(r["est_tokens"] for r in rows),
            "frames": rows,
        })
        ui.info(f"Caption log: {out}")
    except Exception as e:
        ui.warn(f"Could not write caption log: {e}")


def log_transcript(settings, video_path, chunked_text, metadatas, ids) -> None:
    """Every transcript chunk with the span of video it came from."""
    try:
        rows = []
        for i, text in enumerate(chunked_text):
            meta = metadatas[i] if i < len(metadatas) else {}
            rows.append({
                "id": ids[i] if i < len(ids) else None,
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "text": text,
                "chars": len(text),
                "est_tokens": estimate_tokens(text),
            })

        out = _video_dir(settings, video_path) / "transcript.json"
        _write(out, {
            "video": str(video_path),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(rows),
            "total_est_tokens": sum(r["est_tokens"] for r in rows),
            "chunks": rows,
        })
        ui.info(f"Transcript log: {out}")
    except Exception as e:
        ui.warn(f"Could not write transcript log: {e}")


def log_query(settings, query, system_prompt, timeline, previous_chat,
              rendered_human, answer) -> None:
    """Exactly what was sent to the LLM for one question, and what came back."""
    try:
        d = settings.LOGS_DIR / "queries"
        d.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^A-Za-z0-9]+", "-", query.lower())[:50].strip("-") or "query"
        out = d / f"{_utc_stamp()}-{slug}.json"

        timeline_tokens = estimate_tokens(timeline)
        chat_tokens = estimate_tokens(previous_chat)
        system_tokens = estimate_tokens(system_prompt)
        human_tokens = estimate_tokens(rendered_human)

        shown = sum(1 for l in timeline.splitlines() if "] SHOWN:" in l)
        said = sum(1 for l in timeline.splitlines() if "] SAID:" in l)

        _write(out, {
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "model": settings.LLM_MODEL,
            "budgets": {
                "context_token_budget": settings.CONTEXT_TOKEN_BUDGET,
                "chat_token_budget": settings.CHAT_TOKEN_BUDGET,
                "window_pad_sec": settings.WINDOW_PAD_SEC,
            },
            "usage": {
                "timeline_tokens": timeline_tokens,
                "chat_tokens": chat_tokens,
                "system_tokens": system_tokens,
                "human_message_tokens": human_tokens,
                "total_input_tokens": system_tokens + human_tokens,
                "timeline_budget_used_pct": (
                    round(100 * timeline_tokens / settings.CONTEXT_TOKEN_BUDGET, 1)
                    if settings.CONTEXT_TOKEN_BUDGET else None),
                "shown_lines": shown,
                "said_lines": said,
            },
            "context": {
                "system_prompt": system_prompt,
                "timeline": timeline,
                "previous_chat": previous_chat,
                "rendered_human_message": rendered_human,
            },
            "answer": answer,
        })
        ui.info(f"Query log: {out}")
    except Exception as e:
        ui.warn(f"Could not write query log: {e}")
