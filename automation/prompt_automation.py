"""Part 2 of the benchmark harness — build the prompts, don't send them.

Given a list of questions, this runs the full retrieval path against the
already-indexed vector DB (embed → search both collections → rerank →
interleave into a timeline) and renders the exact system prompt and user
prompt that `LLMService.query_llm` would hand to the model.

Nothing is sent to OpenRouter. The output is one key/value record per
question, so a run can be inspected, diffed against a previous run, or fed to
whatever model the benchmark is comparing.

Usage:

    python automation/prompt_automation.py -q "what is a kernel?" -q "why softmax?"
    python automation/prompt_automation.py --questions questions.json
    python automation/prompt_automation.py --questions questions.txt --top-k 8

`--questions` accepts either a .txt (one question per line) or a .json — a
bare list of strings, or a list of objects with a "question" key.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gurrt.cli import ui
from gurrt.config.config import Settings
from gurrt.core.context import format_prior_chat
from gurrt.core.debuglog import estimate_tokens
from gurrt.core.llm import format_timeline
from gurrt.core.models import ModelManager
from gurrt.core.prompts import (LLM_QUERY_PROMPT, LLM_SYSTEM_PROMPT,
                                LOW_FIDELITY_VISUAL_NOTE)
from gurrt.core.search import SearchService
from gurrt.core.vectordb import VectorDB
from gurrt.config.benchmark_config import VIDEO_PATH, OUTPUT_PATH, MANIFEST_PATH, CAPTION_PATH, DEFAULT_OUT,QUESTIONS

#DEFAULT_OUT = Path(r"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_4") / "prompts.json"

# What the ask path substitutes when retrieval comes back empty, and what
# format_prior_chat returns for an empty history. Repeated here so a prompt
# built by this script is byte-identical to one built by the app.
NO_CONTEXT = "No indexed content matched this question."
NO_PRIOR_CHAT = "None."


# ── Question loading ──────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[str]:
    """Read questions from a .txt (one per line) or a .json list."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("questions", [])
        questions = []
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and item.get("question"):
                questions.append(item["question"])
        return [q.strip() for q in questions if q.strip()]

    return [line.strip() for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _span(caption_list: list, asr_list: list):
    """[earliest, latest] second covered by everything retrieved."""
    starts = [h.get("start_sec") for h in (*caption_list, *asr_list)
              if h.get("start_sec") is not None]
    ends = [h.get("end_sec") for h in (*caption_list, *asr_list)
            if h.get("end_sec") is not None]
    if not starts:
        return None
    return [round(min(starts), 1), round(max(ends or starts), 1)]


# ── Prompt construction ───────────────────────────────────────────────────────

class PromptBuilder:
    """Holds the retrieval stack open across a whole batch of questions.

    The embedder and the cross-encoder cost seconds to load and are stateless
    once loaded, so they are built once here rather than per question — which
    is the only real difference from what `VideoRag.ask` does per call.
    """

    def __init__(self, settings: Settings, top_k: int, candidate_k: int,
                 with_memory: bool = False):
        self.settings = settings
        self.top_k = top_k
        self.candidate_k = candidate_k

        self.models = ModelManager(settings)
        self.device = self.models.device
        self.search = SearchService(
            text_embedder=self.models.get_text_embedder(),
            reranker=self.models.get_reranker(),
            vectordb=VectorDB(str(settings.CHROMA_DB_PATH)),
            settings=settings,
        )

        # Prior chat is off by default: it makes a question's prompt depend on
        # every question asked before it, which is exactly what a benchmark
        # needs to hold still.
        self.memory = None
        if with_memory:
            from supermemory import Supermemory
            self.memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    def rebind_db(self) -> None:
        """Re-open the collections after someone re-indexed underneath us.

        A reset drops and recreates both Chroma collections, which leaves the
        handles this builder is holding pointing at deleted ones. Only the DB
        is rebuilt — the embedder and cross-encoder are unaffected by a
        re-index and cost seconds to reload, so they stay put.
        """
        self.search.db = VectorDB(str(self.settings.CHROMA_DB_PATH))

    def _prior_chat(self, question: str) -> str:
        if self.memory is None:
            return NO_PRIOR_CHAT
        try:
            return format_prior_chat(self.memory.search.documents(
                q=question, container_tags=["Previous_Chat"], limit=3))
        except Exception as e:
            ui.warn(f"Prior-chat lookup failed, continuing without it: {e}")
            return NO_PRIOR_CHAT

    def build(self, question: str) -> dict:
        """The system/user pair for one question, plus how it was assembled."""
        caption_list, asr_list, stats = self.search.build_context(
            self.device, question,
            top_k=self.top_k, candidate_k=self.candidate_k)

        timeline = format_timeline(caption_list, asr_list)
        low_fidelity = stats["low_fidelity_visual"]

        system_prompt = LLM_SYSTEM_PROMPT
        if low_fidelity:
            system_prompt += LOW_FIDELITY_VISUAL_NOTE

        previous_chat = self._prior_chat(question)
        user_prompt = LLM_QUERY_PROMPT.format(
            timeline=timeline or NO_CONTEXT,
            previous_chat=previous_chat,
            query=question,
        )

        return {
            "question": question,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "retrieval": {
                "top_k": self.top_k,
                "candidate_k": self.candidate_k,
                "frames_retrieved": len(caption_list),
                "transcript_chunks_retrieved": len(asr_list),
                "captioners": stats["captioners"],
                "low_fidelity_visual": low_fidelity,
                # Which stretch of the lecture the excerpt was drawn from —
                # a wide span usually means the search found nothing focused.
                "span_sec": _span(caption_list, asr_list),
                "timeline_est_tokens": estimate_tokens(timeline),
            },
        }

    def close(self) -> None:
        self.models.release_all()


def build_prompts(questions: list[str], top_k: int, candidate_k: int,
                  with_memory: bool = False) -> list[dict]:
    settings = Settings()
    builder = PromptBuilder(settings, top_k=top_k, candidate_k=candidate_k,
                            with_memory=with_memory)
    records = []
    try:
        for i, question in enumerate(questions, 1):
            ui.step(f"[{i}/{len(questions)}] {question}")
            record = builder.build(question)
            r = record["retrieval"]
            ui.info(f"  {r['frames_retrieved']} frames + "
                    f"{r['transcript_chunks_retrieved']} transcript chunks")
            records.append(record)
    finally:
        builder.close()
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render the system/user prompt pair for each question, "
                    "without calling the LLM.")
    p.add_argument("-q", "--question", action="append", default=[],
                   dest="questions", metavar="TEXT",
                   help="A question. Repeat the flag for more than one.")
    p.add_argument("--questions", type=Path, dest="questions_file",
                   metavar="FILE",
                   help="File of questions — .txt (one per line) or .json.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Where to write the prompt JSON (default: {DEFAULT_OUT}).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Hits kept after reranking (default: from Settings).")
    p.add_argument("--candidate-k", type=int, default=None,
                   help="Hits fetched before reranking (default: from Settings).")
    p.add_argument("--with-memory", action="store_true",
                   help="Include Supermemory prior chat in the user prompt. "
                        "Off by default so prompts stay reproducible.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # questions = list(args.questions)

    questions =QUESTIONS

    if args.questions_file:
        if not args.questions_file.exists():
            ui.error(f"Questions file not found: {args.questions_file}")
            return 1
        questions += load_questions(args.questions_file)
    if not questions:
        ui.error("No questions given — use -q TEXT or --questions FILE.")
        return 1

    settings = Settings()
    top_k = args.top_k if args.top_k is not None else settings.ASK_TOP_K
    candidate_k = (args.candidate_k if args.candidate_k is not None
                   else settings.ASK_CANDIDATE_K)

    try:
        records = build_prompts(questions, top_k=top_k,
                                candidate_k=candidate_k,
                                with_memory=args.with_memory)
    except Exception as e:
        ui.error(f"Prompt build failed: {e}")
        return 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_model": settings.LLM_MODEL,
        "retrieval": {"top_k": top_k, "candidate_k": candidate_k},
        "prior_chat_included": args.with_memory,
        "question_count": len(records),
        "prompts": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    ui.success(f"Built {len(records)} prompt pair(s) → {args.out}")

    empty = [r["question"] for r in records
             if r["retrieval"]["frames_retrieved"] == 0
             and r["retrieval"]["transcript_chunks_retrieved"] == 0]
    if empty:
        ui.warn(f"{len(empty)} question(s) retrieved nothing — is the right "
                f"video indexed? First: {empty[0]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())