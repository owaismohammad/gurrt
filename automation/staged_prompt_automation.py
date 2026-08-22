"""Part 2 of the benchmark harness, staged so two model stacks never share VRAM.

`prompt_automation.py` holds the answering GGUF resident for the whole run and
retrieves *inside* that window, so the embedder, the cross-encoder and several
GB of llama-server weights all compete for the same card. This module splits
the work in two, and nothing else about the ask path changes:

    phase 1  embedder + reranker on the GPU, no llama-server running.
             Every question is retrieved and its SHOWN/SAID timeline built and
             kept in memory. Then both models are dropped and the cache freed.

    phase 2  llama-server starts with the card to itself and answers each
             question from its already-built timeline.

The split is safe because a timeline depends only on the question and the
index — neither of which the answering model touches. Prior chat is the one
thing that *does* evolve during the run, so it stays in phase 2, fetched per
question right before the prompt is rendered and written back straight after
the answer, exactly as the REPL does it. Retrieval is deliberately not given
prior chat in the CLI either, so moving it earlier changes nothing.

Everything else is the shipping code path, imported not copied:
`SearchService.build_context`, `format_timeline`, and the prompt strings from
`gurrt.core.prompts`.

Usage:

    python automation/staged_prompt_automation.py --llm-gguf models/qwen2.5-7b-q4.gguf \\
        --questions questions.txt --out runs/qwen.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from supermemory import Supermemory

from gurrt.cli import ui
from gurrt.config.config import Settings
from gurrt.core.context import format_prior_chat
from gurrt.core.debuglog import estimate_tokens, log_query
from gurrt.core.llm import format_timeline
from gurrt.core.models import ModelManager
from gurrt.core.prompts import (LLM_QUERY_PROMPT, LLM_SYSTEM_PROMPT,
                                LOW_FIDELITY_VISUAL_NOTE)
from gurrt.core.search import SearchService
from gurrt.core.vectordb import VectorDB

from llama_chat import LlamaChatServer
from prompt_automation import NO_CONTEXT, load_questions

DEFAULT_OUT = Path("automation_out") / "answers_staged.json"


def _free_vram_gb() -> float | None:
    """Free VRAM right now, or None on a CPU-only box.

    Logged at each phase boundary: if a run does OOM, the useful question is
    how much headroom there was when the next stack loaded, and that is
    invisible after the fact.
    """
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.mem_get_info(0)[0] / 1024 ** 3, 2)


def _log_vram(label: str) -> None:
    free = _free_vram_gb()
    if free is not None:
        ui.info(f"{label}: {free} GB VRAM free")


# ── Phase 1: retrieval ────────────────────────────────────────────────────────

class RetrievalPhase:
    """Embedder + cross-encoder, used to build every timeline up front.

    Loaded once for the whole question set rather than per question — they are
    stateless, so reloading changes how long retrieval takes and never what it
    returns.
    """

    def __init__(self, settings: Settings,
                 top_k: int | None = None, candidate_k: int | None = None):
        self.settings = settings
        self.top_k = top_k if top_k is not None else settings.ASK_TOP_K
        self.candidate_k = (candidate_k if candidate_k is not None
                            else settings.ASK_CANDIDATE_K)

        self.models = ModelManager(settings)
        self.device = self.models.device
        self.text_embedder = self.models.get_text_embedder()
        self.reranker = self.models.get_reranker()
        # Opened without reset=True: this reads whatever index_automation.py
        # already wrote, and must never clear it.
        self.vectordb = VectorDB(str(settings.CHROMA_DB_PATH))
        self.search = SearchService(text_embedder=self.text_embedder,
                                    reranker=self.reranker,
                                    vectordb=self.vectordb,
                                    settings=self.settings)

    def build_one(self, query: str) -> dict:
        """Retrieve for one question and render its timeline.

        Returns everything phase 2 needs, so the retrieval stack can be torn
        down before the answering model is ever started.
        """
        started = time.time()
        caption_list, asr_list, stats = self.search.build_context(
            self.device, query, top_k=self.top_k, candidate_k=self.candidate_k)
        timeline = format_timeline(caption_list, asr_list)
        elapsed = time.time() - started

        low_fidelity = stats["low_fidelity_visual"]
        system_prompt = LLM_SYSTEM_PROMPT
        if low_fidelity:
            system_prompt += LOW_FIDELITY_VISUAL_NOTE

        return {
            "question": query,
            "timeline": timeline,
            "system_prompt": system_prompt,
            "retrieval_sec": round(elapsed, 1),
            "retrieval": {
                "top_k": self.top_k,
                "candidate_k": self.candidate_k,
                "frames_retrieved": len(caption_list),
                "transcript_chunks_retrieved": len(asr_list),
                "captioners": stats["captioners"],
                "low_fidelity_visual": low_fidelity,
                "timeline_est_tokens": estimate_tokens(timeline),
            },
        }

    def build_all(self, questions: list[str]) -> list[dict]:
        prepared = []
        for i, question in enumerate(questions, 1):
            ui.step(f"[retrieve {i}/{len(questions)}] {question[:70]}")
            item = self.build_one(question)
            r = item["retrieval"]
            if not item["timeline"]:
                ui.warn("    nothing retrieved — this question will be asked "
                        "without context")
            else:
                ui.info(f"    {r['frames_retrieved']} frames + "
                        f"{r['transcript_chunks_retrieved']} chunks "
                        f"(~{r['timeline_est_tokens']} tok) in "
                        f"{item['retrieval_sec']}s")
            prepared.append(item)
        return prepared

    def release(self) -> None:
        """Drop both models so the answering GGUF gets the card to itself.

        `ModelManager.release_all` clears its own handles, but the ones held
        here and inside SearchService keep the weights alive; the collect has
        to happen before empty_cache or there is nothing to reclaim.
        """
        self.search = None
        self.vectordb = None
        self.text_embedder = None
        self.reranker = None
        self.models.release_all()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# ── Phase 2: answering ────────────────────────────────────────────────────────

class AnswerPhase:
    """Turns prepared timelines into answers, one llama-server for the set.

    Prior chat is fetched and written back per question rather than batched:
    `LLM_QUERY_PROMPT` has an `EARLIER IN THIS SESSION` slot, and question N
    can only see question N-1's answer if that answer was stored before this
    one's lookup runs.
    """

    def __init__(self, settings: Settings, chat_server: LlamaChatServer):
        self.settings = settings
        self.chat_server = chat_server
        self.client_memory = Supermemory(api_key=settings.SUPERMEMORY_API_KEY)

    def _prior_chat(self, query: str) -> str:
        """Whatever earlier Q&A Supermemory considers relevant to this one."""
        try:
            return format_prior_chat(self.client_memory.search.documents(
                q=query, container_tags=["Previous_Chat"], limit=3))
        except Exception as e:
            # The CLI would crash here. In a long benchmark run a transient
            # memory-service error should cost one question its history, not
            # the whole run.
            ui.warn(f"    prior-chat lookup failed, continuing without it: {e}")
            return "None."

    def clear_chat(self) -> None:
        """Drop stored prior chat, as a `reset=True` index does."""
        try:
            self.client_memory.documents.delete_bulk(
                container_tags=["Previous_Chat"])
            ui.info("Cleared stored prior chat")
        except Exception as e:
            ui.warn(f"Could not clear prior chat: {e}")

    def answer(self, item: dict) -> dict:
        """Render the user prompt for one prepared question and answer it."""
        query = item["question"]
        system_prompt = item["system_prompt"]

        previous_chat = self._prior_chat(query)
        user_prompt = LLM_QUERY_PROMPT.format(
            timeline=item["timeline"] or NO_CONTEXT,
            previous_chat=previous_chat,
            query=query,
        )

        started = time.time()
        answer, error = None, None
        try:
            answer = self.chat_server.chat(system_prompt, user_prompt)
        except Exception as e:
            error = str(e)
        elapsed = time.time() - started

        log_query(self.settings,
                  query=query,
                  system_prompt=system_prompt,
                  timeline=item["timeline"],
                  previous_chat=previous_chat,
                  rendered_human=user_prompt,
                  answer=answer or f"[FAILED] {error}",
                  low_fidelity_visual=item["retrieval"]["low_fidelity_visual"])

        # Only real answers go back into memory. Storing an error string would
        # feed it to the next question as if it were prior teaching.
        if answer:
            try:
                self.client_memory.add(content=f"{query}\n\n\n{answer}",
                                       container_tags=["Previous_Chat"],
                                       metadata={"note_id": "Retrieved Chat"})
            except Exception as e:
                ui.warn(f"    could not store this turn in memory: {e}")

        return {
            "question": query,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "previous_chat": previous_chat,
            "answer": answer,
            "error": error,
            "retrieval_sec": item["retrieval_sec"],
            "answer_sec": round(elapsed, 1),
            "answering_model": self.chat_server.model_name,
            "retrieval": item["retrieval"],
        }

    def answer_all(self, prepared: list[dict]) -> list[dict]:
        records = []
        for i, item in enumerate(prepared, 1):
            ui.step(f"[answer {i}/{len(prepared)}] {item['question'][:70]}")
            record = self.answer(item)
            if record["error"]:
                ui.error(f"    no answer: {record['error'][:150]}")
            else:
                ui.info(f"    answered in {record['answer_sec']}s "
                        f"({len(record['answer'])} chars)")
            records.append(record)
        return records


# ── Run ───────────────────────────────────────────────────────────────────────

def run(questions: list[str], gguf: Path, out: Path,
        top_k=None, candidate_k=None, ctx_size: int = 8192,
        max_tokens: int = 1024, clear_memory: bool = False) -> list[dict]:
    settings = Settings()

    ui.step(f"Phase 1: building {len(questions)} timeline(s)")
    _log_vram("Before retrieval models")
    retrieval = RetrievalPhase(settings, top_k=top_k, candidate_k=candidate_k)
    retrieval_started = time.time()
    try:
        prepared = retrieval.build_all(questions)
    finally:
        # Released even on failure: leaving the embedder resident after a
        # crash would hand the same VRAM problem to the next run.
        retrieval.release()
    retrieval_sec = time.time() - retrieval_started
    _log_vram("After releasing retrieval models")
    ui.success(f"Timelines built in {round(retrieval_sec, 1)}s — retrieval "
               "models released")

    ui.step(f"Phase 2: answering with {gguf.stem}")
    answer_started = time.time()
    with LlamaChatServer(gguf, ctx_size=ctx_size,
                         max_tokens=max_tokens) as server:
        _log_vram("With answering model resident")
        answering = AnswerPhase(settings, server)
        if clear_memory:
            answering.clear_chat()
        records = answering.answer_all(prepared)
    answer_sec = time.time() - answer_started

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answering_model": gguf.stem,
        "answering_model_path": str(gguf),
        "question_count": len(records),
        "answered": sum(1 for r in records if r["answer"]),
        "phase_timings_sec": {
            "retrieval": round(retrieval_sec, 1),
            "answering": round(answer_sec, 1),
        },
        "results": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ask questions against the current index in two phases — "
                    "retrieve everything first, then load the answering GGUF — "
                    "so the two model stacks never hold VRAM at once.")
    p.add_argument("--llm-gguf", type=Path, required=True,
                   help="GGUF used to answer. Swap this to benchmark models.")
    # p.add_argument("-q", "--question", action="append", default=[],
    #                dest="questions", metavar="TEXT",
    #                help="A question. Repeat the flag for more than one.")
    # p.add_argument("--questions", type=Path, dest="questions_file",
    #                metavar="FILE",
                #    help="File of questions — .txt (one per line) or .json.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Where results are written (default: {DEFAULT_OUT}).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Hits kept after reranking (default: from Settings).")
    p.add_argument("--candidate-k", type=int, default=None,
                   help="Hits fetched before reranking (default: from Settings).")
    p.add_argument("--ctx-size", type=int, default=8192,
                   help="llama-server context window (default: 8192).")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max tokens per answer (default: 1024).")
    p.add_argument("--clear-memory", action="store_true",
                   help="Delete stored prior chat before answering, so the "
                        "run starts with an empty session history.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    questions = [
    "Why is the term 'naive' used? How does the algorithm's assumption of word independence define its name?",
    "How does supervised learning apply here? Since labels A and B are hidden, what role does the training data play in revealing these concepts?",
    "What defines the 'evidence' for each word? How is the probability of a specific word appearing in a document calculated for each label?",
    "What is the significance of the 'prior'? How does the prior probability influence the final classification compared to the evidence gathered from the words?",
    "What happens if a word has never been seen before? How does the algorithm handle new words that weren't in the training set?",
    "Why is word order ignored? How does the 'bag of words' approach simplify the classification process compared to models that analyze grammar or syntax?",
    "How does the algorithm handle varying document lengths? Does a longer text with more words provide more confidence in the classification, and if so, why?",
    "What if both labels use the same words with the same probability? Is there any way for the model to make a prediction in such a scenario?",
    "How is the ratio used for decision-making? After multiplying all evidences, how exactly does the final ratio map to a specific label choice?",
    "What makes this 'powerful' in real-world applications? Given its simplified assumptions, why is it still widely used for tasks like spam detection or authorship attribution?"
]
    # if args.questions_file:
    #     if not args.questions_file.exists():
    #         ui.error(f"Questions file not found: {args.questions_file}")
    #         return 1
    #     questions += load_questions(args.questions_file)
    if not questions:
        ui.error("No questions given — use -q TEXT or --questions FILE.")
        return 1

    # Checked before phase 1 so a typo in the path costs nothing: retrieving
    # a full question set only to fail at server start wastes all of it.
    if not args.llm_gguf.exists():
        ui.error(f"GGUF not found: {args.llm_gguf}")
        return 1

    try:
        records = run(questions, args.llm_gguf, args.out,
                      top_k=args.top_k,
                      candidate_k=args.candidate_k,
                      ctx_size=args.ctx_size,
                      max_tokens=args.max_tokens,
                      clear_memory=args.clear_memory)
    except Exception as e:
        ui.error(f"Run failed: {e}")
        return 1

    answered = sum(1 for r in records if r["answer"])
    ui.success(f"{answered}/{len(records)} answered → {args.out}")
    return 0 if answered == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
