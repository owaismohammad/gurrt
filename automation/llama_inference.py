"""Part 2+3 of the benchmark harness, combined: build context, then answer.

Given a `questions.csv` (any columns — a `question` column, or the first
column if none is named that, drives retrieval; every other column is passed
through untouched to the output), this:

  phase 1  retrieves context for each question against the already-indexed
           vector DB (embed -> search both collections -> rerank ->
           timeline), renders the same system/user prompt pair
           `LLMService.query_llm` would, and writes them to prompts.json.
           The embedder and cross-encoder are then released from the GPU —
           they can't share VRAM with the answering server started next.

  phase 2  starts llama-server and answers every prompt concurrently, same
           as before. Failures (timeouts, non-200s, empty completions) are
           retried with backoff; a question that still has nothing after
           every retry is never dropped — it gets a row with
           "ERROR: <reason>" in the response column instead of vanishing.

Usage:

    llama_inference(questions_csv=Video_ID / "questions.csv",
                    output_dir=Video_ID,
                    max_workers=10)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from platformdirs import user_config_dir

# Lets `from prompt_automation import PromptBuilder` resolve no matter how
# this module is invoked (direct script run vs. imported by index_automation).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_automation import PromptBuilder

from gurrt.config.config import LlamaServerManager, Settings
from gurrt.utils.llama_server_utils import wait_for_server

home = Path(user_config_dir("gurrt"))

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = (5, 15, 30)
CHAT_PORT = 8080


# ── Question loading ──────────────────────────────────────────────────────────

def _load_questions_csv(questions_csv: Path) -> tuple[pd.DataFrame, str]:
    """Read the CSV, pick the question column, drop blank/duplicate questions.

    Every other column is left as-is on the returned DataFrame so it can be
    carried through to the output untouched.
    """
    df = pd.read_csv(questions_csv)
    if df.empty:
        raise ValueError(f"{questions_csv} has no rows")

    question_col = "question" if "question" in df.columns else df.columns[0]
    df[question_col] = df[question_col].astype(str).str.strip()
    df = df[df[question_col] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{questions_csv} has no non-empty questions")

    dupes = df[question_col].duplicated()
    if dupes.any():
        print(f"[WARN] {dupes.sum()} duplicate question(s) in {questions_csv.name} — "
              f"they will share one answer.")

    return df, question_col


# ── Phase 1: context + prompt construction ─────────────────────────────────────

def build_prompts(df: pd.DataFrame, question_col: str, output_dir: Path,
                  top_k: int | None = None, candidate_k: int | None = None) -> dict:
    """Retrieve context for every question and render its prompt pair.

    Runs the embedder/cross-encoder stack, writes prompts.json, then frees
    the GPU before the caller starts llama-server.
    """
    questions = list(dict.fromkeys(df[question_col].tolist()))

    settings = Settings()
    builder = PromptBuilder(
        settings,
        top_k=top_k if top_k is not None else settings.ASK_TOP_K,
        candidate_k=candidate_k if candidate_k is not None else settings.ASK_CANDIDATE_K,
        with_memory=False,
    )
    records = []
    try:
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] building context: {question[:70]}")
            record = builder.build(question)
            r = record["retrieval"]
            if r["frames_retrieved"] == 0 and r["transcript_chunks_retrieved"] == 0:
                print(f"  [WARN] nothing retrieved for: {question[:70]}")
            records.append(record)
    finally:
        builder.close()

    payload = {
        "target_model": settings.LLM_MODEL,
        "retrieval": {"top_k": builder.top_k, "candidate_k": builder.candidate_k},
        "question_count": len(records),
        "prompts": records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.json"
    prompts_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"Wrote {len(records)} prompt pair(s) -> {prompts_path}")
    return payload


# ── Phase 2: concurrent inference ──────────────────────────────────────────────

def ask_question(rec: dict, port: int, timeout: int = 300) -> tuple[str, str, bool]:
    """Send one question to the server, retrying on failure or an empty answer.

    Returns (question, response_text, ok). `ok` is False only once every
    retry is exhausted; `response_text` is then an "ERROR: ..." string so the
    question still gets a row instead of silently vanishing.
    """
    question = rec["question"]
    request_body = {
        "model": "gemma-4-12b-it",
        "messages": [
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["user_prompt"]},
        ],
        "temperature": 0.0,
    }

    last_error = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json=request_body, timeout=timeout,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                if text and text.strip():
                    return question, text, True
                last_error = "empty response from model"
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as ex:
            last_error = str(ex)

        if attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF_SEC[min(attempt - 1, len(RETRY_BACKOFF_SEC) - 1)]
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {question[:60]}...: {last_error} "
                  f"(retrying in {backoff}s)")
            time.sleep(backoff)

    print(f"[FAILED] {question[:60]}...: {last_error} (exhausted {MAX_RETRIES} attempts)")
    return question, f"ERROR: {last_error}", False


def run_inference(prompts_payload: dict, df: pd.DataFrame, question_col: str,
                  output_dir: Path, max_workers: int = 10,
                  port: int = CHAT_PORT) -> Path:
    """Start llama-server and answer every prompt in `prompts_payload` concurrently."""
    overall_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "response.csv"

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        answers = dict(zip(existing_df[question_col], existing_df["response"]))
        # Re-attempt anything that failed on a previous run instead of resuming it as-is.
        answers = {q: a for q, a in answers.items()
                  if not (isinstance(a, str) and a.startswith("ERROR:"))}
    else:
        answers = {}

    total_q = len(prompts_payload["prompts"])
    pending = [rec for rec in prompts_payload["prompts"] if rec["question"] not in answers]
    print(f"\n=== {total_q} questions total, {total_q - len(pending)} already answered, "
          f"{len(pending)} to send ===")

    def _write_csv() -> None:
        merged = df.copy()
        merged["response"] = merged[question_col].map(lambda q: answers.get(q, ""))
        merged.to_csv(output_csv_path, index=False)

    if not pending:
        _write_csv()
        return output_csv_path

    llama_server_manager = LlamaServerManager(home)
    cmd = [
        str(llama_server_manager.server_bin),
        "-m", str(llama_server_manager.inference_llm_path),
        "-ngl", "99",
        "--parallel", str(max_workers),
        "-c", str(8192 * max_workers),
        "--port", str(port),
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
    ]

    server_env = os.environ.copy()
    if sys.platform != "win32":
        bin_dir = str(llama_server_manager.server_bin.parent)
        server_env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + server_env.get("LD_LIBRARY_PATH", "")

    process_query = subprocess.Popen(cmd, env=server_env)

    try:
        if not wait_for_server():
            raise RuntimeError("llama-server did not become healthy in time")
        print("\nInference Engine Ready")

        processed_this_run = 0
        failed_this_run = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(ask_question, rec, port): rec for rec in pending}

            for future in as_completed(futures):
                rec = futures[future]
                question_text, answer_text, ok = future.result()

                answers[question_text] = answer_text
                processed_this_run += 1
                if not ok:
                    failed_this_run += 1

                elapsed_total = time.time() - overall_start
                status = "Answered" if ok else "FAILED"
                print(f"[{processed_this_run}/{len(pending)}] {status}: "
                      f"{rec['question'][:60]}... (total elapsed: {elapsed_total/60:.1f}m)")

                _write_csv()  # every question lands on disk as soon as it settles

        total_elapsed = time.time() - overall_start
        print(f"\n=== DONE ===")
        print(f"Questions: {total_q}")
        print(f"Answered this run: {processed_this_run - failed_this_run}")
        print(f"Failed (after {MAX_RETRIES} retries each): {failed_this_run}")
        print(f"Output: {output_csv_path}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        return output_csv_path
    finally:
        print("Cleaning up server process...")
        process_query.terminate()
        try:
            process_query.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Server did not exit after terminate(), killing it.")
            process_query.kill()
            process_query.wait()


# ── Entry point ───────────────────────────────────────────────────────────────

def llama_inference(questions_csv: Path, output_dir: Path,
                    max_workers: int = 10,
                    top_k: int | None = None,
                    candidate_k: int | None = None,
                    port: int = CHAT_PORT) -> Path:
    """Answer every question in `questions_csv` against the currently indexed video.

    Phase 1 retrieves context per question (embedder + reranker on GPU) and
    writes prompts.json next to the output. Phase 2 starts llama-server and
    answers every prompt concurrently. Every input column is preserved in the
    output response.csv; a `response` column is appended, holding either the
    model's answer or, if every retry was exhausted, an "ERROR: ..." string —
    no question is ever silently skipped.
    """
    questions_csv = Path(questions_csv)
    output_dir = Path(output_dir)

    df, question_col = _load_questions_csv(questions_csv)
    prompts_payload = build_prompts(df, question_col, output_dir,
                                    top_k=top_k, candidate_k=candidate_k)
    return run_inference(prompts_payload, df, question_col, output_dir,
                        max_workers=max_workers, port=port)


# if __name__ == "__main__":
#     llama_inference(questions_csv=Path("questions.csv"), output_dir=Path("automation_out"))
