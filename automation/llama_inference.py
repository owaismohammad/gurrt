import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from gurrt.config.benchmark_config import OUTPUT_PATH, DEFAULT_OUT, RESPONSE_PATH
from gurrt.config.config import LlamaServerManager
from gurrt.utils.llama_server_utils import wait_for_server
from platformdirs import user_config_dir

home = Path(user_config_dir("gurrt"))

def ask_question(rec, timeout=300):
    """Send one question to the server and return (question, answer_text_or_None, error_or_None)."""
    question = rec["question"]
    request_body = {
        "model": "gemma-3-4b-it",
        "messages": [
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["user_prompt"]},
        ],
        "temperature": 0.0
    }
    try:
        resp = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json=request_body, timeout=timeout
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            return question, text, None
        else:
            return question, None, f"HTTP {resp.status_code}"
    except Exception as ex:
        return question, None, str(ex)


def llama_inference(prompt_json_path: Path = DEFAULT_OUT,
                     output_dir: Path = OUTPUT_PATH,
                     max_workers: int = 10,
                     ):
    overall_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "response.csv"

    prompts_payload = json.loads(prompt_json_path.read_text(encoding="utf-8"))

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        answers = dict(zip(existing_df["question"], existing_df["response"]))
    else:
        answers = {}

    total_q = len(prompts_payload["prompts"])
    already_done = sum(1 for rec in prompts_payload["prompts"] if rec["question"] in answers)

    print(f"\n=== {prompt_json_path.name} | {total_q} questions total, "
          f"{already_done} already answered ===")

    processed_this_run = 0
    llama_server_manager = LlamaServerManager(home)

    cmd = [
        str(llama_server_manager.server_bin),
        "-m", str(llama_server_manager.llm_path),
        # "--mmproj", str(llama_server_manager.mmproj_path),
        "-ngl", "99",
        "--parallel", str(max_workers),
        "-c", str(8192 * max_workers),
        "--port", "8080",
        # "-n", "320",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
    ]

    import os
    import sys
    server_env = os.environ.copy()
    if sys.platform != "win32":
        bin_dir = str(llama_server_manager.server_bin.parent)
        server_env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + server_env.get("LD_LIBRARY_PATH", "")

    process_query = subprocess.Popen(cmd, env=server_env)

    try:
        wait_for_server()
    except Exception as e:
        print(f"Error during server startup: {e}")
        process_query.terminate()
        return
    print("\nInference Engine Ready")

    try:
        pending = [rec for rec in prompts_payload["prompts"] if rec["question"] not in answers]
        print(f"Sending {len(pending)} questions with up to {max_workers} concurrent requests...\n")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(ask_question, rec): rec for rec in pending}

            for future in as_completed(futures):
                rec = futures[future]
                question = rec["question"]
                q_start = futures.get("_unused", None)  # placeholder, not used for timing per-request start
                question_text, answer_text, error = future.result()

                if error is not None:
                    print(f"[ERROR] {question[:60]}...: {error}")
                    continue

                answers[question_text] = answer_text
                processed_this_run += 1
                elapsed_total = time.time() - overall_start
                print(f"[{processed_this_run}/{len(pending)}] Answered: {question[:60]}... "
                      f"(total elapsed: {elapsed_total/60:.1f}m)")

        # Save CSV once, after all questions are processed
        pd.DataFrame(
            [{"question": rec["question"], "response": answers.get(rec["question"], "")}
             for rec in prompts_payload["prompts"]],
            columns=["question", "response"]
        ).to_csv(output_csv_path, index=False)

        total_elapsed = time.time() - overall_start
        print(f"\n=== DONE ===")
        print(f"Questions: {total_q}")
        print(f"Answered this run: {processed_this_run}")
        print(f"Output: {output_csv_path}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
    finally:
        print("Cleaning up server process...")
        process_query.terminate()
        process_query.wait()

if __name__ == "__main__":
    llama_inference()