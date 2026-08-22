import time
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests

from gurrt.config.benchmark_config import OUTPUT_PATH, DEFAULT_OUT, RESPONSE_PATH
from gurrt.config.config import LlamaServerManager
from gurrt.utils.llama_server_utils import wait_for_server


def llama_inference(prompt_json_path: Path = DEFAULT_OUT,
                     output_dir: Path = OUTPUT_PATH,
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
    llama_server_manager = LlamaServerManager()

    cmd = [
        str(llama_server_manager.server_bin),
        "-m", str(llama_server_manager.llm_path),
        "--mmproj", str(llama_server_manager.mmproj_path),
        "-ngl", "99",
        "--parallel", "1",
        "-c", "8192",
        "--port", "8080",
    ]

    process_query = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
    except Exception as e:
        print(f"Error during server startup: {e}")
        process_query.terminate()
        return
    print("\nInference Engine Ready")

    try:
        for k, rec in enumerate(prompts_payload["prompts"]):
            question = rec["question"]
            print(f"\n[Q{k+1}/{total_q}] {question}")

            if question in answers:
                continue

            q_start = time.time()
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
                    json=request_body, timeout=45
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                else:
                    print(f"[ERROR] Q{k+1}/{total_q}: HTTP {resp.status_code}")
                    continue
            except Exception as ex:
                print(f"[ERROR] Q{k+1}/{total_q}: {ex}")
                continue

            answers[question] = text
            processed_this_run += 1

            # Incrementally save so progress isn't lost
            pd.DataFrame(
                [{"question": q, "response": answers[q]} for q in answers],
                columns=["question", "response"]
            ).to_csv(output_csv_path, index=False)

            elapsed_q = time.time() - q_start
            elapsed_total = time.time() - overall_start
            print(f"[Q{k+1}/{total_q}] -> {elapsed_q:.1f}s (total elapsed: {elapsed_total/60:.1f}m)")

        # Final CSV
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