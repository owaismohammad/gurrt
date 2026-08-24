import io
import base64
import asyncio
import time
import aiohttp
from typing import List, Dict, Any
import requests
import json
from gurrt.utils.utils import temporal_persistence_filter
from pathlib import Path
from huggingface_hub import hf_hub_download
from gurrt.config.config import LlamaServerManager
from gurrt.core.prompts import GEMMA_CAPTION_PROMPT
from gurrt.cli import ui
from gurrt.utils.downloads import watch_download, hf_file_size



def _convert_pil_to_base64(pil_img) -> str:
    """Converts a PIL image object to a base64 string completely in memory."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90, subsampling=0)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def _caption_single_frame_worker(
    session: aiohttp.ClientSession, 
    b64_image: str, 
    index: int, 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Sends an individual base64 string to the running local Gemma 3 engine."""
    server_url = "http://localhost:8080/v1/chat/completions"
    
    payload = {
        "model": "gemma-4-e4b-it", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text":GEMMA_CAPTION_PROMPT

                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ],
        "temperature": 0.0
    }
    
    async with semaphore:
        try:
            async with session.post(server_url, json=payload, timeout=900) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    caption = result["choices"][0]["message"]["content"]
                    return {"index": index, "text": caption, "success": True}
                else:
                    ui.warn(f"Engine error on frame {index}: HTTP {resp.status}")
                    return {"index": index, "text": "Error: Failed to generate description.", "success": False}
        except Exception as e:
            ui.error(f"Server timeout on frame {index}: {e}")
            return {"index": index, "text": "Error: Pipeline connection exception.", "success": False}

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SEC = 2

def batch_caption_frames(frame_list: list, concurrency_limit: int ) -> List[Dict[str, Any]]:
    total = len(frame_list)

    with ui.make_progress() as progress:
        task_id = progress.add_task("  Captioning frames", total=total)

        async def run_pipeline():
            semaphore = asyncio.Semaphore(concurrency_limit)
            tasks = []
            b64_by_index: Dict[int, str] = {}

            async def tracked_worker(session, b64_str, idx):
                result = await _caption_single_frame_worker(session, b64_str, idx, semaphore)
                progress.advance(task_id)
                return result

            async with aiohttp.ClientSession() as session:
                for idx, pil_frame in enumerate(frame_list):
                    try:
                        b64_str = _convert_pil_to_base64(pil_frame)
                        b64_by_index[idx] = b64_str
                        tasks.append(asyncio.create_task(tracked_worker(session, b64_str, idx)))
                    except Exception as e:
                        ui.warn(f"Skipping corrupt frame {idx}: {e}")

                results = await asyncio.gather(*tasks)
                results_by_index = {r["index"]: r for r in results if r is not None}

                # Failures are almost always the client's own request queueing
                # behind the server's single-threaded vision encoder, not a bad
                # frame — so retrying serially (no contention) against the now
                # idle server recovers nearly all of them instead of poisoning
                # the index with an "Error: ..." placeholder caption.
                failed_indices = [i for i, r in results_by_index.items() if not r["success"]]
                if failed_indices:
                    ui.step(f"Retrying {len(failed_indices)} frame(s) that failed captioning...")
                    retry_semaphore = asyncio.Semaphore(1)
                    for idx in failed_indices:
                        for attempt in range(1, RETRY_ATTEMPTS + 1):
                            result = await _caption_single_frame_worker(
                                session, b64_by_index[idx], idx, retry_semaphore
                            )
                            if result["success"]:
                                results_by_index[idx] = result
                                ui.success(f"Frame {idx} captioned on retry {attempt}")
                                break
                            if attempt < RETRY_ATTEMPTS:
                                await asyncio.sleep(RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
                        else:
                            ui.error(f"Frame {idx} failed after {RETRY_ATTEMPTS} retries — dropping from index")
                            del results_by_index[idx]

            results = sorted(results_by_index.values(), key=lambda x: x["index"])
            return results

        return asyncio.run(run_pipeline())

def wait_for_server():
    ui.step("Waiting for captioning server to start...")
    for attempt in range(40):
        try:
            if requests.get("http://localhost:8080/health", timeout=1).status_code == 200:
                ui.success("Captioning server ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.5)
    return False


def process_video(video_path):
    ui.step("Scanning video for scene changes...")
    return temporal_persistence_filter(video_path=video_path)


def download_gemma3_models(models_dir: Path,
                        #    config_dir:Path,
                           llama_server_manager: LlamaServerManager):
    """
    Sequentially downloads Gemma 3 model weights and its associated 
    multimodal vision projector from Hugging Face Hub.
    """

    (llama_server_manager.models_dir).mkdir(exist_ok=True, parents=True)
    #enable_progress_bars()  
    # llama_server_manager = LlamaServerManager(config_dir= config_dir)
    huggingface_repo = llama_server_manager.hf_repo
    files = [
        llama_server_manager.model_filename, 
        llama_server_manager.mmproj_filename
    ]

    for filename in files:
        target_path = models_dir / filename

        if target_path.exists():
            ui.info(f"{filename} already present, skipping")
            continue

        watch_download(
            description=f"  {filename}",
            worker=lambda f=filename: hf_hub_download(
                repo_id=huggingface_repo,
                filename=f,
                local_dir=str(models_dir),
            ),
            watch_dir=models_dir,
            total_bytes=hf_file_size(huggingface_repo, filename),
        )
        ui.success(f"Downloaded {filename}")

def download_gemma4_inference_model(models_dir: Path,
                        #    config_dir:Path,
                           llama_server_manager: LlamaServerManager):
    """
    Downloads Gemma 4 inference model weights from Hugging Face Hub.
    """

    (llama_server_manager.models_dir).mkdir(exist_ok=True, parents=True)
    #enable_progress_bars()  
    # llama_server_manager = LlamaServerManager(config_dir= config_dir)
    huggingface_repo = llama_server_manager.inference_hf_repo
    filename = llama_server_manager.inference_model_filename

    target_path = models_dir / filename

    if target_path.exists():
        ui.info(f"{filename} already present, skipping")
        return

    watch_download(
        description=f"  {filename}",
        worker=lambda f=filename: hf_hub_download(
            repo_id=huggingface_repo,
            filename=f,
            local_dir=str(models_dir),
        ),
        watch_dir=models_dir,
        total_bytes=hf_file_size(huggingface_repo, filename),
    )
    ui.success(f"Downloaded {filename}")