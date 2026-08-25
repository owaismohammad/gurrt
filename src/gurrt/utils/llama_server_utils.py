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
from gurrt.core.prompts import GEMMA_CAPTION_PROMPT, VLM_SYSTEM_PROMPT
from gurrt.cli import ui
from gurrt.utils.downloads import watch_download, hf_file_size



def _convert_pil_to_base64(pil_img) -> str:
    """Converts a PIL image object to a base64 string completely in memory."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90, subsampling=0)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# async def _caption_single_frame_worker(
#     session: aiohttp.ClientSession, 
#     b64_image: str, 
#     index: int, 
#     semaphore: asyncio.Semaphore
# ) -> Dict[str, Any]:
#     """Sends an individual base64 string to the running local Gemma 3 engine."""
#     server_url = "http://localhost:8080/v1/chat/completions"
    
#     payload = {
#         "model": "gemma-3-4b-it", 
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text", 
#                         "text":GEMMA_CAPTION_PROMPT

#                     },
#                     {
#                         "type": "image_url", 
#                         "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
#                     }
#                 ]
#             }
#         ],
#         "temperature": 0.0
#     }
    
#     async with semaphore:
#         try:
#             async with session.post(server_url, json=payload, timeout=300) as resp:
#                 if resp.status == 200:
#                     result = await resp.json()
#                     caption = result["choices"][0]["message"]["content"]
#                     return {"index": index, "text": caption}
#                 else:
#                     ui.warn(f"Engine error on frame {index}: HTTP {resp.status}")
#                     return {"index": index, "text": "Error: Failed to generate description."}
#         except Exception as e:
#             ui.error(f"Server timeout on frame {index}: {e}")
#             return {"index": index, "text": "Error: Pipeline connection exception."}
async def _server_alive(session: aiohttp.ClientSession) -> bool:
    """Quick health check to distinguish 'transient hiccup' from 'server is down'."""
    try:
        async with session.get("http://localhost:8080/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _wait_for_server_async(session: aiohttp.ClientSession, max_wait: int = 120):
    """Poll the health endpoint until the server comes back, or give up after max_wait seconds."""
    waited = 0
    while waited < max_wait:
        if await _server_alive(session):
            return True
        await asyncio.sleep(2)
        waited += 2
    return False


async def _caption_single_frame_worker(
    session: aiohttp.ClientSession,
    b64_image: str,
    index: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> Dict[str, Any]:
    """Sends an individual base64 string to the running local Gemma 3 engine.
    Retries with exponential backoff on timeout/connection/server errors
    so a transient failure doesn't cost you the frame's caption."""
    server_url = "http://localhost:8080/v1/chat/completions"

    payload = {
        "model": "gemma-4-E4B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        "temperature": 0.0
    }

    async with semaphore:
        last_error = "unknown error"
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(
                    server_url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        caption = result["choices"][0]["message"]["content"]
                        return {"index": index, "text": caption}
                    else:
                        body = (await resp.text())[:200]
                        last_error = f"HTTP {resp.status}: {body}"
                        ui.warn(f"Frame {index} attempt {attempt}/{max_retries} failed: {last_error}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)
                ui.warn(f"Frame {index} attempt {attempt}/{max_retries} error: {last_error}")

            if attempt < max_retries:
                # if the server itself crashed/restarted, wait for it instead of
                # burning retries against a dead endpoint
                if not await _server_alive(session):
                    ui.warn(f"Server unresponsive — waiting for it to recover (frame {index})...")
                    await _wait_for_server_async(session)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))  # 2s, 4s, 8s, 16s, 32s...

        ui.error(f"Frame {index} permanently failed after {max_retries} attempts: {last_error}")
        return {"index": index, "text": f"Error: Failed after {max_retries} attempts ({last_error})"}
def batch_caption_frames(frame_list: list, concurrency_limit: int = 16) -> List[Dict[str, Any]]:
    total = len(frame_list)

    with ui.make_progress() as progress:
        task_id = progress.add_task("  Captioning frames", total=total)

        async def run_pipeline():
            semaphore = asyncio.Semaphore(concurrency_limit)
            tasks = []

            async def tracked_worker(session, b64_str, idx):
                result = await _caption_single_frame_worker(session, b64_str, idx, semaphore)
                progress.advance(task_id)
                return result

            async with aiohttp.ClientSession() as session:
                for idx, pil_frame in enumerate(frame_list):
                    try:
                        b64_str = _convert_pil_to_base64(pil_frame)
                        tasks.append(asyncio.create_task(tracked_worker(session, b64_str, idx)))
                    except Exception as e:
                        ui.warn(f"Skipping corrupt frame {idx}: {e}")

                results = await asyncio.gather(*tasks)

            results = [r for r in results if r is not None]
            results.sort(key=lambda x: x["index"])
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