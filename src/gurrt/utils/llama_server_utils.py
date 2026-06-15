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



def _convert_pil_to_base64(pil_img) -> str:
    """Converts a PIL image object to a base64 string completely in memory."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
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
        "model": "gemma-3-4b-it", 
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
        "temperature": 0.1
    }
    
    async with semaphore:
        try:
            async with session.post(server_url, json=payload, timeout=45) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    caption = result["choices"][0]["message"]["content"]
                    return {"index": index, "text": caption}
                else:
                    ui.warn(f"Engine error on frame {index}: HTTP {resp.status}")
                    return {"index": index, "text": "Error: Failed to generate description."}
        except Exception as e:
            ui.error(f"Server timeout on frame {index}: {e}")
            return {"index": index, "text": "Error: Pipeline connection exception."}

def batch_caption_frames(frame_list: list, concurrency_limit: int = 4) -> List[Dict[str, Any]]:
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


def download_gemma3_models(models_dir: Path):
    """
    Sequentially downloads Gemma 3 model weights and its associated 
    multimodal vision projector from Hugging Face Hub.
    """

    models_dir.mkdir(exist_ok=True, parents=True)
    #enable_progress_bars()  
    llama_server_manager = LlamaServerManager()
    huggingface_repo = llama_server_manager.hf_repo
    files = [
        llama_server_manager.model_filename, 
        llama_server_manager.mmproj_filename
    ]

    for filename in files:
        target_path = models_dir / filename
        
        if not target_path.exists():
            ui.step(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=huggingface_repo,
                filename=filename,
                local_dir=str(models_dir),
            )
            ui.success(f"Downloaded {filename}")
        else:
            ui.info(f"{filename} already present, skipping")