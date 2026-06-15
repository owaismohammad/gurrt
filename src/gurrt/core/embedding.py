from pathlib import Path
import time
import torch
from gurrt.core.models import ModelManager
from gurrt.cli import ui
from gurrt.utils.utils import (
                            batched_captioning,
                            batched_captioning_blip,
                            temporal_persistence_filter,
                            captioning_ollama)
from gurrt.utils.llama_server_utils import batch_caption_frames

def frame_detection(video_path: Path,
                    models: ModelManager,
                    flag: bool,
                    clip_model, 
                    clip_processor, 
                    device):
    
    frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path= video_path)
    if flag :
        batch_size=4
    else:
        batch_size=8    
    smol_model, smol_processor = models.get_smol(flag = flag)
    caption_list, embeddings_list = batched_captioning(frame_list= frame_PIL, 
                                                    batch_size= batch_size, 
                                                    clip_model= clip_model, 
                                                    clip_processor= clip_processor, 
                                                    smol_model= smol_model,
                                                    smol_processor= smol_processor,
                                                    device = device)
    metadatas = [
            {
            "caption": caption_list[i],
            "timestamp_ms": timestamps_list[i],
            "fps": fps,
            "source_path": str(video_path)
            }
                for i in range(len(caption_list))
                ]
    return embeddings_list, metadatas, ids

def frame_detection_blip(video_path: Path,
                    models: ModelManager,
                    clip_model, 
                    clip_processor, 
                    device):
    
    frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path= video_path)
    blip_model, blip_processor = models.get_blip()    
    caption_list, embeddings_list = batched_captioning_blip(frame_list= frame_PIL, 
                                                    batch_size=8, 
                                                    clip_model= clip_model, 
                                                    clip_processor= clip_processor, 
                                                    blip_model= blip_model, 
                                                    blip_processor= blip_processor,
                                                    device = device)
    metadatas = [
            {
            "caption": caption_list[i],
            "timestamp_ms": timestamps_list[i],
            "fps": fps,
            "source_path": str(video_path)
            }
                for i in range(len(caption_list))
                ]
    return embeddings_list, metadatas, ids

def captioning_and_embedding_llama_server(
    frame_PIL,
    timestamps_list,
    ids,
    fps,
    video_path,
    clip_model,
    clip_processor,
    device
):
    #frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path=video_path)

    ui.info(f"Dispatching {len(frame_PIL)} frames to captioning server...")
    captioned_nodes = []
    start_time = time.time()
    try:
        captioned_nodes = batch_caption_frames(frame_list=frame_PIL, concurrency_limit=4)
    except Exception as e:
        ui.error(f"Batch captioning failed: {e}")
        return [], [], []
    end_time = time.time()
    ui.info(f"Captioning done in {end_time - start_time:.1f}s — extracting embeddings...")
    caption_list = [node["text"] for node in captioned_nodes]

    metadatas = [
        {
            "caption": caption_list[i],
            "timestamp_ms": timestamps_list[i],
            "fps": fps,
            "source_path": str(video_path),
        }
        for i in range(len(caption_list))
    ]

    embeddings = []
    start_time = time.time()
    with ui.make_progress() as progress:
        task_id = progress.add_task("  Extracting CLIP embeddings", total=len(frame_PIL))
        for i, frame in enumerate(frame_PIL):
            try:
                inputs = clip_processor(images=frame, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = clip_model.get_image_features(**inputs)
                image_embedding = outputs.pooler_output
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
                embeddings.append(image_embedding)
            except Exception as e:
                ui.error(f"CLIP embedding failed on frame {i}: {e}")
            finally:
                progress.advance(task_id)
    end_time = time.time()
    ui.info(f"CLIP embeddings done in {end_time - start_time:.1f}s")
    return embeddings, metadatas, ids

def frame_detection_ollama(video_path: Path,
                            clip_model, 
                            clip_processor, 
                            model_name:str,
                            device):
    frame_PIL, timestamps_list, ids, fps = temporal_persistence_filter(video_path= video_path)
    embeddings, metadatas, ids =  captioning_ollama(video_path= video_path,
                                                    clip_model= clip_model,
                                                    clip_processor= clip_processor,
                                                    model_name= model_name,
                                                    frame_PIL= frame_PIL,
                                                    timestamps_list = timestamps_list,
                                                    fps = fps, 
                                                    device= device)
    return embeddings, metadatas, ids
