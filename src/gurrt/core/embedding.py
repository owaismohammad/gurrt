from pathlib import Path
import time
from gurrt.core.models import ModelManager
from gurrt.cli import ui
from gurrt.utils.utils import (
                            batched_captioning,
                            batched_captioning_blip,
                            temporal_persistence_filter,
                            captioning_ollama,
                            embed_texts)
from gurrt.utils.llama_server_utils import batch_caption_frames


def _build_records(caption_list, timestamps_list, end_times, ids, fps,
                   video_path, text_embedder):
    """Turn captions into vector-DB rows keyed by the caption text.

    Frames are indexed by what they *say*, not by what they look like: a CLIP
    image vector barely encodes small on-screen text, which is most of what a
    lecture slide carries. Embedding the caption also puts frames and
    transcript chunks in one shared space, so a single query vector searches
    both collections.
    """
    n = len(caption_list)
    metadatas = [
        {
            "caption": caption_list[i],
            "start_sec": timestamps_list[i],
            "end_sec": end_times[i],
            "fps": fps,
            "source_path": str(video_path),
        }
        for i in range(n)
    ]
    embeddings = embed_texts(caption_list, text_embedder)
    return embeddings, metadatas, ids[:n]


def frame_detection(video_path: Path,
                    models: ModelManager,
                    flag: bool,
                    text_embedder,
                    device):

    frame_PIL, timestamps_list, end_times, ids, fps = temporal_persistence_filter(video_path= video_path)
    if flag :
        batch_size=4
    else:
        batch_size=8
    smol_model, smol_processor = models.get_smol(flag = flag)
    caption_list = batched_captioning(frame_list= frame_PIL,
                                                    batch_size= batch_size,
                                                    smol_model= smol_model,
                                                    smol_processor= smol_processor,
                                                    device = device)
    return _build_records(caption_list, timestamps_list, end_times, ids, fps,
                          video_path, text_embedder)


def frame_detection_blip(video_path: Path,
                    models: ModelManager,
                    text_embedder,
                    device):

    frame_PIL, timestamps_list, end_times, ids, fps = temporal_persistence_filter(video_path= video_path)
    blip_model, blip_processor = models.get_blip()
    caption_list = batched_captioning_blip(frame_list= frame_PIL,
                                                    batch_size=8,
                                                    blip_model= blip_model,
                                                    blip_processor= blip_processor,
                                                    device = device)
    return _build_records(caption_list, timestamps_list, end_times, ids, fps,
                          video_path, text_embedder)


def captioning_and_embedding_llama_server(
    frame_PIL,
    timestamps_list,
    end_times,
    ids,
    fps,
    video_path,
    text_embedder,
):
    ui.info(f"Dispatching {len(frame_PIL)} frames to captioning server...")
    captioned_nodes = []
    start_time = time.time()
    try:
        captioned_nodes = batch_caption_frames(frame_list=frame_PIL, concurrency_limit=4)
    except Exception as e:
        ui.error(f"Batch captioning failed: {e}")
        return [], [], []
    end_time = time.time()
    ui.info(f"Captioning done in {end_time - start_time:.1f}s — embedding captions...")

    # Corrupt frames are skipped by the captioner, so rows are built off each
    # frame's own index. Building them from separate loops lets one skip shift
    # every caption onto the wrong timestamp.
    caption_by_index = {node["index"]: node["text"] for node in captioned_nodes}
    kept = sorted(caption_by_index)

    caption_list = [caption_by_index[i] for i in kept]
    metadatas = [
        {
            "caption": caption_by_index[i],
            "start_sec": timestamps_list[i],
            "end_sec": end_times[i],
            "fps": fps,
            "source_path": str(video_path),
        }
        for i in kept
    ]
    final_ids = [ids[i] for i in kept]

    start_time = time.time()
    embeddings = embed_texts(caption_list, text_embedder)
    end_time = time.time()
    ui.info(f"Caption embeddings done in {end_time - start_time:.1f}s")
    return embeddings, metadatas, final_ids


def frame_detection_ollama(video_path: Path,
                            text_embedder,
                            model_name:str,
                            device):
    frame_PIL, timestamps_list, end_times, ids, fps = temporal_persistence_filter(video_path= video_path)
    caption_list = captioning_ollama(frame_PIL= frame_PIL,
                                     model_name= model_name)
    return _build_records(caption_list, timestamps_list, end_times, ids, fps,
                          video_path, text_embedder)
