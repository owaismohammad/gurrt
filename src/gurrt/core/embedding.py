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
from gurrt.core.debuglog import log_keyframes
# from automation.caption_bridge import export_keyframes, load_captions, reload_frames
from gurrt.config.benchmark_config import VIDEO_PATH, OUTPUT_PATH, MANIFEST_PATH, CAPTION_PATH, DEFAULT_OUT


def _build_records(caption_list, timestamps_list, end_times, ids, fps,
                   video_path, text_embedder, captioner, settings=None,
                   frame_PIL=None):
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
            # Recorded so `ask` knows how far to trust these captions.
            "captioner": captioner,
        }
        for i in range(n)
    ]
    if settings is not None and frame_PIL is not None:
        log_keyframes(settings, video_path, frame_PIL[:n], timestamps_list[:n])

    embeddings = embed_texts(caption_list, text_embedder)
    return embeddings, metadatas, ids[:n]


def frame_detection(video_path: Path,
                    models: ModelManager,
                    flag: bool,
                    text_embedder,
                    device,
                    settings=None):

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
                          video_path, text_embedder, captioner="smolvlm",
                          settings=settings, frame_PIL=frame_PIL)


def frame_detection_blip(video_path: Path,
                         out_dir:Path,
                    models: ModelManager,
                    text_embedder,
                    device,
                    settings=None,
                    ):

    # Before Captioning
    frame_PIL, timestamps_list, end_times, ids, fps=  temporal_persistence_filter(video_path= video_path,
        persistence_window_sec= 1.5,
        hash_threshold= 15,
        fps_selected= 2
        )
    manifest = export_keyframes(
    frame_PIL, timestamps_list, end_times, ids,
    out_dir=Path(out_dir),
    fps=fps, video_path=video_path,
)   
    print(manifest)

    # After Captioning
    
#     caption_list, timestamps_list, end_times, ids,fps = load_captions( 
#     MANIFEST_PATH,  # Manifest path
#     CAPTION_PATH, # Captions path
# )
#     frame_PIL = reload_frames(MANIFEST_PATH) # Manifest path

    # return _build_records(caption_list, timestamps_list, end_times, ids, fps,
    #                           video_path, text_embedder, captioner="blip2",
    #                           settings=settings, frame_PIL=frame_PIL)
    # blip_model, blip_processor = models.get_blip()
    # caption_list = batched_captioning_blip(frame_list= frame_PIL,
    #                                                 batch_size=8,
    #                                                 blip_model= blip_model,
    #                                                 blip_processor= blip_processor,
    #                                                 device = device)
    # return _build_records(caption_list, timestamps_list, end_times, ids, fps,
    #                       video_path, text_embedder, captioner="blip2",
    #                       settings=settings, frame_PIL=frame_PIL)


def captioning_and_embedding_llama_server(
    frame_PIL,
    timestamps_list,
    end_times,
    ids,
    fps,
    video_path,
    text_embedder,
    max_workers: int,
    out_dir_bench : Path = None,
    settings=None,
    
):
    ui.info(f"Dispatching {len(frame_PIL)} frames to captioning server...")
    captioned_nodes = []
    start_time = time.time()
    try:
        captioned_nodes = batch_caption_frames(frame_list=frame_PIL, concurrency_limit= max_workers)
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
            "captioner": "gemma3",
        }
        for i in kept
    ]
    final_ids = [ids[i] for i in kept]

    if settings is not None:
        # Only the frames that produced a caption, so the images match
        # captions.json exactly.
        log_keyframes(settings, video_path,
                      [frame_PIL[i] for i in kept],
                      [timestamps_list[i] for i in kept],
                      out_dir_bench)

    start_time = time.time()
    embeddings = embed_texts(caption_list, text_embedder)
    end_time = time.time()
    ui.info(f"Caption embeddings done in {end_time - start_time:.1f}s")
    return embeddings, metadatas, final_ids


def frame_detection_ollama(video_path: Path,
                            text_embedder,
                            model_name:str,
                            device,
                            settings=None):
    frame_PIL, timestamps_list, end_times, ids, fps = temporal_persistence_filter(video_path= video_path)
    caption_list = captioning_ollama(frame_PIL= frame_PIL,
                                     model_name= model_name)
    return _build_records(caption_list, timestamps_list, end_times, ids, fps,
                          video_path, text_embedder,
                          captioner=f"ollama:{model_name}",
                          settings=settings, frame_PIL=frame_PIL)
