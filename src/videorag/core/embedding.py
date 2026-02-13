from pathlib import Path
from videorag.utils.utils import scene_split, frame_listing, batched_captioning


def scene_detection_frame_sampling(
                                   video_path: Path,
                                   clip_model, 
                                   clip_processor, 
                                   blip_processor, 
                                   blip_model):
    scene_list = scene_split(video_path)
    frame_PIL, timestamps_list, ids, fps = frame_listing(scene_list= scene_list, 
                                                         video_path= video_path)
    caption_list, embeddings_list = batched_captioning(frame_list= frame_PIL, 
                                                       batch_size=16, 
                                                       clip_model= clip_model, 
                                                       clip_processor= clip_processor, 
                                                       blip_model= blip_model, 
                                                       blip_processor= blip_processor)
    metadatas = [
            {
                    "caption": caption_list[i],
                    "timestamp_ms": timestamps_list[i],
                    "fps": fps,
                    "source_path": video_path
            }
                for i in range(len(caption_list))
                ]
    return embeddings_list, metadatas, ids


