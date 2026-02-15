from pathlib import Path
from gurrt.utils.utils import detect_scenes, scene_split, frame_listing, batched_captioning, uniform_frame_sampling, uniform_frame_sampling_ollama


def scene_detection_frame_sampling(
                                   video_path: Path,
                                   clip_model, 
                                   clip_processor, 
                                   blip_processor, 
                                   blip_model,
                                   device):

    scene_list = scene_split(video_path)
    if not scene_list:
            print("\033[1;32mSince No Scene Detected!\nFalling over to Uniform Sampling Technique\033[0m")
            embeddings, metadatas, ids =  uniform_frame_sampling(path = video_path,
                                                                 clip_model= clip_model,
                                                                 clip_processor= clip_processor,
                                                                 blip_processor=blip_processor,
                                                                 blip_model=blip_model,
                                                                 device= device)
            return embeddings, metadatas, ids
    frame_PIL, timestamps_list, ids, fps = frame_listing(scene_list= scene_list, 
                                                         video_path= video_path)
    caption_list, embeddings_list = batched_captioning(frame_list= frame_PIL, 
                                                       batch_size=16, 
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
                    "source_path": video_path
            }
                for i in range(len(caption_list))
                ]
    return embeddings_list, metadatas, ids


def scene_detection_frame_sampling_ollama(
                                   video_path: Path,
                                   clip_model, 
                                   clip_processor, 
                                   model_name:str,
                                   device):

    scene_list = scene_split(video_path)
    if not scene_list:
            print("\033[1;32mSince No Scene Detected!\nFalling over to Uniform Sampling Technique\033[0m")
            embeddings, metadatas, ids =  uniform_frame_sampling_ollama(
                                                                 video_path= video_path,
                                                                 clip_model= clip_model,
                                                                 clip_processor= clip_processor,
                                                                 model_name= model_name,
                                                                 device= device)
            return embeddings, metadatas, ids
    embeddings, metadatas, ids = detect_scenes(video_path=video_path,
                                               scene_list=scene_list,
                                               clip_model=clip_model,
                                               clip_processor=clip_processor,
                                               model= model_name,
                                               device = device)
    
    
    return embeddings, metadatas, ids
