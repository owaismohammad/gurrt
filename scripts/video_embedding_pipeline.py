import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from app.vector_db import frame_embedding_collection
from utils.utils import scene_split, frame_listing, batched_captioning,device,generate_caption
from dotenv import load_dotenv
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
load_dotenv()

INPUT_VIDEO = os.getenv(key = 'INPUT_VIDEO')
MODEL_CACHE_DIR = os.getenv(key = 'MODEL_CACHE_DIR')

if INPUT_VIDEO is None:
    raise RuntimeError("INPUT_VIDEO path is not set")
elif MODEL_CACHE_DIR is None:
    raise RuntimeError("CLIP_CACHE path is not set")

clip_path = os.path.join(MODEL_CACHE_DIR, "clip_model")
clip_model = CLIPModel.from_pretrained(clip_path, local_files_only= True).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)

# blip_path = os.path.join(MODEL_CACHE_DIR, "blip_model")
# blip_processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True)
# blip_model = BlipForConditionalGeneration.from_pretrained(blip_path, local_files_only=True).to(device)


def scene_detection_frame_sampling(INPUT_VIDEO, clip_model, clip_processor, blip_processor, blip_model):
    scene_list = scene_split(INPUT_VIDEO)
    frame_PIL, timestamps_list, ids, fps = frame_listing(scene_list= scene_list, 
                                                         video_path= INPUT_VIDEO)
    caption_list, embeddings_list = batched_captioning(frame_list= frame_PIL, 
                                                       batch_size=16, 
                                                       clip_model= clip_model, 
                                                       clip_processor= clip_processor, 
                                                       blip_model= blip_model, 
                                                       blip_processor= blip_processor)
    metadatas = []
    for i in range(len(caption_list)):
        metadatas.append({
                    "caption": caption_list[i],
                    "timestamp_ms": timestamps_list[i],
                    "fps": fps,
                    "source_path": INPUT_VIDEO
                })
    print("EMBEDDING_LIST_LENGTH:- ",len(embeddings_list))
    print("METADATA_LIST_LENGTH:- ",len(metadatas))
    print("IDS_LIST_LENGTH:- ",len(ids))
    return embeddings_list, metadatas, ids

def detect_scenes(video_path, scene_list):
    cap = cv2.VideoCapture(video_path)   
    embeddings = []
    metadatas = []
    ids = []
    
    with tqdm(total = len(scene_list), desc = "Processing frames") as pbar:
        for i, scene in enumerate(scene_list):
            start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
            mid_time = (start_time + end_time) / 2
            timestamps = mid_time
            labels = "middle"
            
            buffer = BytesIO()
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamps * 1000)
            ret, frame = cap.read()
            
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                caption=generate_caption(image,buffer)
                print(caption)
                inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
                # blip_input = blip_processor(images = image, return_tensors = 'pt').to(device)
                with torch.no_grad():
                    outputs =clip_model.get_image_features(inputs.pixel_values)
                    # blip_outputs = blip_model.generate(**blip_input,
                    #                                 # max_length = 60,
                    #                                 # min_length = 20,
                    #                                 # no_repeat_ngram_size=2,
                    #                                 # num_beams = 5,
                    #                                 )
                
                # caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
                image_embedding = outputs.pooler_output
                image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
                image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
                timestamp_sec = timestamps*1000
                frame_id = f"{video_path}:{timestamp_sec}"
            
                ids.append(frame_id)
                embeddings.append(image_embedding)

                metadatas.append({
                    "frame_idx": f"frame_no_{i}_{labels}",
                    "caption": caption,
                    "timestamp_ms": timestamp_sec,
                    "source_path": video_path
                })
            pbar.update(1)
               
    return embeddings, metadatas, ids
scene_list=scene_split(INPUT_VIDEO)
embeddings, metadatas, ids = detect_scenes(INPUT_VIDEO, scene_list)

# embeddings, metadatas, ids = scene_detection_frame_sampling(INPUT_VIDEO, clip_model, clip_processor, blip_processor, blip_model)
frame_embedding_collection.add(
    ids = ids,
    embeddings= embeddings,
    metadatas= metadatas,
)


