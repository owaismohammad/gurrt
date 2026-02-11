import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from utils.utils import device
import cv2
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch    
from app.vector_db import frame_embedding_collection
from tqdm import tqdm
from PIL import Image
from utils.utils import scene_split, frame_listing, batched_captioning
from dotenv import load_dotenv

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

blip_path = os.path.join(MODEL_CACHE_DIR, "blip_model")
blip_processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_path, local_files_only=True).to(device)

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
embeddings, metadatas, ids = scene_detection_frame_sampling(INPUT_VIDEO, clip_model, clip_processor, blip_processor, blip_model)
frame_embedding_collection.add(
    ids = ids,
    embeddings= embeddings,
    metadatas= metadatas,
)