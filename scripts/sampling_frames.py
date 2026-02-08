import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from utils.utils import generate_caption, device
import cv2
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
from app.vector_db import frame_embedding_collection
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
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


def uniform_frame_sampling(path: str):
    cap= cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_no = 0
    embeddings = []
    metadatas = []
    ids = []
    # buffer = BytesIO()
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_no % int(fps) == 0:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
                # caption=generate_caption(image,buffer)
                # print(caption)
                blip_input = blip_processor(images = image,
                                            text="Describe the scene in a factual, objective manner.",
                                            return_tensors = 'pt').to(device)
                with torch.no_grad():
                    outputs = clip_model.get_image_features(inputs.pixel_values)
                    blip_outputs = blip_model.generate(**blip_input,
                                                       max_length = 60,
                                                       min_length = 20,
                                                       no_repeat_ngram_size=2,
                                                       num_beams = 5,
                                                       )
                
                caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
                image_embedding = outputs.pooler_output
                image_embedding = image_embedding / image_embedding.norm(dim = -1, keepdim= True)
                image_embedding = image_embedding.squeeze(0).cpu().numpy().tolist()
                frame_id = f"{path}:{timestamp_sec}"
                
                ids.append(frame_id)
                embeddings.append(image_embedding)

                metadatas.append({
                    "frame_idx": frame_no,
                    "caption": caption,
                    "timestamp_ms": timestamp_sec,
                    "fps": fps,
                    "source_path": path
                })
                
            frame_no +=1
            pbar.update(1)
    return embeddings, metadatas, ids

embeddings, metadatas, ids = uniform_frame_sampling(path = INPUT_VIDEO)
frame_embedding_collection.add(
    ids = ids,
    embeddings= embeddings,
    metadatas= metadatas,
)