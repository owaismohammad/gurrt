from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import genrate_caption
import cv2
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
from app.vector_db import frame_embedding_collection
# from app.prompts import BLIP_CUSTOM_PROMPT
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

CLIP_MODEL = os.getenv(key = 'CLIP_MODEL')
BLIP_MODEL = os.getenv(key = 'BLIP_MODEL')
INPUT_VIDEO = os.getenv(key = 'INPUT_VIDEO')

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL,
                                       use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

# blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
# blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)


def uniform_frame_sampling(path: str):
    cap= cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_no = 0
    embeddings = []
    metadatas = []
    ids = []
    buffer = BytesIO()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % int(round(fps)) == 0:
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            inputs = clip_processor(images = image, return_tensors = 'pt').to(device)
            caption=genrate_caption(image,buffer)
           # print(caption)
            #blip_input = blip_processor(images = image, return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = clip_model.get_image_features(inputs.pixel_values)
            #     blip_outputs = blip_model.generate(**blip_input,
            #                                        max_length = 500,
            #                                        min_length = 150,
            #                                        no_repeat_ngram_size=2,
            #                                        num_beams = 5,
            #                                        )
            
            # caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
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
        
    return embeddings, metadatas, ids

embeddings, metadatas, ids = uniform_frame_sampling(path = INPUT_VIDEO)
print(len(embeddings[0]))

frame_embedding_collection.add(
    ids = ids,
    embeddings= embeddings,
    metadatas= metadatas,
)