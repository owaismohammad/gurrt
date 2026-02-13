from multiprocessing import process
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import CLIPProcessor, CLIPModel
from src.vectordb import frame_embedding_collection
from utils.utils import device
import torch
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_CACHE_DIR")
if MODEL_DIR is None:
    raise RuntimeError("MODEL_DIR path not found")
clip_path = os.path.join(MODEL_DIR, "clip_model")
model = CLIPModel.from_pretrained(clip_path,local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)

def query_search(prompt: str):
    text_embedding  = processor(text = [prompt], return_tensors = 'pt').to(device)
    
    with torch.no_grad():
        output = model.get_text_features(**text_embedding)
        # normalize for cosine similarity
    text_features = output.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0]  # shape (512,)


text_vector = query_search('person explains limit of x to the power h minus 1  divided by h')
print(len(text_vector))
results = frame_embedding_collection.query(
    query_embeddings=[text_vector],
    n_results= 5,
    include = ["metadatas", "embeddings", "distances", "documents"]
)
print(results)