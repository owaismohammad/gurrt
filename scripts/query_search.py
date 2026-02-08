from multiprocessing import process
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import CLIPProcessor, CLIPModel
from app.vector_db import frame_embedding_collection
import torch
from dotenv import load_dotenv

load_dotenv()

CLIP_MODEL = os.getenv("CLIP_MODEL")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(CLIP_MODEL,use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

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
    n_results= 5
)
print(results["metadatas"])