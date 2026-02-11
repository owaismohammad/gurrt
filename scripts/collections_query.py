import sys
import os
from typing import Any, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from utils.utils import device, rerank, rerank_docs, caption_frame_collection
from app.vector_db import frame_embedding_collection, asr_collection
import torch
load_dotenv()

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")


if MODEL_CACHE_DIR is None:
    raise RuntimeError("MODEL_CACHE_DIR path not found")

clip_path = os.path.join(MODEL_CACHE_DIR, "clip_model")
model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)



def query_collection(query: str, n_results: int = 10):
    text_embedding  = processor(text = [query], return_tensors = 'pt').to(device)
    
    with torch.no_grad():
        output = model.get_text_features(**text_embedding)

    text_features = output.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()[0]  # shape (512,)
    
    results = frame_embedding_collection.query(
    query_embeddings=[text_features],
    n_results= n_results
)   
    results_reranked = rerank(query, results, MODEL_CACHE_DIR, n_results)
    
    results_audio = asr_collection.query(
    query_embeddings=[text_features],
    n_results= n_results
)   
    results_reranked_audio = rerank_docs(query, results_audio, MODEL_CACHE_DIR, n_results)
    captions_list = caption_frame_collection(results_reranked)
    print(results_audio["documents"])
    
    
    asr_list = results_reranked_audio["documents"][0]
    return captions_list, asr_list


