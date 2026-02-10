import sys
import os
from typing import Any, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from utils.utils import device, rerank
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
    results_reranked = rerank(query, results, n_results)
    
    results_audio = asr_collection.query(
    query_embeddings=[text_features],
    n_results= n_results
)   
    results_reranked_audio = rerank(query, results_audio, n_results)
    
    captions_list = caption_frame_collection(results_reranked)
    return results_reranked, results_reranked_audio

def caption_frame_collection(results_reranked: Dict[str, Any]) -> list:
    caption_list = []
    results_ids = results_reranked["ids"][0]
    metadatas = results_reranked["metadatas"][0]
    for i, metadata in enumerate(metadatas):
        if metadata["caption"]:
            caption_list.append(metadata["caption"])
                
    return caption_list

def asr_