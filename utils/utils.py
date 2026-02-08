import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from ollama import chat
import moviepy as mp
from sentence_transformers import CrossEncoder
import torch
from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.getenv("MODEL_CACHE_DIR")
if MODEL_DIR is None:
    raise RuntimeError("MODEL_CACHE_DIR is not set")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)



def audio_extraction(path: str):
    audio_file = '../outputs/audio_file.mp3'
    video = mp.VideoFileClip(path)
    audio = video.audio
    
    audio.write_audiofile(audio_file)
    
    audio.close()
    video.close()
    return audio_file


def generate_caption(frame,buffer):
    frame.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    response = chat(
    model='gemma3',
    messages=[
        {
    "role": "system",
    "content": "You are a helpful assistant that can analyze images and provide captions."
    },

    {
        'role': 'user',
        'content': 'What is in this image? .',
        'images': [img_bytes],
    }
    ],
    )

    return response.message.content



def rerank(query: str, results: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    reranker_model = CrossEncoder(f"{MODEL_DIR}/reranker_model", device=device) 
    if not results['documents'][0]:
        return results
    captions = []
    metadata = results['metadatas'][0]
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for dic in metadata:
        for key, val in dic.items():
            if key == "caption":
                captions.append(val)
    pairs = [[query, caption] for caption in captions]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(captions, metadatas, distances, scores), key=lambda x: x[3], reverse=True)

    final_cap = []
    final_metas = []
    final_dists = []

    for cap, meta, dist, score in ranked[:top_k]:
        meta['relevance_score'] = float(score)
        final_cap.append(cap)
        final_metas.append(meta)
        final_dists.append(dist)

    return {
        'captions': [final_cap],
        'metadatas': [final_metas],
        'distances': [final_dists]
    }
    