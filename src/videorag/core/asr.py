# import sys
import os
from pathlib import Path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
# from videorag.core.vectordb import asr_collection
from videorag.utils.utils import audio_extraction, audio_to_text, chunk_text, device
# from transformers import CLIPModel, CLIPProcessor
# from faster_whisper import WhisperModel
# from dotenv import load_dotenv

# load_dotenv()

def audio_extract_chunk_and_embed(
                                video_path: Path, 
                                clip_model, 
                                clip_processor, 
                                whisper_model):
    audio_file = audio_extraction(path= video_path)
    text = audio_to_text(audio_file, model= whisper_model )
    chunked_text = chunk_text(text=text)
    clip_inputs = clip_processor(text= chunked_text, return_tensors="pt", padding= True, ).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()
    
    video_id = os.path.basename(video_path)
    ids = [
        f"{video_id}_chunk_{i}" 
        for i in range(len(chunked_text))
    ]
    
    metadatas = [
        {"video_path": video_path, "type": "audio_transcript"}
        for _ in range(len(chunked_text))
    ]
    return chunked_text, metadatas, text_features, ids


# if __name__ == "__main__":
#     WHISPER_MODEL = os.getenv(key = 'WHISPER_MODEL')
#     INPUT_VIDEO = os.getenv(key = "INPUT_VIDEO")
#     MODEL_CACHE_DIR = os.getenv(key = 'MODEL_CACHE_DIR')

#     if INPUT_VIDEO is None:
#         raise RuntimeError("INPUT_VIDEO path is not set")
#     elif MODEL_CACHE_DIR is None:
#         raise RuntimeError("CLIP_CACHE path is not set")
#     elif WHISPER_MODEL is None:
#         raise RuntimeError("WHISPER_MODEL is not set")
#     elif INPUT_VIDEO is None:
#         raise RuntimeError("INNPUT_VIDEO is not set")

#     clip_path = os.path.join(MODEL_CACHE_DIR, "clip_model")
#     clip_model = CLIPModel.from_pretrained(clip_path, local_files_only= True).to(device)
#     clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
#     whisper_model = WhisperModel(
#         WHISPER_MODEL,
#         device="cuda",
#         compute_type="int8_float16"
#     )
#     chunked_text, metadatas, text_embeddings, ids = audio_extract_chunk_and_embed(INPUT_VIDEO, clip_model, clip_processor, whisper_model)

#     asr_collection.add(
#         ids = ids,
#         embeddings= text_embeddings,
#         metadatas= metadatas,
#         documents= chunked_text
#     )