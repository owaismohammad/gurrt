import os
from pathlib import Path
import torch

from gurrt.config.config import Settings
from gurrt.utils.utils import audio_extraction, audio_to_text, chunk_text
from gurrt.cli import ui

def audio_extract_chunk_and_embed(video_path: Path,
                                settings: Settings, 
                                clip_model, 
                                clip_processor, 
                                whisper_model,
                                device):
    ui.step("Extracting audio track...")
    audio_file = audio_extraction(path=video_path, settings=settings)
    ui.step("Transcribing audio...")
    text = audio_to_text(audio_file, 
                        model= whisper_model,
                        beam_size= 1)
    chunked_text = chunk_text(text=text)
    clip_inputs = clip_processor(text= chunked_text, 
                                return_tensors="pt", 
                                padding= True, 
                                truncation = True).to(device)
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
        {"video_path": str(video_path), "type": "audio_transcript"}
        for _ in range(len(chunked_text))
    ]
    return chunked_text, metadatas, text_features, ids