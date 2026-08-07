import os
from pathlib import Path

from gurrt.config.config import Settings
from gurrt.utils.utils import (audio_extraction, audio_to_segments,
                               chunk_segments, embed_texts)
from gurrt.cli import ui

def audio_extract_chunk_and_embed(video_path: Path,
                                settings: Settings,
                                text_embedder,
                                whisper_model,
                                device):
    ui.step("Extracting audio track...")
    audio_file = audio_extraction(path=video_path, settings=settings)
    ui.step("Transcribing audio...")
    segments = audio_to_segments(audio_file,
                                model= whisper_model,
                                beam_size= 1)
    chunks = chunk_segments(segments)
    chunked_text = [c["text"] for c in chunks]
    text_features = embed_texts(chunked_text, text_embedder)
    video_id = os.path.basename(video_path)
    ids = [
        f"{video_id}_chunk_{i}" 
        for i in range(len(chunked_text))
    ]
    metadatas = [
        {
            "video_path": str(video_path),
            "type": "audio_transcript",
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
        }
        for c in chunks
    ]
    return chunked_text, metadatas, text_features, ids