import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import device
from time import time
from faster_whisper import WhisperModel
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from app.vector_db import frame_embedding_collection, asr_collection
from scripts.video_embedding_pipeline import scene_detection_frame_sampling
from scripts.asr_pipeline import audio_extract_chunk_and_embed
from dotenv import load_dotenv
load_dotenv()

INPUT_VIDEO = os.getenv(key= "INPUT_VIDEO")
if INPUT_VIDEO is None:
    raise RuntimeError("INPUT_VIDEO path is not set")

MODEL_CACHE_DIR = os.getenv(key = 'MODEL_CACHE_DIR')
WHISPER_MODEL = os.getenv(key = 'WHISPER_MODEL')

if MODEL_CACHE_DIR is None:
    raise RuntimeError("CLIP_CACHE path is not set")
elif WHISPER_MODEL is None:
    raise RuntimeError("WHISPER_MODEL is not set")

clip_path = os.path.join(MODEL_CACHE_DIR, "clip_model")
clip_model = CLIPModel.from_pretrained(clip_path,
                                       local_files_only= True).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_path,
                                               local_files_only=True)

blip_path = os.path.join(MODEL_CACHE_DIR, "blip_model")
blip_processor = BlipProcessor.from_pretrained(blip_path,
                                               local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_path,
                                                          local_files_only=True).to(device)
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="int8_float16"
)
def vectordb_collections_formation(input_video):
    start_time_frame_sampling = time()
    frame_embeddings, frame_metadatas, frame_ids = scene_detection_frame_sampling(input_video,
                                                                clip_model,
                                                                clip_processor,
                                                                blip_processor,
                                                                blip_model)
    end_time_frame_sampling = time()
    print(f"FOR FRAME EXTRACTION SAMPLING EMBEDDING AND CAPTIONING IT TOOK {end_time_frame_sampling - start_time_frame_sampling}")
    
    start_time_audio_sampling = time()
    chunked_text, metadatas_audio, text_embeddings, ids_audio =audio_extract_chunk_and_embed(input_video,
                                                                                 clip_model, clip_processor, whisper_model)
    end_time_audio_sampling = time()
    print(f"FOR AUDIO EXTRACTION EMBEDDING AND TRANSCRIPTING IT TOOK {end_time_audio_sampling  - start_time_audio_sampling}")
    
    frame_embedding_collection.add(
    ids = frame_ids,
    embeddings= frame_embeddings,
    metadatas= frame_metadatas
)
    asr_collection.add(
    ids = ids_audio,
    embeddings= text_embeddings,
    metadatas= metadatas_audio,
    documents= chunked_text
)
    
f = vectordb_collections_formation(INPUT_VIDEO)