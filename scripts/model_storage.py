import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
load_dotenv()

CLIP_MODEL = os.getenv(key = 'CLIP_MODEL')
BLIP_MODEL = os.getenv(key = 'BLIP_MODEL')
RERANKER_MODEL = os.getenv(key = 'RERANKER_MODEL')
MODEL_CACHE_DIR = os.getenv(key = 'MODEL_CACHE_DIR')

if CLIP_MODEL is None:
    raise RuntimeError("CHROMA_DB_PATH is not set")
elif RERANKER_MODEL is None:
    raise RuntimeError("RERANKER_MODEL is not set")
elif MODEL_CACHE_DIR is None:
    raise RuntimeError("MODEL_CACHE_DIR is not set")
elif BLIP_MODEL is None:
    raise RuntimeError("BLIP_MODEL is not set")

model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

clip_path = os.path.join(MODEL_CACHE_DIR,"clip_model")
model.save_pretrained(clip_path)
processor.save_pretrained(clip_path)

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL, )
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL, use_safetensors=True)

blip_path = os.path.join(MODEL_CACHE_DIR,"blip_model")
blip_model.save_pretrained(blip_path)
blip_processor.save_pretrained(blip_path)

reranker_model = CrossEncoder(RERANKER_MODEL)

reranker_path = os.path.join(MODEL_CACHE_DIR, "reranker_model")
reranker_model.save(reranker_path) 
