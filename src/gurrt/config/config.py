from pathlib import Path
from platformdirs import user_config_dir
import json

class Settings:
    def __init__(self):
        home = Path(user_config_dir("gurrt"))
        home.mkdir(exist_ok=True, parents= True)
        
        config_file = home / "config.json"
        cfg = {}
        
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self.GROQ_API_KEY = cfg.get('GROQ_API_KEY')
            self.SUPERMEMORY_API_KEY = cfg.get("SUPERMEMORY_API_KEY")
        else:
            raise RuntimeError("API Keys not found")
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.LLM_MODEL="llama-3.1-8b-instant"
        self.RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.BLIP_MODEL = "Salesforce/blip-image-captioning-large"
        self.WHISPER_MODEL = "large-v2"

        self.MODEL_CACHE_DIR = home / "models"
        self.CHROMA_DB_PATH= home / "chroma_db"
        self.AUDIO_PATH = home / "output.mp3"
