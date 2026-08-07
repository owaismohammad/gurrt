from pathlib import Path
from platformdirs import user_config_dir
import json
import sys
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
        # Every collection is keyed by text (captions and transcript alike), so
        # one embedder serves both the index and the query side.
        self.TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
        self.LLM_MODEL="llama-3.1-8b-instant"
        self.RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.BLIP_MODEL = "Salesforce/blip-image-captioning-large"
        # Context budget. The model's window is far larger than these numbers,
        # but an 8B model stops attending well long before it runs out of room:
        # a wall of loosely-related context buries the query instead of
        # answering it. Keep the retrieved material small and dense.
        self.CONTEXT_TOKEN_BUDGET = 3500   # the lecture timeline
        self.CHAT_TOKEN_BUDGET = 600       # prior conversation
        self.WINDOW_PAD_SEC = 30.0         # context pulled either side of a hit

        # BLIP and SmolVLM describe the room rather than the board: they cannot
        # read slide text, so their captions are near-useless for lecture
        # content. When an index was built with one of them, spend the budget
        # on speech instead and tell the model not to trust the visuals.
        self.LOW_FIDELITY_CAPTIONERS = {"blip2", "smolvlm"}
        # Subtracted from frame anchors' rerank scores, so audio windows win
        # the budget. Additive, not multiplicative: cross-encoder scores go
        # negative, and scaling a negative score makes it bigger.
        self.VISUAL_ANCHOR_PENALTY = 3.0

        self.MODEL_CACHE_DIR = home / "models"
        self.CHROMA_DB_PATH= home / "chroma_db"
        self.AUDIO_PATH = home / "output.wav"
        # Inspection logs: captions and transcript per video, plus the exact
        # context sent to the LLM per query.
        self.LOGS_DIR = home / "logs"
       
# config_dir = Path(user_config_dir("gurrt"))

# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# is_windows = sys.platform == "win32"
# BIN_DIR = config_dir / "bin"

# SERVER_BIN = config_dir / "bin" / ("llama-server.exe" if is_windows else "llama-server")    
# MODELS_DIR = config_dir / "models"
# hf_repo = "unsloth/gemma-3-4b-it-GGUF"
# model="gemma-3-4b-it-Q4_0.gguf"
# mmproj_model = "mmproj-F16.gguf"
# llm_path = MODELS_DIR / model
# clip_path = MODELS_DIR / mmproj_model
# llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

class LlamaServerManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir("gurrt"))
        
        self.is_windows = sys.platform == "win32"
        
        self.bin_dir = self.config_dir / "bin"
        self.models_dir = self.config_dir / "models"
        
        self.server_bin = self.bin_dir / ("llama-server.exe" if self.is_windows else "llama-server")
        self.hf_repo = "unsloth/gemma-3-4b-it-GGUF"
        self.model_filename = "gemma-3-4b-it-Q4_K_M.gguf"
        self.mmproj_filename = "mmproj-F16.gguf"
        
        self.llm_path = self.models_dir / self.model_filename
        self.mmproj_path = self.models_dir / self.mmproj_filename 
        
        self.llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"