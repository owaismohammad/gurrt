from pathlib import Path
from platformdirs import user_config_dir
import json
import sys
class Settings:
    def __init__(self):
        home = Path(user_config_dir("gurrt"))   #/home/yourname/.config/gurrt ->linux
        home.mkdir(exist_ok=True, parents= True)
        
        config_file = home / "config.json"
        cfg = {}
        
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self.OPENROUTER_API_KEY = cfg.get("OPENROUTER_API_KEY")
            self.SUPERMEMORY_API_KEY = cfg.get("SUPERMEMORY_API_KEY")
        else:
            raise RuntimeError("API Keys not found")

        if not self.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found - run /init to save it."
            )
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        # Every collection is keyed by text (captions and transcript alike), so
        # one embedder serves both the index and the query side.
        self.TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
        self.RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.BLIP_MODEL = "Salesforce/blip-image-captioning-large"

        # OpenRouter. Retrieved context is sent as-is; the only levers on how
        # much goes to the model are how many hits are kept below.
        self.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        self.LLM_MODEL = "google/gemma-4-26b-a4b-it:free"
        self.LLM_TIMEOUT_SEC = 180

        self.ASK_TOP_K = 5           # hits kept after reranking
        self.ASK_CANDIDATE_K = 40    # hits fetched before reranking

        # BLIP and SmolVLM describe the room rather than the board: they cannot
        # read slide text, so their captions are near-useless for lecture
        # content. When an index was built with one of them, the prompt tells
        # the model to reason from speech rather than trust the visuals.
        self.LOW_FIDELITY_CAPTIONERS = {"blip2", "smolvlm"}

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

# class LlamaServerManager:
#     def __init__(self):
#         self.config_dir = Path(user_config_dir("gurrt"))
        
#         self.is_windows = sys.platform == "win32"
        
#         self.bin_dir = self.config_dir / "bin"
#         self.models_dir = self.config_dir / "models"
        
#         self.server_bin = self.bin_dir / ("llama-server.exe" if self.is_windows else "llama-server")
#         self.hf_repo = "unsloth/gemma-3-4b-it-GGUF"
#         self.model_filename = "gemma-3-4b-it-Q4_K_M.gguf"
#         self.mmproj_filename = "mmproj-F16.gguf"
        
#         self.llm_path = self.models_dir / self.model_filename
#         self.mmproj_path = self.models_dir / self.mmproj_filename 
        
#         self.llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


class LlamaServerManager:
    def __init__(self, config_dir):
        self.os = sys.platform
        llama = ""
        if self.os == "win32":
            llama = "llama-server.exe"
            self.llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        elif self.os == "linux":
            llama = "llama-server"
            self.llama_release_url = "https://drive.google.com/file/d/1LcwCvXAFfMmSzT-jXbAlBqVy8xVoCesR/view?usp=drive_link"
        self.bin_dir = config_dir / "bin"
        self.server_bin = self.bin_dir / llama
        
        
        # self.config_dir = Path(user_config_dir("gurrt"))
        # self.is_windows = sys.platform == "win32"
        # self.server_bin = self.bin_dir / ("llama-server.exe" if self.is_windows else "llama-server")
        
        
        self.models_dir = config_dir / "models"
        self.hf_repo = "unsloth/gemma-3-4b-it-GGUF"
        self.model_filename = "gemma-3-4b-it-Q4_K_M.gguf"
        self.mmproj_filename = "mmproj-F16.gguf"

        self.llm_path = self.models_dir / self.model_filename
        self.mmproj_path = self.models_dir / self.mmproj_filename 
        # self.llama_release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        
