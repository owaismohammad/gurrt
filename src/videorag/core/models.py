import torch
from transformers import  CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import CrossEncoder
from faster_whisper import WhisperModel

from videorag.config.config import Settings

class ModelManager:
    def __init__(self, settings: Settings):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.settings = settings
        self.cache = self.settings.MODEL_CACHE_DIR
        
        self._clip = None
        self._clip_processor = None
        
        self._blip = None
        self._blip_processor = None
        
        self._whisper = None
        self._reranker = None
        
    def _to_device(self, model):
        return model.to(self.device)
    
    def _free_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    def get_clip(self):
        path = self.cache / "clip_model"
        
        self._clip_processor = CLIPProcessor.from_pretrained(path, local_files_only= True)
        self._clip = CLIPModel.from_pretrained(path, local_files_only= True)
        
        return self._to_device(self._clip), self._clip_processor
    
    def release_clip(self):
        self._clip = None
        self._clip_processor = None
        self._free_gpu()
        
    def get_blip(self):
        path = self.cache / "blip_model"
        
        self._blip_processor = BlipProcessor.from_pretrained(path, local_files_only= True)
        self._blip = BlipForConditionalGeneration.from_pretrained(path, local_files_only= True)
        return self._to_device(self._blip), self._blip_processor
    
    def release_blip(self):
        if self._blip is not None:
            self._blip.to("cpu")
            del self._blip
        self._blip = None
        self._blip_processor = None
        self._free_gpu()
        
    def get_whisper(self):
        self._whisper = WhisperModel(str(self.settings.WHISPER_MODEL),
                                     device= self.device,
                                     compute_type="int8_float16")
        return self._whisper
    
    def release_whisper(self):
        self._whisper = None
        self._free_gpu()
        
    def get_reranker(self):
        path = self.cache / "reranker_model"
        self._reranker = CrossEncoder(str(path))
        return self._reranker
        
    def release_all(self):
        self._clip = None
        self._clip_processor = None
        
        self._blip = None
        self._blip_processor = None
        
        self._whisper = None
        self._reranker = None
        self._free_gpu()
        
def download_models(cache_dir):
    print("Downloading CLIP....")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors = True)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip.save_pretrained(cache_dir / "clip_model")
    proc.save_pretrained(cache_dir / "clip_model")

    print("Downloading BLIP....")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors= True)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip.save_pretrained(cache_dir / "blip_model")
    blip_proc.save_pretrained(cache_dir / "blip_model")
    print("Downloading Reranker....")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker.save(str(cache_dir / "reranker_model"))