import torch
from transformers import  (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration, 
    SmolVLMProcessor, 
    SmolVLMForConditionalGeneration)
from sentence_transformers import CrossEncoder
from faster_whisper import WhisperModel, BatchedInferencePipeline
from huggingface_hub import snapshot_download

from gurrt.config.config import Settings
from gurrt.cli import ui

class ModelManager:
    def __init__(self, settings: Settings):
        
        self.device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[1]>= 4* 10**9 else "cpu"
        ui.info(f"Running on {self.device.upper()}")
        self.settings = settings
        self.cache = self.settings.MODEL_CACHE_DIR
        
        self._clip = None
        self._clip_processor = None
        
        # self._blip = None
        # self._blip_processor = None
        
        self._smol = None
        self._smol_processor = None
        
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
    
    def get_smol(self, flag:bool):
        path = self.cache / "smolVLM_model"
        
        self._smol_processor = SmolVLMProcessor.from_pretrained(path, local_files_only= True)
        if not flag:
            ui.warn("GPU memory < 4 GB — disabling image splitting for SmolVLM")
            self._smol_processor.image_processor.do_image_splitting = False

        self._smol = SmolVLMForConditionalGeneration.from_pretrained(path,
                                                                    local_files_only= True)
        self._smol = torch.compile(self._smol, mode="reduce-overhead")
        return self._to_device(self._smol), self._smol_processor

    def release_smol(self):
        if self._smol is not None:
            self._smol.to("cpu")
            del self._smol
        self._smol = None
        self._smol_processor = None
        self._free_gpu()

    def get_whisper(self):
        path = self.cache / "whisper_model"
        if self.device == "cuda":
            compute_type = "int8_float16"
        else:
            compute_type = "int8"
        self._whisper = WhisperModel(str(path),
                                    device= self.device,
                                    compute_type=compute_type)
        batched = BatchedInferencePipeline(model=self._whisper)
        return batched
    
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
    ui.step("Downloading CLIP...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip.save_pretrained(cache_dir / "clip_model")
    proc.save_pretrained(cache_dir / "clip_model")

    ui.step("Downloading BLIP...")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip.save_pretrained(cache_dir / "blip_model")
    blip_proc.save_pretrained(cache_dir / "blip_model")

    ui.step("Downloading SmolVLM...")
    smolVLM = SmolVLMForConditionalGeneration.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", use_safetensors=True)
    smolVLM_proc = SmolVLMProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    smolVLM.save_pretrained(cache_dir / "smolVLM_model")
    smolVLM_proc.save_pretrained(cache_dir / "smolVLM_model")

    ui.step("Downloading Reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker.save(str(cache_dir / "reranker_model"))

    ui.step("Downloading Faster Whisper...")
    snapshot_download(
        repo_id="Systran/faster-distil-whisper-large-v2",
        local_dir=str(cache_dir / "whisper_model"),
    )