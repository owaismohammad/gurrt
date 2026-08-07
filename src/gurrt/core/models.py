import torch
from transformers import  (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration, 
    SmolVLMProcessor, 
    SmolVLMForConditionalGeneration)
from sentence_transformers import CrossEncoder, SentenceTransformer
from faster_whisper import WhisperModel, BatchedInferencePipeline
from huggingface_hub import snapshot_download

from gurrt.config.config import Settings
from gurrt.cli import ui
from gurrt.utils.downloads import (watch_download, hf_cache_dir_for,
                                   hf_repo_size)

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
        self._text_embedder = None

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
        
    def get_text_embedder(self):
        path = self.cache / "text_embed_model"
        self._text_embedder = SentenceTransformer(str(path), device=self.device)
        return self._text_embedder

    def release_text_embedder(self):
        self._text_embedder = None
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
        self._text_embedder = None
        self._free_gpu()
        
def _fetch(repo_id: str, label: str, worker, watch_dir=None):
    """Download one model with a real byte-progress bar.

    huggingface_hub gives no progress callback, so the transfer runs on a
    worker thread while we watch its cache directory fill.
    """
    watch_download(
        description=f"  {label}",
        worker=worker,
        watch_dir=watch_dir or hf_cache_dir_for(repo_id),
        total_bytes=hf_repo_size(repo_id),
    )


def download_models(cache_dir):
    clip_id = "openai/clip-vit-base-patch32"
    blip_id = "Salesforce/blip-image-captioning-base"
    smol_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    embed_id = "BAAI/bge-small-en-v1.5"
    rerank_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    whisper_id = "Systran/faster-distil-whisper-large-v2"

    def _clip():
        clip = CLIPModel.from_pretrained(clip_id, use_safetensors=True)
        proc = CLIPProcessor.from_pretrained(clip_id)
        clip.save_pretrained(cache_dir / "clip_model")
        proc.save_pretrained(cache_dir / "clip_model")

    def _blip():
        blip = BlipForConditionalGeneration.from_pretrained(blip_id, use_safetensors=True)
        blip_proc = BlipProcessor.from_pretrained(blip_id)
        blip.save_pretrained(cache_dir / "blip_model")
        blip_proc.save_pretrained(cache_dir / "blip_model")

    def _smol():
        smolVLM = SmolVLMForConditionalGeneration.from_pretrained(smol_id, use_safetensors=True)
        smolVLM_proc = SmolVLMProcessor.from_pretrained(smol_id)
        smolVLM.save_pretrained(cache_dir / "smolVLM_model")
        smolVLM_proc.save_pretrained(cache_dir / "smolVLM_model")

    def _embed():
        SentenceTransformer(embed_id).save(str(cache_dir / "text_embed_model"))

    def _rerank():
        CrossEncoder(rerank_id).save(str(cache_dir / "reranker_model"))

    def _whisper():
        snapshot_download(repo_id=whisper_id,
                          local_dir=str(cache_dir / "whisper_model"))

    _fetch(clip_id, "CLIP", _clip)
    _fetch(blip_id, "BLIP", _blip)
    _fetch(smol_id, "SmolVLM", _smol)
    _fetch(embed_id, "Text embedder", _embed)
    _fetch(rerank_id, "Reranker", _rerank)
    # snapshot_download writes straight to the target, so watch that instead
    # of the shared cache.
    _fetch(whisper_id, "Faster Whisper", _whisper,
           watch_dir=cache_dir / "whisper_model")
    ui.success("All models downloaded")