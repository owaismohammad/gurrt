# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from dotenv import load_dotenv
# from transformers import CLIPProcessor, CLIPModel
from videorag.config.config import Settings
from videorag.utils.utils import device, rerank, rerank_docs, caption_frame_collection
from videorag.core.vectordb import VectorDB
import torch
# load_dotenv()

# MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")


# if MODEL_CACHE_DIR is None:
#     raise RuntimeError("MODEL_CACHE_DIR path not found")

# clip_path = os.path.join(MODEL_CACHE_DIR, "clip_model")
# model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
# processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)


class SearchService:
    def __init__(self, clip_model, clip_processor, cache_dir: str):
        self.model = clip_model
        self.processor = clip_processor
        self.cache_dir = cache_dir
        self.settings = Settings()
        self.db = VectorDB(str(self.settings.CHROMA_DB_PATH))
        
    def _embed_text(self, query):
        text_embedding  = self.processor(text = [query], return_tensors = 'pt').to(device)
    
        with torch.no_grad():
            output = self.model.get_text_features(**text_embedding)

        text_features = output.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()[0]  # shape (512,)
        
        return text_features
    def query_collection(self, query: str, n_results: int = 10):
        
        self.text_features = self._embed_text(query=query)
        
        results = self.db.search_frame(
        query_embedding= self.text_features,
        n_results= n_results
    )   
        results_audio =self.db.search_audio(
        query_embedding= self.text_features,
        n_results= n_results
    )   
        
        results_reranked = rerank(query, results, self.cache_dir, n_results)
        results_reranked_audio = rerank_docs(query, results_audio, self.cache_dir, n_results)
        
        captions_list = caption_frame_collection(results_reranked)
        asr_list = results_reranked_audio["documents"][0]
        
        return captions_list, asr_list


