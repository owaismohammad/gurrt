from gurrt.config.config import Settings
from gurrt.utils.utils import rerank, rerank_docs, caption_frame_collection
from gurrt.core.vectordb import VectorDB
import torch

class SearchService:
    def __init__(self,
                clip_model,
                clip_processor,
                reranker,
                vectordb: VectorDB,
                settings: Settings):
        self.model = clip_model
        self.processor = clip_processor
        self.reranker = reranker
        
        self.settings = settings
        self.cache_dir = self.settings.MODEL_CACHE_DIR
        self.db = vectordb

    def _embed_text(self,
                    query,
                    device):
        text_embedding  = self.processor(text = [query], return_tensors = 'pt').to(device)
    
        with torch.no_grad():
            output = self.model.get_text_features(**text_embedding)
        text_features = output.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()[0]  # shape (512,)
        return text_features

    def query_collection(self,
                        device,
                        query: str,
                        top_k: int = 5,
                        candidate_k: int = 40
                        ):
        """Retrieve a wide candidate pool, then let the cross-encoder pick top_k.

        Fetching and reranking the same number is a no-op: the reranker only
        adds value when it has more candidates than it is allowed to keep.
        """
        self.text_features = self._embed_text(query=query,
                                            device = device)
        results = self.db.search_frame(query_embedding= self.text_features,
                                        n_results= candidate_k)
        results_audio =self.db.search_audio(query_embedding= self.text_features,
                                            n_results= candidate_k)
        results_reranked = rerank(query,
                                results,
                                self.reranker,
                                top_k)
        results_reranked_audio = rerank_docs(query,
                                            results_audio,
                                            self.reranker,
                                            top_k)
        captions_list = caption_frame_collection(results_reranked)
        asr_list = [
            {"text": doc,
             "start_sec": meta.get("start_sec"),
             "end_sec": meta.get("end_sec")}
            for doc, meta in zip(results_reranked_audio["documents"][0],
                                 results_reranked_audio["metadatas"][0])
        ]
        return captions_list, asr_list


