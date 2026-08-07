from gurrt.config.config import Settings
from gurrt.utils.utils import (rerank, rerank_docs, caption_frame_collection,
                               embed_texts)
from gurrt.core.vectordb import VectorDB

class SearchService:
    def __init__(self,
                text_embedder,
                reranker,
                vectordb: VectorDB,
                settings: Settings):
        self.text_embedder = text_embedder
        self.reranker = reranker

        self.settings = settings
        self.cache_dir = self.settings.MODEL_CACHE_DIR
        self.db = vectordb

    def _embed_text(self,
                    query,
                    device):
        """One query vector searches both collections.

        Frame captions and transcript chunks are embedded by the same model at
        index time, so they share a space and need only one query embedding.
        """
        return embed_texts([query], self.text_embedder)[0]

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


