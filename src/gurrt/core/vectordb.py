import chromadb
from gurrt.cli import ui

class VectorDB:
    def __init__(self, db_path:str, reset: bool = False):
        
        
        self.client = chromadb.PersistentClient(path = db_path)
        if reset:
            self._reset_collection()
        self.caption_collection = self.client.get_or_create_collection(
            name="frame_embedding_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.asr_collection = self.client.get_or_create_collection(
            name="asr_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_frames(self, ids, embeddings, metadata):
        self.caption_collection.add(
            ids= ids,
            embeddings= embeddings,
            metadatas= metadata
        )

    def add_asr(self, ids, embeddings, metadata, documents):
        self.asr_collection.add(
            ids = ids,
            embeddings= embeddings,
            metadatas= metadata,
            documents=documents
        )
    
    def search_frame(self, query_embedding, n_results:int):
        return self.caption_collection.query(
        query_embeddings=[query_embedding],
        n_results= n_results
    )
        
    def search_audio(self, query_embedding, n_results:int):
        return self.asr_collection.query(
        query_embeddings=[query_embedding],
        n_results= n_results
    )   
        
    def _in_range(self, collection, start_sec: float, end_sec: float):
        """Rows whose [start_sec, end_sec] overlaps the requested span.

        Overlap, not nearest-timestamp: a slide held for three minutes must
        match everything said during it, not only the instant it appeared.
        """
        try:
            return collection.get(
                where={"$and": [{"start_sec": {"$lte": end_sec}},
                                {"end_sec": {"$gte": start_sec}}]},
                include=["metadatas", "documents"],
            )
        except Exception:
            return {"metadatas": [], "documents": [], "ids": []}

    def frames_in_range(self, start_sec: float, end_sec: float):
        return self._in_range(self.caption_collection, start_sec, end_sec)

    def audio_in_range(self, start_sec: float, end_sec: float):
        return self._in_range(self.asr_collection, start_sec, end_sec)

    def _reset_collection(self):
        try:
            self.client.delete_collection("frame_embedding_collection")
            self.client.delete_collection("asr_collection")
            ui.info("Vector database reset")
        except Exception:
            pass
        