import chromadb

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
        
    def _reset_collection(self):
        try:
            self.client.delete_collection("frame_embedding_collection")
            self.client.delete_collection("asr_collection")
            print("\033[1;32mRefreshing Collection\033[0m")
        except:
            print("\033[1;32mInitializing New Collection\033[0m")
        