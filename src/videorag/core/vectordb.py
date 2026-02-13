import chromadb

class VectorDB:
    def __init__(self, db_path:str):
        
        self.client = chromadb.PersistentClient(path = db_path)
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