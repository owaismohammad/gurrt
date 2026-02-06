import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
db_path = os.getenv('CHROMA_DB_PATH')

client = chromadb.PersistentClient(path = db_path)

frame_embedding_collection = client.get_or_create_collection(
    name="frame_embedding_collection",
    metadata={"hnsw:space": "cosine"}
)
frame_caption_collection = client.get_or_create_collection(
    name="frame_caption_collection",
    metadata={"hnsw:space": "cosine"}
)

asr_collection = client.get_or_create_collection(
    name="asr_collection",
    metadata={"hnsw:space": "cosine"}
)

