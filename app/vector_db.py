import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
db_path = os.getenv('CHROMA_DB_PATH') or None

if db_path is None:
    raise RuntimeError("CHROMA_DB_PATH is not set")

client = chromadb.PersistentClient(path = db_path)

frame_embedding_collection = client.get_or_create_collection(
    name="frame_embedding_collection",
    metadata={"hnsw:space": "cosine"}
)

asr_collection = client.get_or_create_collection(
    name="asr_collection",
    metadata={"hnsw:space": "cosine"}
)

