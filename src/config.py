from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (like API keys) from the .env file in our project root
load_dotenv()

# __file__ is "c:/Users/s4r13/Documents/RAG/src/config.py"
# .parent gives "c:/Users/s4r13/Documents/RAG/src"
# .parent.parent gives "c:/Users/s4r13/Documents/RAG" (our project root!)
BASE_DIR = Path(__file__).resolve().parent.parent

# Define where the raw document files live and where ChromaDB will save vectors
# You can override DATA_DIR in your .env file to point to any dataset folder!
DATA_DIR = BASE_DIR / "RSO Supplemental Materials"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Folders to skip during ingestion so we don't index duplicate files
EXCLUDED_DIRS = {
    "RSO Supplemental Material - Zip File 2020"
}

# Chunking settings (in characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Database & Embedding Model settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "knowledge_base"

# Retrieval settings: number of relevant chunks to send to the LLM per question
TOP_K_RESULTS = 10