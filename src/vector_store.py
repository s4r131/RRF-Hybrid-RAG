from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from functools import lru_cache

from src.config import(CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Initialize and return a HuggingFaceEmbeddings model for vectorizing text."""
    try:
        return HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {"local_files_only": True},
            encode_kwargs = {"normalize_embeddings": True}
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name = model_name,
            encode_kwargs = {"normalize_embeddings": True}
        )

def create_vector_store(chunks: List[Document], batch_size: int = 500) -> Chroma:
    """Embed document chunks in batches and persist them to ChromaDB on disk."""
    embedding_model = get_embedding_model()

    vector_store = Chroma(collection_name = COLLECTION_NAME, embedding_function = embedding_model, persist_directory = str(CHROMA_DB_DIR))

    total = len(chunks)
    print(f"Indexing {total} chunks into ChromaDB at '{CHROMA_DB_DIR}'...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f" -> Indexed {min(i + batch_size, total)} / {total} chunks")

    print("Vector store creation complete!")
    return vector_store

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Load the existing persisted ChromaDB vector store from disk"""
    embedding_model = get_embedding_model()

    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=str(CHROMA_DB_DIR))
