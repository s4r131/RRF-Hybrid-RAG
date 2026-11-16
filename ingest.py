import shutil
from src.config import CHROMA_DB_DIR
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store

def run_ingestion(reset_db: bool = True) -> None:
    """Run the full document ingestion pipeline: Load -> Chunk -> Embed & Store."""
    if reset_db and CHROMA_DB_DIR.exists():
        print(f"Removing existing vector store at '{CHROMA_DB_DIR}'...")
        shutil.rmtree(CHROMA_DB_DIR)

    # Step 1: Load PDFs and attach metadata
    documents = load_documents()
    if not documents:
        print("No documents found to ingest. Exiting.")
        return

    # Step 2: Split documents into overlapping chunks.
    chunks = chunk_documents(documents)

    # Step 3: Embed chunks and save to ChromaDB
    create_vector_store(chunks)
    print("Ingestion pipeline finished successfully!")

if __name__ == "__main__":
    run_ingestion()