from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents: List[Document], chunk_size: int=CHUNK_SIZE, chunk_overlap: int=CHUNK_OVERLAP) -> List[Document]:
    """Split full-page Documents into smaller overlapping chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} full-page documents into {len(chunks)} smaller chunks.")
    return chunks