from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from src.config import DATA_DIR, EXCLUDED_DIRS

def should_skip_path(file_path: Path) -> bool:
    """Return True if the file sits inside any excluded directory, False otherwise."""
    return bool(set(file_path.parts) & EXCLUDED_DIRS)

def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Scan data_dir for PDFs, skip excluded directories, and enrich with category metadata."""
    all_documents: List[Document] = []

    # Recursively find all .pdf files in DATA_DIR and sort them alphabetically
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} total PDF files. Filtering and loading....")

    for pdf_path in pdf_files:
        # 1. Skip if inside any excluded directory
        if should_skip_path(pdf_path):
            continue

        # 2. Extract the top-level category/section folder name from the relative path
        rel_path = pdf_path.relative_to(data_dir)
        category_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "General"

        # 3. Load the PDF and create a Document with metadata
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                # Skip completely empty pages (no text, no images)
                if not page.page_content.strip():
                    continue

                page.metadata["file_name"] = pdf_path.name
                page.metadata["category"] = category_name
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1

                all_documents.append(page)
        except Exception as e:
            print(f"[Warning] failed to load {pdf_path.name}: {e}")

    print(f"Loaded {len(all_documents)} non-empty pages from {data_dir.name}.")
    return all_documents