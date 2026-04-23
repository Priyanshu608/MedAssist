"""
Ingestion pipeline — splits documents into chunks for embedding.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into embedding-sized chunks.

    - chunk_size=500, chunk_overlap=50
    - Metadata (source, type, name, fetch_date) is preserved on every chunk
    - chunk_index is added to metadata for ordering

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
        chunks.extend(doc_chunks)

    return chunks
