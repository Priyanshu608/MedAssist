"""
Vector store management using FAISS + HuggingFace embeddings.

Note: ChromaDB 1.5.x is incompatible with Python 3.14 (pydantic v1 shim broken).
FAISS is used as the persistent vector store instead — same LangChain API surface.

Persistence layout:
    faiss_index/
        index.faiss   — FAISS index file
        index.pkl     — docstore + metadata
"""
from pathlib import Path
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once per process
_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load (or return cached) HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        print("  Loading embedding model (first run may download ~80 MB)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  Embedding model ready.")
    return _embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_vectorstore() -> FAISS:
    """
    Load existing FAISS index from disk, or create an empty one.
    Returns the FAISS vectorstore instance.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = _get_embeddings()
    index_path = Path(FAISS_INDEX_DIR)

    if (index_path / "index.faiss").exists():
        logger.info("Loading existing FAISS index from %s", index_path)
        _vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("No existing index found — creating empty FAISS store")
        # Seed with a single dummy document so FAISS initialises correctly.
        # The dummy doc is never returned in real queries (content is unique/nonsensical).
        dummy = Document(
            page_content="__medassist_init__",
            metadata={"source": "init", "type": "init", "name": "init"},
        )
        _vectorstore = FAISS.from_documents([dummy], embeddings)

    return _vectorstore


def add_documents(vectorstore: FAISS, documents: list[Document], batch_size: int = 200) -> None:
    """
    Embed and add documents to the FAISS index, then persist to disk.

    Args:
        vectorstore: The FAISS instance returned by initialize_vectorstore().
        documents:   List of LangChain Document objects to add.
        batch_size:  Number of documents to embed per batch.
    """
    total = len(documents)
    print(f"  Embedding {total} chunks in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        vectorstore.add_documents(batch)
        done = min(start + batch_size, total)
        print(f"    [{done}/{total}] embedded")

    _save(vectorstore)
    print(f"  Index saved to {FAISS_INDEX_DIR}/")


def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = RETRIEVAL_K,
    filter: dict | None = None,
) -> list[Document]:
    """Return the k most relevant documents for a query."""
    return vectorstore.similarity_search(query, k=k, filter=filter)


def get_retriever(vectorstore: FAISS | None = None, search_type: str = "similarity", k: int = RETRIEVAL_K):
    """
    Return a LangChain retriever.
    Loads the index from disk if no vectorstore is passed.
    """
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def is_populated() -> bool:
    """
    Lightweight check — returns True if a saved FAISS index exists on disk
    with more than the single dummy init document.
    """
    index_path = Path(FAISS_INDEX_DIR)
    index_file = index_path / "index.faiss"
    pkl_file   = index_path / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        return False

    # Size guard: a real index is always larger than the 1-doc dummy seed
    return index_file.stat().st_size > 8_192


def reset_vectorstore() -> None:
    """Delete the saved FAISS index so setup_knowledge_base can rebuild it."""
    import shutil
    global _vectorstore
    _vectorstore = None
    index_path = Path(FAISS_INDEX_DIR)
    if index_path.exists():
        shutil.rmtree(index_path)
        logger.info("FAISS index deleted.")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(vectorstore: FAISS) -> None:
    """Persist the FAISS index to disk."""
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
