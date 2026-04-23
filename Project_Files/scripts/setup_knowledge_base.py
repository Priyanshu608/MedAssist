"""
One-command knowledge base builder for MedAssist.

Usage (from project root with venv active):
    python scripts/setup_knowledge_base.py

Steps:
    1. Fetch medical data from OpenFDA, PubMed, MedlinePlus
    2. Chunk documents with RecursiveCharacterTextSplitter
    3. Embed with all-MiniLM-L6-v2
    4. Store in FAISS (persistent local index)
    5. Run a test retrieval query
"""
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import MedicalDataFetcher
from src.ingestion import chunk_documents
from src.vectorstore import (
    initialize_vectorstore,
    add_documents,
    get_retriever,
    is_populated,
    reset_vectorstore,
)


def main() -> None:
    print("=" * 55)
    print("  MedAssist - Knowledge Base Setup")
    print("=" * 55)

    # ── Check existing index ──────────────────────────────────────────────────
    if is_populated():
        print("\nAn existing knowledge base was found.")
        answer = input("Rebuild from scratch? [y/N]: ").strip().lower()
        if answer != "y":
            print("Keeping existing index. Run the app with: streamlit run app.py")
            return
        print("Deleting existing index...")
        reset_vectorstore()

    start_time = time.time()

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("\n[Step 1/4] Fetching medical data from public APIs...")
    fetcher = MedicalDataFetcher()  # default delay=0.3 s
    docs = fetcher.fetch_all()
    print(f"\n  Fetched {len(docs)} documents total.")

    if not docs:
        print("ERROR: No documents fetched. Check your internet connection.")
        sys.exit(1)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    print("\n[Step 2/4] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"  {len(docs)} documents -> {len(chunks)} chunks")
    print(f"  (chunk_size={500}, chunk_overlap={50})")

    # ── Step 3: Initialise vector store ──────────────────────────────────────
    print("\n[Step 3/4] Initialising FAISS vector store...")
    vs = initialize_vectorstore()

    # ── Step 4: Embed + store ─────────────────────────────────────────────────
    print("\n[Step 4/4] Embedding and storing chunks...")
    add_documents(vs, chunks)

    elapsed = time.time() - start_time
    print(f"\nKnowledge base built in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Verification retrieval ────────────────────────────────────────────────
    print("\n[Verification] Running test retrieval query...")
    retriever = get_retriever(vs, k=3)
    test_query = "What are the side effects of ibuprofen?"
    results = retriever.invoke(test_query)

    print(f"  Query: '{test_query}'")
    print(f"  Retrieved {len(results)} chunks")
    for i, r in enumerate(results, 1):
        src  = r.metadata.get("source", "?")
        name = r.metadata.get("name", "?")
        preview = r.page_content[:100].replace("\n", " ")
        print(f"  [{i}] {src}/{name}: {preview}...")

    print("\n" + "=" * 55)
    print("  Setup complete! Run the app:")
    print("  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
