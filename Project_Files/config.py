"""
Configuration and environment variables for MedAssist.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"   # FAISS replaces ChromaDB (Python 3.14 compat)
CHROMA_DB_DIR = BASE_DIR / "chroma_db"       # kept for legacy reference
SRC_DIR = BASE_DIR / "src"

# ── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2

# ── Embedding Settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── ChromaDB Settings ─────────────────────────────────────────────────────────
CHROMA_COLLECTION = "medical_knowledge"

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retrieval Settings ────────────────────────────────────────────────────────
RETRIEVAL_K = 5

# ── API Rate Limiting ─────────────────────────────────────────────────────────
API_DELAY_SECONDS = 1.0
API_TIMEOUT_SECONDS = 30

# ── UI Colors ─────────────────────────────────────────────────────────────────
COLOR_PRIMARY = "#1a73e8"   # Medical blue
COLOR_SECONDARY = "#34a853" # Health green
COLOR_WARNING = "#fbbc04"   # Caution yellow
COLOR_DANGER = "#ea4335"    # Alert red
COLOR_BG = "#f8f9fa"        # Light gray background
COLOR_CARD = "#ffffff"      # Card background
