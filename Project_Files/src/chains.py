"""
RAG chain builders for each MedAssist mode.

Each builder returns a RunnableParallel that yields:
    {
        "answer":           str   — LLM response
        "source_documents": list  — retrieved Document objects for citation
    }

invoke_chain() wraps every call with:
  - Groq 429 / network error detection → user-friendly messages
  - Empty-retrieval guard → honest "not enough info" reply
  - Full error classification so the UI can show appropriate UI state
"""
import logging

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS

from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_K
from src.prompts import MEDICAL_QA_PROMPT, SYMPTOM_CHECKER_PROMPT, DRUG_INTERACTION_PROMPT

logger = logging.getLogger(__name__)

# ── Singleton LLM ─────────────────────────────────────────────────────────────
_llm: ChatGroq | None = None

# Minimum number of *real* source documents before we trust the answer
_MIN_REAL_SOURCES = 1

_NO_INFO_REPLY = (
    "I don't have enough information in my knowledge base to answer this question "
    "confidently.\n\n"
    "**Suggestions:**\n"
    "- Try rephrasing the question with different keywords.\n"
    "- For drug questions use generic names (e.g. *ibuprofen* not *Advil*).\n"
    "- If the knowledge base hasn't been built yet, run the setup script.\n\n"
    "*Always consult a licensed healthcare professional for medical advice.*"
)


def _get_llm() -> ChatGroq:
    """Return (or create) the shared ChatGroq instance with two-layer retry."""
    global _llm
    if _llm is None:
        base = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            groq_api_key=GROQ_API_KEY,
            max_retries=3,          # Groq SDK exponential back-off (handles 429)
        )
        # LangChain-level wrapper: adds jitter + one more attempt layer
        _llm = base.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )
    return _llm


# ══════════════════════════════════════════════════════════════════════════════
# Public chain builders
# ══════════════════════════════════════════════════════════════════════════════

def get_medical_qa_chain(vectorstore: FAISS):
    """RAG chain for Medical Q&A — retrieves from all document types."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )
    return _build_chain(retriever, MEDICAL_QA_PROMPT, vectorstore, k=RETRIEVAL_K)


def get_symptom_chain(vectorstore: FAISS):
    """RAG chain for Symptom Checker — filters to disease + health_topic docs."""
    _filter = lambda meta: meta.get("type") in ("disease", "health_topic")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": _filter,
            "fetch_k": 40,
        },
    )
    return _build_chain(retriever, SYMPTOM_CHECKER_PROMPT, vectorstore,
                        filter_fn=_filter, fetch_k=40, k=RETRIEVAL_K)


def get_drug_chain(vectorstore: FAISS):
    """RAG chain for Drug Interactions — filters to drug docs only."""
    _filter = lambda meta: meta.get("type") == "drug"
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": _filter,
            "fetch_k": 60,
        },
    )
    return _build_chain(retriever, DRUG_INTERACTION_PROMPT, vectorstore,
                        filter_fn=_filter, fetch_k=60, k=RETRIEVAL_K)


# ══════════════════════════════════════════════════════════════════════════════
# Public invoke wrapper
# ══════════════════════════════════════════════════════════════════════════════

def invoke_chain(chain, query: str) -> dict:
    """
    Invoke a RAG chain with full error handling.

    Returns a dict:
        answer           : str  — LLM response (or friendly error/no-info message)
        source_documents : list — retrieved Document objects
        error            : str | None — raw error string if something failed
        error_type       : str | None — "rate_limit" | "auth" | "timeout" |
                                        "network" | "context_length" | "unknown"
    """
    try:
        result = chain.invoke(query)
        answer  = result.get("answer", "")
        sources = result.get("source_documents", [])

        # ── Empty / init-only retrieval guard ──────────────────────────────
        real_sources = [s for s in sources if s.metadata.get("type") != "init"]
        if len(real_sources) < _MIN_REAL_SOURCES:
            logger.warning("Retrieval returned no meaningful documents for query: %r", query[:80])
            return {
                "answer": _NO_INFO_REPLY,
                "source_documents": [],
                "error": None,
                "error_type": None,
            }

        return {
            "answer": answer,
            "source_documents": sources,
            "error": None,
            "error_type": None,
        }

    except Exception as exc:
        raw = str(exc)
        etype = _classify_error(raw)
        friendly = _friendly_error(etype, raw)
        logger.error("Chain invocation failed [%s]: %s", etype, exc)
        return {
            "answer": "",
            "source_documents": [],
            "error": friendly,
            "error_type": etype,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Error helpers
# ══════════════════════════════════════════════════════════════════════════════

def _classify_error(error_str: str) -> str:
    e = error_str.lower()
    if "429" in e or "rate_limit" in e or "rate limit" in e or "too many requests" in e:
        return "rate_limit"
    if "401" in e or "invalid api key" in e or "authentication" in e or "api_key" in e:
        return "auth"
    if "timeout" in e or "timed out" in e or "read timeout" in e:
        return "timeout"
    if "connection" in e or "network" in e or "connect" in e or "unreachable" in e:
        return "network"
    if "context_length" in e or "too many tokens" in e or "max_tokens" in e:
        return "context_length"
    return "unknown"


def _friendly_error(error_type: str, raw: str) -> str:
    msgs = {
        "rate_limit": (
            "The AI service is temporarily rate-limited (too many requests).\n"
            "Please wait **30 seconds** and try again."
        ),
        "auth": (
            "Invalid Groq API key.\n"
            "Please check your `.env` file and ensure `GROQ_API_KEY` is set correctly.\n"
            "Get a free key at https://console.groq.com"
        ),
        "timeout": (
            "The request timed out. The AI service may be busy.\n"
            "Please try again in a moment."
        ),
        "network": (
            "Could not connect to the AI service.\n"
            "Please check your internet connection and try again."
        ),
        "context_length": (
            "Your query is too long for the model to process.\n"
            "Please shorten your question and try again."
        ),
        "unknown": f"An unexpected error occurred. Please try again.\n\n*Details:* `{raw[:200]}`",
    }
    return msgs.get(error_type, msgs["unknown"])


# ══════════════════════════════════════════════════════════════════════════════
# Internal chain assembly
# ══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: list) -> str:
    """Format retrieved documents into a context block with source labels."""
    parts = []
    for doc in docs:
        src  = doc.metadata.get("source", "Unknown")
        name = doc.metadata.get("name", "unknown")
        parts.append(f"[Source: {src} — {name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_chain(retriever, prompt, vectorstore, filter_fn=None, fetch_k=20, k=RETRIEVAL_K):
    """
    Build a RunnableParallel that returns answer + source_documents in one call.

    source_documents are fetched via similarity_search_with_score so that each
    Document's metadata gains a 'relevance_score' key (0–100 float) that the UI
    can display as a coloured relevance badge.

    Conversion: FAISS returns L2 distance d for normalised embeddings.
    cosine_similarity = 1 - d² / 2  →  relevance_score = clamp(cosine_sim, 0, 1) × 100
    """
    llm = _get_llm()

    def _retrieve_scored(query: str) -> list:
        kwargs: dict = {"k": k, "fetch_k": fetch_k}
        if filter_fn is not None:
            kwargs["filter"] = filter_fn
        pairs = vectorstore.similarity_search_with_score(query, **kwargs)
        docs = []
        for doc, dist in pairs:
            cosine_sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
            doc.metadata["relevance_score"] = round(cosine_sim * 100, 1)
            docs.append(doc)
        return docs

    answer_chain = (
        {
            "context":  retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        answer=answer_chain,
        source_documents=RunnableLambda(_retrieve_scored),
    )
