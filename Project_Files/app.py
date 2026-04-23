"""
MedAssist — AI-Powered Healthcare Knowledge Assistant
Main Streamlit application.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="MedAssist — AI Healthcare Assistant",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from src.vectorstore import is_populated, initialize_vectorstore, reset_vectorstore
from src.chains import get_medical_qa_chain, get_symptom_chain, get_drug_chain, invoke_chain

# ── Emergency keyword list ────────────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "chest tightness", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath", "trouble breathing",
    "heart attack", "stroke", "severe bleeding", "unconscious", "fainted",
    "overdose", "suicidal", "anaphylaxis", "severe allergic", "choking",
    "loss of consciousness", "sudden confusion", "facial drooping",
    "arm weakness", "slurred speech", "sudden severe headache",
]

# ── SVG logo ──────────────────────────────────────────────────────────────────
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" '
    'width="{size}" height="{size}">'
    '<defs><clipPath id="mc-clip"><circle cx="100" cy="100" r="94"/></clipPath></defs>'
    '<circle cx="100" cy="100" r="94" fill="#0F172A"/>'
    '<g clip-path="url(#mc-clip)">'
    '<g stroke="#1E3A8A" stroke-width="1.5" fill="none" stroke-linecap="round">'
    '<polyline points="120,58 120,12 168,12"/>'
    '<polyline points="80,58 80,12 32,12"/>'
    '<polyline points="120,142 120,188 168,188"/>'
    '<polyline points="80,142 80,188 32,188"/>'
    '<polyline points="142,80 188,80 188,32"/>'
    '<polyline points="142,120 188,120 188,168"/>'
    '<polyline points="58,80 12,80 12,32"/>'
    '<polyline points="58,120 12,120 12,168"/>'
    '</g>'
    '<g fill="#3B82F6">'
    '<circle cx="168" cy="12" r="5"/><circle cx="32" cy="12" r="5"/>'
    '<circle cx="168" cy="188" r="5"/><circle cx="32" cy="188" r="5"/>'
    '<circle cx="188" cy="32" r="5"/><circle cx="188" cy="168" r="5"/>'
    '<circle cx="12" cy="32" r="5"/><circle cx="12" cy="168" r="5"/>'
    '</g></g>'
    '<circle cx="100" cy="100" r="76" fill="none" stroke="#1E3A8A" stroke-width="1"/>'
    '<rect x="79" y="44" width="42" height="112" rx="8" fill="white"/>'
    '<rect x="44" y="79" width="112" height="42" rx="8" fill="white"/>'
    '<rect x="79" y="79" width="42" height="42" rx="6" fill="#2563EB"/>'
    '</svg>'
)


def _logo(size: int = 120) -> str:
    return _LOGO_SVG.format(size=size)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp { background-color: #F8FAFC; }

        /* ── Sidebar — dark navy ─────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #0F172A;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
        section[data-testid="stSidebar"] .disclaimer { color: #94A3B8 !important; }
        section[data-testid="stSidebar"] hr {
            border-color: #334155 !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radio"] {
            background: transparent; border-radius: 8px;
            padding: 6px 10px; border: none;
            transition: background 0.15s;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radio"]:hover {
            background: rgba(255,255,255,0.07);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] {
            background: rgba(30,64,175,0.45);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] p {
            color: #93C5FD !important; font-weight: 600;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: #1E3A6E; border: 1px solid #2D4F9E;
            color: #F1F5F9 !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #1E40AF;
        }

        /* ── Page header ─────────────────────────────────────────────────── */
        .page-header {
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 14px; margin-bottom: 12px;
        }
        .page-header h1 {
            margin: 0; font-size: 1.5rem; font-weight: 700; color: #0F172A;
        }
        .page-header p {
            margin: 4px 0 0; font-size: 0.85rem; color: #64748B;
        }

        /* ── Info cards ──────────────────────────────────────────────────── */
        .info-card {
            background: #FFFFFF; border-radius: 8px; padding: 12px 16px;
            margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #E2E8F0; color: #1E293B;
        }
        .info-card.yellow {
            background: #FFFBEB; border-color: #FDE68A; color: #78350F;
        }
        .info-card.red {
            background: #FEF2F2; border-color: #FECACA; color: #991B1B;
        }

        /* ── Emergency banner ────────────────────────────────────────────── */
        .emergency-banner {
            background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px;
            padding: 16px 20px; margin-bottom: 16px;
        }
        .emergency-banner h3 { color: #991B1B; margin: 0 0 6px; font-size: 1.05rem; }
        .emergency-banner p  { color: #7F1D1D; margin: 0; font-size: 0.9rem; }

        /* ── Chat messages ───────────────────────────────────────────────── */
        .stChatMessage {
            background: #FFFFFF; border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 8px;
        }

        /* ── Buttons ─────────────────────────────────────────────────────── */
        .stButton > button {
            background: #1E40AF;
            color: white !important; border: none; border-radius: 8px;
            padding: 10px 24px; font-weight: 500; font-size: 0.95rem;
            transition: background 0.15s;
        }
        .stButton > button:hover { background: #1D3A9E !important; color: white !important; }

        /* ── Inputs ──────────────────────────────────────────────────────── */
        .stTextInput > div > div > input,
        .stTextArea textarea {
            border-radius: 8px; border: 1px solid #CBD5E1;
            font-family: 'Inter', sans-serif;
            background: #FFFFFF; color: #1E293B;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: #1E40AF; box-shadow: 0 0 0 3px rgba(30,64,175,0.12);
        }
        .stTextInput label, .stTextArea label {
            color: #1E293B !important; font-weight: 500;
        }

        /* ── Status badges ───────────────────────────────────────────────── */
        .status-badge {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.78rem; font-weight: 600; margin-top: 6px;
        }
        .status-badge.populated { background: #DCFCE7; color: #166534; }
        .status-badge.empty     { background: #FEF9C3; color: #713F12; }
        .status-badge.loading   { background: #DBEAFE; color: #1E40AF; }

        /* ── Severity badges ─────────────────────────────────────────────── */
        .badge-mild     { background:#DCFCE7; color:#166534; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
        .badge-moderate { background:#FEF9C3; color:#713F12; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
        .badge-severe   { background:#FEF2F2; color:#991B1B; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

        /* ── Source card ─────────────────────────────────────────────────── */
        .source-card {
            background: #F8FAFC; border-radius: 8px; padding: 10px 14px;
            margin-bottom: 8px; border: 1px solid #E2E8F0; color: #1E293B;
        }

        /* ── Response meta ───────────────────────────────────────────────── */
        .response-meta { font-size: 0.72rem; color: #64748B; margin-top: 4px; }

        /* ── Disclaimer ──────────────────────────────────────────────────── */
        .disclaimer { font-size: 0.72rem; color: #94A3B8; line-height: 1.5; }

        /* ── Landing page ────────────────────────────────────────────────── */
        .hero-title   { font-size:2.8rem; font-weight:700; color:#0F172A; letter-spacing:-0.03em;
                        margin:16px 0 6px; text-align:center; }
        .hero-tagline { font-size:1.05rem; color:#64748B; text-align:center; margin:0; }
        .about-card   { background:#FFFFFF; border-radius:8px; padding:24px;
                        border:1px solid #E2E8F0; box-shadow:0 1px 4px rgba(0,0,0,0.07); }
        .pipeline-step { background:#F1F5F9; border-radius:8px; padding:16px 12px; text-align:center; }
        .pipeline-num   { font-size:1.5rem; font-weight:700; color:#1E40AF; }
        .pipeline-label { font-size:0.88rem; font-weight:600; color:#0F172A; margin:6px 0 4px; }
        .pipeline-desc  { font-size:0.76rem; color:#64748B; line-height:1.4; }
        .tech-chip  { display:inline-block; background:#DBEAFE; color:#1E40AF;
                      padding:6px 14px; border-radius:20px; font-size:0.83rem; font-weight:500; margin:4px 3px; }
        .source-item    { background:#F8FAFC; border-radius:8px; padding:16px; border:1px solid #E2E8F0; }
        .source-item h4 { margin:0 0 6px; font-size:0.95rem; color:#0F172A; font-weight:600; }
        .source-item p  { margin:0; font-size:0.78rem; color:#64748B; line-height:1.5; }

        /* ── Login card ──────────────────────────────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
            border-color: #E2E8F0 !important;
        }

        /* ── Expander headers — dark text ───────────────────────────────── */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        .streamlit-expanderHeader,
        .streamlit-expanderHeader p {
            color: #1E293B !important;
            font-weight: 500 !important;
        }
        [data-testid="stExpander"] summary svg {
            fill: #1E293B !important;
            color: #1E293B !important;
        }

        /* ── General text contrast in main area ─────────────────────────── */
        .main p, .main li, .main span { color: #1E293B; }
        .stChatMessage p,
        .stChatMessage li,
        .stChatMessage span { color: #1E293B !important; }

        /* ── Reduce default top padding in content area ──────────────────── */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }

        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    defaults: dict = {
        "page":          "landing",
        "authenticated": False,
        "login_error":   False,
        "mode": "Medical Q&A",
        "qa_history": [],
        "symptom_history": [],
        "drug_history": [],
        "symptom_input_text": "",
        "chains_loaded": False,
        "setup_running": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# Chain / model loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_chains() -> None:
    """Load vectorstore + build 3 chains, cache everything in session_state."""
    vs = initialize_vectorstore()
    st.session_state["vectorstore"] = vs
    st.session_state["chains"] = {
        "qa":      get_medical_qa_chain(vs),
        "symptom": get_symptom_chain(vs),
        "drug":    get_drug_chain(vs),
    }
    st.session_state["chains_loaded"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge base in-app setup
# ══════════════════════════════════════════════════════════════════════════════

def run_kb_setup() -> None:
    """
    Run the full knowledge-base pipeline inside Streamlit.
    Shows live progress via st.status.
    """
    from src.data_fetcher import MedicalDataFetcher
    from src.ingestion import chunk_documents
    from src.vectorstore import add_documents

    render_header("Knowledge Base Setup", "Building your local medical knowledge base...")

    with st.status("Building knowledge base — please keep this tab open...", expanded=True) as status:

        st.write("**Step 1/3** — Fetching from OpenFDA, PubMed & MedlinePlus in parallel (~30 s)...")
        fetcher = MedicalDataFetcher()
        all_docs = fetcher.fetch_all()
        st.write(f"  Fetched **{len(all_docs)}** documents.")

        st.write(f"**Step 2/3** — Chunking {len(all_docs)} documents...")
        chunks = chunk_documents(all_docs)
        st.write(f"  Created **{len(chunks)}** chunks.")

        st.write("**Step 3/3** — Embedding & saving FAISS index (~1 min)...")
        vs = initialize_vectorstore()
        add_documents(vs, chunks)

        status.update(label="Knowledge base ready!", state="complete", expanded=False)

    st.success(f"Done! {len(chunks)} chunks stored. Loading AI models...")
    st.session_state["setup_running"] = False
    _load_chains()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def has_emergency_symptoms(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list, key: str) -> None:
    """Render an expandable sources section below an assistant message."""
    if not sources:
        return
    real = [s for s in sources if s.metadata.get("type") != "init"]
    if not real:
        return

    # Deduplicate by source+name
    seen: set = set()
    unique: list = []
    for doc in real:
        dk = f"{doc.metadata.get('source')}/{doc.metadata.get('name')}"
        if dk not in seen:
            seen.add(dk)
            unique.append(doc)

    with st.expander(f"Sources — {len(unique)} referenced", expanded=False):
        for i, doc in enumerate(unique):
            src     = doc.metadata.get("source", "Unknown")
            name    = doc.metadata.get("name", "unknown").title()
            dtype   = doc.metadata.get("type", "")
            date    = doc.metadata.get("fetch_date", "")
            preview = doc.page_content[:220].replace("\n", " ").strip()

            score = doc.metadata.get("relevance_score")
            if score is not None:
                if score >= 70:
                    s_color, s_bg = "#166534", "#DCFCE7"
                elif score >= 45:
                    s_color, s_bg = "#713F12", "#FEF9C3"
                else:
                    s_color, s_bg = "#991B1B", "#FEF2F2"
                score_html = (
                    f'<span style="float:right;background:{s_bg};color:{s_color};'
                    f'padding:2px 8px;border-radius:20px;font-size:0.75rem;font-weight:600">'
                    f'{score:.0f}% match</span>'
                )
            else:
                score_html = ""

            st.markdown(
                f'<div class="source-card">'
                f'<strong style="color:#0F172A">{src}</strong>'
                f' &mdash; <span style="color:#334155">{name}</span>{score_html}'
                f'<br><small style="color:#64748B">type: {dtype} &bull; fetched: {date}</small>'
                f'<br><em style="font-size:0.85rem;color:#475569">{preview}...</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_response_meta(elapsed: float | None, n_sources: int) -> None:
    if elapsed is not None:
        real = max(0, n_sources - 1)
        st.markdown(
            f'<div class="response-meta">{elapsed:.1f}s &nbsp;&middot;&nbsp; {real} sources</div>',
            unsafe_allow_html=True,
        )


def render_error(error_msg: str, error_type: str | None = None) -> None:
    """Display a user-friendly error card based on error type."""
    labels = {
        "rate_limit":     "Rate Limit",
        "auth":           "Authentication Error",
        "timeout":        "Request Timeout",
        "network":        "Network Error",
        "context_length": "Input Too Long",
        "unknown":        "Error",
    }
    label = labels.get(error_type or "unknown", "Error")
    st.markdown(
        f'<div class="info-card red">'
        f'<strong>{label}</strong><br>'
        f'{error_msg.replace(chr(10), "<br>")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def validate_query(text: str, min_len: int = 5) -> str | None:
    """Return an error string if the query is invalid, else None."""
    stripped = text.strip()
    if not stripped:
        return "Please enter a question or description."
    if len(stripped) < min_len:
        return f"Your input is too short (minimum {min_len} characters)."
    if len(stripped) > 2000:
        return "Your input is too long (maximum 2000 characters). Please shorten it."
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Pre-auth pages
# ══════════════════════════════════════════════════════════════════════════════

_HIDE_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"]  { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
</style>
"""


def render_landing_page() -> None:
    st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

    # Hero
    _, hero_col, _ = st.columns([1, 2, 1])
    with hero_col:
        st.markdown(
            f'<div style="text-align:center;padding-top:32px">{_logo(140)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="hero-title">MedAssist</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-tagline">AI-Powered Healthcare Knowledge Assistant<br>'
            'Evidence-based answers from trusted medical sources.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    _, main_col, _ = st.columns([0.5, 4, 0.5])
    with main_col:

        # About card
        st.markdown(
            '<div class="about-card">'
            '<h3 style="margin:0 0 10px;font-size:1.1rem;color:#0F172A;font-weight:700">About MedAssist</h3>'
            '<p style="margin:0;font-size:0.9rem;color:#334155;line-height:1.7">'
            'MedAssist is an open-source Retrieval-Augmented Generation (RAG) system that '
            'retrieves relevant passages from a curated medical knowledge base and synthesises '
            'accurate, readable answers using a large language model. It covers general medical '
            'questions, symptom analysis, and drug interaction checks — all grounded in '
            'peer-reviewed and regulatory sources.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # How it works
        st.markdown(
            '<h3 style="color:#0F172A;font-weight:700;margin:0 0 12px">How It Works</h3>',
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        steps = [
            ("01", "Data Collection",
             "Medical documents fetched from OpenFDA, PubMed, and MedlinePlus APIs"),
            ("02", "Embedding",
             "Documents split into chunks and encoded as dense vectors with Sentence Transformers"),
            ("03", "Retrieval",
             "Your query is embedded and matched against the FAISS vector index"),
            ("04", "AI Generation",
             "Top chunks are sent to Groq LLaMA 3.3 70B to generate a grounded answer"),
        ]
        for col, (num, label, desc) in zip([p1, p2, p3, p4], steps):
            col.markdown(
                f'<div class="pipeline-step">'
                f'<div class="pipeline-num">{num}</div>'
                f'<div class="pipeline-label">{label}</div>'
                f'<div class="pipeline-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Data sources
        st.markdown(
            '<h3 style="color:#0F172A;font-weight:700;margin:0 0 12px">Data Sources</h3>',
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        data_sources = [
            ("OpenFDA",
             "Drug labels, interactions, dosing, adverse events, and contraindications "
             "from the FDA public drug database."),
            ("PubMed / NCBI",
             "Peer-reviewed research abstracts covering diseases, treatments, and clinical "
             "findings from the NCBI Entrez API."),
            ("MedlinePlus",
             "Consumer-friendly health topic summaries produced by the US National Library "
             "of Medicine."),
        ]
        for col, (title, desc) in zip([s1, s2, s3], data_sources):
            col.markdown(
                f'<div class="source-item"><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tech stack
        st.markdown(
            '<h3 style="color:#0F172A;font-weight:700;margin:0 0 12px">Technology Stack</h3>',
            unsafe_allow_html=True,
        )
        chips = [
            "Python 3.14", "LangChain", "FAISS",
            "Groq LLM", "Streamlit", "Sentence Transformers",
        ]
        st.markdown(
            "".join(f'<span class="tech-chip">{c}</span>' for c in chips),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown(
            '<div class="info-card yellow">'
            '<strong>Medical Disclaimer</strong> — MedAssist is for <em>educational purposes only</em>. '
            'It does not provide medical diagnosis, treatment recommendations, or professional medical '
            'advice. Always consult a qualified healthcare professional for personal health concerns.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Get Started button
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("Get Started", use_container_width=True, type="primary"):
                st.session_state["page"] = "login"
                st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)


def render_login_page() -> None:
    st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    _, card_col, _ = st.columns([1, 1.4, 1])
    with card_col:
        with st.container(border=True):
            st.markdown(
                f'<div style="text-align:center;padding:16px 0 8px">'
                f'{_logo(72)}'
                f'<h2 style="margin:12px 0 4px;font-size:1.4rem;color:#0F172A;font-weight:700">'
                f'Welcome to MedAssist</h2>'
                f'<p style="margin:0 0 20px;font-size:0.88rem;color:#64748B">'
                f'Sign in to access the healthcare assistant</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.get("login_error"):
                st.markdown(
                    '<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;'
                    'padding:10px 14px;margin-bottom:12px;color:#991B1B;font-size:0.88rem">'
                    'Incorrect username or password. Please try again.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            username = st.text_input("Username", key="login_u", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="login_p",
                                     placeholder="Enter password")

            if st.button("Sign In", use_container_width=True, type="primary"):
                if username == "user" and password == "user":
                    st.session_state["authenticated"] = True
                    st.session_state["page"] = "app"
                    st.session_state["login_error"] = False
                    st.rerun()
                else:
                    st.session_state["login_error"] = True
                    st.rerun()

            st.markdown(
                '<p style="text-align:center;font-size:0.78rem;color:#94A3B8;margin:16px 0 4px">'
                'Demo credentials: <code>user</code> / <code>user</code></p>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 18px'>"
            "<h2 style='font-size:1.05rem;font-weight:700;margin:0;letter-spacing:-0.01em'>"
            "MedAssist</h2>"
            "<p style='font-size:0.73rem;margin:3px 0 0;opacity:0.55'>AI Healthcare Assistant</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        options = ["Medical Q&A", "Symptom Checker", "Drug Interactions"]
        mode = st.radio(
            "mode",
            options,
            index=options.index(st.session_state.mode),
            label_visibility="collapsed",
        )
        st.session_state.mode = mode

        st.divider()

        # Knowledge base status
        st.markdown(
            "<p style='font-size:0.75rem;font-weight:500;margin:0 0 6px;opacity:0.6'>"
            "Knowledge Base</p>",
            unsafe_allow_html=True,
        )
        populated = is_populated()
        chains_ok = st.session_state.get("chains_loaded", False)

        if populated and chains_ok:
            st.markdown("<span class='status-badge populated'>Ready</span>",
                        unsafe_allow_html=True)
        elif populated and not chains_ok:
            st.markdown("<span class='status-badge loading'>Loading models...</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-badge empty'>Not built</span>",
                        unsafe_allow_html=True)
            st.caption("Click below to build the knowledge base (takes ~5-7 min).")
            if st.button("Build Knowledge Base", use_container_width=True):
                reset_vectorstore()
                st.session_state["chains_loaded"] = False
                st.session_state["setup_running"] = True
                st.rerun()

        if populated and chains_ok:
            st.divider()
            if st.button("Rebuild KB", use_container_width=True):
                reset_vectorstore()
                st.session_state["chains_loaded"] = False
                st.session_state["setup_running"] = True
                st.rerun()

        st.divider()
        st.markdown(
            "<div class='disclaimer'>Educational use only. Always consult a licensed "
            "healthcare professional.</div>",
            unsafe_allow_html=True,
        )

    return mode


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1 — Medical Q&A
# ══════════════════════════════════════════════════════════════════════════════

def render_medical_qa() -> None:
    render_header(
        "Medical Q&A",
        "Ask questions about diseases, conditions, treatments, and medications.",
    )

    chains_ok = st.session_state.get("chains_loaded", False)

    if not chains_ok:
        st.info("Build the knowledge base first using the sidebar button.")
        return

    chain = st.session_state["chains"]["qa"]

    # Sample questions
    with st.expander("Sample questions — click to ask"):
        cols = st.columns(2)
        samples = [
            "What is metformin used for?",
            "What are the side effects of ibuprofen?",
            "How is hypertension treated?",
            "What causes type 2 diabetes?",
            "Tell me about asthma medications",
            "What is the difference between ibuprofen and acetaminophen?",
        ]
        for i, q in enumerate(samples):
            if cols[i % 2].button(q, key=f"qa_sample_{i}"):
                _process_qa(chain, q)
                st.rerun()

    # Chat history
    for i, msg in enumerate(st.session_state.qa_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("is_error"):
                render_error(msg["content"], msg.get("error_type"))
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), key=f"qa_src_{i}")
                render_response_meta(msg.get("elapsed"), len(msg.get("sources", [])))

    # Clear button
    if st.session_state.qa_history:
        if st.button("Clear Chat", key="clear_qa"):
            st.session_state.qa_history = []
            st.rerun()

    # Input
    if prompt := st.chat_input("Ask a medical question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = _process_qa(chain, prompt)
            if result:
                if result.get("is_error"):
                    render_error(result["answer"], result.get("error_type"))
                else:
                    st.markdown(result["answer"])
                render_sources(result["sources"], key="qa_src_new")
                render_response_meta(result["elapsed"], len(result["sources"]))
        st.rerun()


def _process_qa(chain, query: str) -> dict | None:
    """Invoke QA chain, append to history, return result dict."""
    err = validate_query(query)
    if err:
        st.warning(err)
        return None

    t0 = time.time()
    r = invoke_chain(chain, query)
    elapsed = time.time() - t0

    if r["error"]:
        answer  = r["error"]
        sources = []
        is_err  = True
    else:
        answer  = r["answer"]
        sources = r["source_documents"]
        is_err  = False

    st.session_state.qa_history.append({"role": "user", "content": query})
    st.session_state.qa_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "elapsed": elapsed,
        "is_error": is_err,
        "error_type": r.get("error_type"),
    })
    return {"answer": answer, "sources": sources, "elapsed": elapsed,
            "is_error": is_err, "error_type": r.get("error_type")}


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Symptom Checker
# ══════════════════════════════════════════════════════════════════════════════

def render_symptom_checker() -> None:
    render_header(
        "Symptom Checker",
        "Describe your symptoms for a knowledge-based analysis with possible conditions.",
    )

    chains_ok = st.session_state.get("chains_loaded", False)

    if not chains_ok:
        st.info("Build the knowledge base first using the sidebar button.")
        return

    chain = st.session_state["chains"]["symptom"]

    st.markdown(
        "<div class='info-card yellow'>"
        "<strong>Not a diagnosis.</strong> This analysis is for educational purposes only "
        "and does not replace professional medical evaluation."
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick-select chips
    st.markdown(
        '<p style="font-weight:600;color:#1E293B;margin:0 0 8px">'
        'Quick symptom selector — click to append</p>',
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(5)
    common = [
        "Fever", "Headache", "Fatigue", "Cough", "Shortness of breath",
        "Chest pain", "Nausea", "Joint pain", "Dizziness", "Sore throat",
    ]
    for i, symptom in enumerate(common):
        if chip_cols[i % 5].button(symptom, key=f"chip_{i}"):
            existing = st.session_state.symptom_input_text
            sep = ", " if existing else ""
            st.session_state.symptom_input_text = existing + sep + symptom.lower()
            st.rerun()

    symptom_text = st.text_area(
        "Describe your symptoms in detail:",
        value=st.session_state.symptom_input_text,
        placeholder="e.g. I have had a persistent headache, fever of 101 F, and body aches for 3 days...",
        height=120,
        key="symptom_textarea",
    )

    analyse_clicked = st.button("Analyse Symptoms", type="primary", key="analyse_btn")

    # Chat history
    for i, msg in enumerate(st.session_state.symptom_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("is_error"):
                render_error(msg["content"], msg.get("error_type"))
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), key=f"sym_src_{i}")
                render_response_meta(msg.get("elapsed"), len(msg.get("sources", [])))

    if st.session_state.symptom_history:
        if st.button("Clear History", key="clear_symptom"):
            st.session_state.symptom_history = []
            st.rerun()

    if analyse_clicked:
        val_err = validate_query(symptom_text, min_len=10)
        if val_err:
            st.warning(val_err)
        else:
            # Emergency check — show banner before analysis
            if has_emergency_symptoms(symptom_text):
                st.markdown(
                    "<div class='emergency-banner'>"
                    "<h3>Seek Immediate Emergency Care</h3>"
                    "<p>Your symptoms may indicate a medical emergency. "
                    "<strong>Call 911 or go to the nearest emergency room immediately.</strong> "
                    "Do not wait for an online analysis.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with st.chat_message("user"):
                st.markdown(symptom_text)

            with st.chat_message("assistant"):
                with st.spinner("Analysing symptoms..."):
                    t0 = time.time()
                    r = invoke_chain(chain, symptom_text)
                    elapsed = time.time() - t0

                if r["error"]:
                    render_error(r["error"], r.get("error_type"))
                    answer, sources = r["error"], []
                else:
                    answer  = r["answer"]
                    sources = r["source_documents"]
                    st.markdown(answer)

                render_sources(sources, key="sym_src_new")
                render_response_meta(elapsed, len(sources))

            st.session_state.symptom_history.append({"role": "user", "content": symptom_text})
            st.session_state.symptom_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "elapsed": elapsed,
                "is_error": bool(r["error"]),
                "error_type": r.get("error_type"),
            })
            st.session_state.symptom_input_text = ""


# ══════════════════════════════════════════════════════════════════════════════
# Mode 3 — Drug Interaction Checker
# ══════════════════════════════════════════════════════════════════════════════

def render_drug_interactions() -> None:
    render_header(
        "Drug Interaction Checker",
        "Enter 2 or more drug names to check for interactions, side effects, and contraindications.",
    )

    chains_ok = st.session_state.get("chains_loaded", False)

    if not chains_ok:
        st.info("Build the knowledge base first using the sidebar button.")
        return

    chain = st.session_state["chains"]["drug"]

    st.markdown(
        "<div class='info-card red'>"
        "<strong>Always consult your pharmacist or physician</strong> "
        "before combining medications."
        "</div>",
        unsafe_allow_html=True,
    )

    # Drug input fields
    col1, col2 = st.columns(2)
    drug1 = col1.text_input("Drug 1", placeholder="e.g. ibuprofen", key="drug1")
    drug2 = col2.text_input("Drug 2", placeholder="e.g. aspirin",   key="drug2")
    drug3 = st.text_input("Drug 3 (optional)", placeholder="e.g. warfarin", key="drug3")

    check_clicked = st.button("Check Interactions", type="primary")

    # Sample queries
    with st.expander("Sample checks — click to run"):
        samples = [
            ("ibuprofen", "aspirin", ""),
            ("warfarin",  "aspirin", ""),
            ("metformin", "lisinopril", ""),
            ("sertraline", "tramadol", ""),
        ]
        sample_cols = st.columns(2)
        for i, (d1, d2, d3) in enumerate(samples):
            label = f"{d1} + {d2}" + (f" + {d3}" if d3 else "")
            if sample_cols[i % 2].button(label, key=f"drug_sample_{i}"):
                _process_drug_check(chain, d1, d2, d3)
                st.rerun()

    # Chat history
    for i, msg in enumerate(st.session_state.drug_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("is_error"):
                render_error(msg["content"], msg.get("error_type"))
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), key=f"drug_src_{i}")
                render_response_meta(msg.get("elapsed"), len(msg.get("sources", [])))

    if st.session_state.drug_history:
        if st.button("Clear History", key="clear_drug"):
            st.session_state.drug_history = []
            st.rerun()

    if check_clicked:
        drugs = [d.strip().lower() for d in [drug1, drug2, drug3] if d.strip()]
        if len(drugs) < 2:
            st.warning("Please enter at least 2 drug names.")
        elif any(len(d) < 2 for d in drugs):
            st.warning("Drug names must be at least 2 characters long.")
        else:
            _process_drug_check(chain, *drugs[:3])
            st.rerun()


def _process_drug_check(chain, drug1: str, drug2: str, drug3: str = "") -> None:
    """Invoke drug chain, show results inline, append to history."""
    drugs = [d for d in [drug1, drug2, drug3] if d]
    query = f"Check interactions between: {', '.join(drugs)}"

    user_content = "**Checking interactions for:** " + " + ".join(
        f"`{d}`" for d in drugs
    )

    with st.chat_message("user"):
        st.markdown(user_content)

    with st.chat_message("assistant"):
        with st.spinner(f"Checking interactions for {', '.join(drugs)}..."):
            t0 = time.time()
            r = invoke_chain(chain, query)
            elapsed = time.time() - t0

        if r["error"]:
            render_error(r["error"], r.get("error_type"))
            answer, sources = r["error"], []
        else:
            answer  = r["answer"]
            sources = r["source_documents"]
            _render_severity_badge(answer)
            st.markdown(answer)

        render_sources(sources, key=f"drug_src_new_{int(t0)}")
        render_response_meta(elapsed, len(sources))

    st.session_state.drug_history.append({"role": "user", "content": user_content})
    st.session_state.drug_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "elapsed": elapsed,
        "is_error": bool(r["error"]),
        "error_type": r.get("error_type"),
    })


def _render_severity_badge(answer_text: str) -> None:
    """Detect severity keywords in the LLM answer and render a badge."""
    t = answer_text.lower()
    if any(w in t for w in ["severe interaction", "serious interaction",
                             "contraindicated", "do not combine"]):
        st.markdown("<span class='badge-severe'>SEVERE INTERACTION</span>",
                    unsafe_allow_html=True)
    elif any(w in t for w in ["moderate interaction", "use caution",
                               "monitor closely", "increased risk"]):
        st.markdown("<span class='badge-moderate'>MODERATE — Use Caution</span>",
                    unsafe_allow_html=True)
    elif any(w in t for w in ["mild interaction", "minor interaction",
                               "generally safe", "low risk"]):
        st.markdown("<span class='badge-mild'>MILD — Generally Safe</span>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session_state()

    # ── Pre-auth routing ───────────────────────────────────────────────────────
    if not st.session_state.authenticated:
        if st.session_state.page == "login":
            render_login_page()
        else:
            render_landing_page()
        return

    # ── Run in-app KB setup if triggered ──────────────────────────────────────
    if st.session_state.setup_running:
        render_sidebar()
        run_kb_setup()
        return

    # ── Load chains once when KB is ready ─────────────────────────────────────
    if is_populated() and not st.session_state.chains_loaded:
        with st.spinner("Loading AI models..."):
            _load_chains()

    mode = render_sidebar()

    if mode == "Medical Q&A":
        render_medical_qa()
    elif mode == "Symptom Checker":
        render_symptom_checker()
    elif mode == "Drug Interactions":
        render_drug_interactions()


if __name__ == "__main__":
    main()
