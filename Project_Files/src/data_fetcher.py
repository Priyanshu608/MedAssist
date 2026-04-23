"""
MedicalDataFetcher — fetches medical data from free public APIs.

Sources:
  - OpenFDA  : drug labels, interactions, warnings (30 drugs)
  - PubMed   : research abstracts for 15 disease topics (up to 10 each)
  - MedlinePlus: consumer health topic summaries (15 topics)
"""
import re
import time
import logging
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Drug list (OpenFDA) ────────────────────────────────────────────────────────
DRUG_LIST = [
    "ibuprofen", "aspirin", "acetaminophen", "amoxicillin", "metformin",
    "lisinopril", "atorvastatin", "omeprazole", "losartan", "amlodipine",
    "metoprolol", "albuterol", "gabapentin", "hydrochlorothiazide", "sertraline",
    "fluoxetine", "prednisone", "tramadol", "furosemide", "pantoprazole",
    "escitalopram", "montelukast", "rosuvastatin", "levothyroxine", "clopidogrel",
    "warfarin", "ciprofloxacin", "azithromycin", "diazepam", "lorazepam",
]

# ── Disease topics (PubMed) ───────────────────────────────────────────────────
DISEASE_TOPICS = [
    "diabetes treatment", "hypertension management", "asthma therapy",
    "heart disease prevention", "depression treatment", "anxiety disorders",
    "pain management", "antibiotic resistance", "cancer screening",
    "vaccination guidelines", "cholesterol management", "kidney disease",
    "liver disease", "thyroid disorders", "allergic reactions",
]

# ── Health topics (MedlinePlus) ───────────────────────────────────────────────
HEALTH_TOPICS = [
    "diabetes", "hypertension", "asthma", "heart disease", "depression",
    "anxiety", "chronic pain", "cancer", "vaccination", "high cholesterol",
    "kidney disease", "thyroid", "allergy", "arthritis", "obesity",
]

# ── API endpoints ─────────────────────────────────────────────────────────────
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"


class MedicalDataFetcher:
    """
    Fetches medical data from public APIs and returns LangChain Document objects.

    Each document has:
      page_content : str  — the text content
      metadata     : dict — source, type, name, fetch_date
    """

    def __init__(self, delay: float = 0.3, timeout: int = 30):
        self.delay = delay
        self.timeout = timeout
        self.fetch_date = date.today().isoformat()

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_drug_data(self) -> list[Document]:
        """Fetch drug label data from OpenFDA for all 30 drugs."""
        docs: list[Document] = []
        print(f"\n  Fetching {len(DRUG_LIST)} drugs from OpenFDA...")
        for drug in DRUG_LIST:
            print(f"    {drug}…", end=" ", flush=True)
            doc = self._fetch_openfda_drug(drug)
            if doc:
                docs.append(doc)
                print("OK")
            else:
                print("-- (not found)")
        return docs

    def fetch_disease_data(self) -> list[Document]:
        """Fetch PubMed abstracts for all 15 disease topics (up to 10 each)."""
        docs: list[Document] = []
        print(f"\n  Fetching {len(DISEASE_TOPICS)} topics from PubMed...")
        for topic in DISEASE_TOPICS:
            print(f"    '{topic}'…", end=" ", flush=True)
            topic_docs = self._fetch_pubmed_topic(topic)
            docs.extend(topic_docs)
            print(f"OK ({len(topic_docs)} articles)")
        return docs

    def fetch_health_topics(self) -> list[Document]:
        """Fetch consumer health summaries from MedlinePlus."""
        docs: list[Document] = []
        print(f"\n  Fetching {len(HEALTH_TOPICS)} topics from MedlinePlus...")
        for topic in HEALTH_TOPICS:
            print(f"    '{topic}'…", end=" ", flush=True)
            topic_docs = self._fetch_medlineplus(topic)
            docs.extend(topic_docs)
            print(f"OK ({len(topic_docs)} summaries)")
        return docs

    def fetch_all(self) -> list[Document]:
        """
        Run all 3 fetchers concurrently (ThreadPoolExecutor) and return
        the combined list of Documents.

        Each source enforces its own per-call delay independently, so
        rate limits are respected per API. Total wall-clock time is
        dominated by the slowest source (~OpenFDA at 30 drugs × 0.3 s).
        """
        from concurrent.futures import ThreadPoolExecutor

        print("-" * 55)
        print("  MedAssist - Fetching Medical Knowledge Base")
        print("  (OpenFDA, PubMed, MedlinePlus running in parallel)")
        print("-" * 55)

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_drug    = executor.submit(self.fetch_drug_data)
            fut_disease = executor.submit(self.fetch_disease_data)
            fut_topics  = executor.submit(self.fetch_health_topics)
        # All 3 futures are complete when the context exits (shutdown wait=True)
        drug_docs    = fut_drug.result()
        disease_docs = fut_disease.result()
        topic_docs   = fut_topics.result()

        all_docs = drug_docs + disease_docs + topic_docs
        print(f"\n{'-' * 55}")
        print(f"  Drug: {len(drug_docs)}  Disease: {len(disease_docs)}  Topics: {len(topic_docs)}")
        print(f"  Total fetched: {len(all_docs)} documents")
        print("-" * 55)
        return all_docs

    # ═══════════════════════════════════════════════════════════════════════════
    # OpenFDA helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _fetch_openfda_drug(self, drug_name: str) -> Optional[Document]:
        """Try generic_name then brand_name search on OpenFDA."""
        search_fields = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
            # Fallback: free-text in indications field
            f'indications_and_usage:"{drug_name}"',
        ]
        for search in search_fields:
            data = self._get_json(OPENFDA_URL, {"search": search, "limit": 1})
            if data and data.get("results"):
                content = self._build_drug_content(drug_name, data["results"][0])
                if content:
                    return Document(
                        page_content=content,
                        metadata={
                            "source": "OpenFDA",
                            "type": "drug",
                            "name": drug_name,
                            "fetch_date": self.fetch_date,
                        },
                    )
        return None

    @staticmethod
    def _build_drug_content(drug_name: str, result: dict) -> str:
        """Assemble a readable text block from an OpenFDA label result."""
        openfda = result.get("openfda", {})
        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])

        lines = [f"DRUG: {drug_name.upper()}"]
        if brand_names:
            lines.append(f"Brand Name(s): {', '.join(brand_names[:3])}")
        if generic_names:
            lines.append(f"Generic Name(s): {', '.join(generic_names[:3])}")

        label_fields = [
            ("Description", "description"),
            ("Indications and Usage", "indications_and_usage"),
            ("Contraindications", "contraindications"),
            ("Warnings", "warnings"),
            ("Drug Interactions", "drug_interactions"),
            ("Adverse Reactions", "adverse_reactions"),
            ("Dosage and Administration", "dosage_and_administration"),
        ]
        for heading, key in label_fields:
            values = result.get(key, [])
            if values:
                # Join and truncate to keep documents reasonable sized
                text = " ".join(values)[:3000].strip()
                # Strip HTML/XML tags if present
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s{2,}", " ", text)
                lines.append(f"\n{heading}:\n{text}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # PubMed helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _fetch_pubmed_topic(self, topic: str) -> list[Document]:
        """Search PubMed and return up to 10 abstract Documents for a topic."""
        # Step 1 — get PMIDs
        search_data = self._get_json(PUBMED_SEARCH_URL, {
            "db": "pubmed",
            "term": topic,
            "retmax": 5,
            "retmode": "json",
        })
        if not search_data:
            return []

        pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        # Step 2 — fetch all abstracts in one call (plain text)
        abstract_text = self._get_text(PUBMED_FETCH_URL, {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
        })
        if not abstract_text:
            return []

        return self._parse_pubmed_abstracts(abstract_text, topic, pmids)

    @staticmethod
    def _parse_pubmed_abstracts(
        text: str, topic: str, pmids: list[str]
    ) -> list[Document]:
        """
        Split the plain-text efetch response into individual article Documents.
        Articles are separated by blank lines + an ordinal number line.
        """
        # Split on lines that are purely a digit sequence (article separator)
        articles = re.split(r"\n{2,}(?=\d+\. )", text.strip())
        docs: list[Document] = []
        for i, article in enumerate(articles):
            article = article.strip()
            if len(article) < 80:   # skip trivially short fragments
                continue
            docs.append(Document(
                page_content=f"Research topic: {topic}\n\n{article}",
                metadata={
                    "source": "PubMed",
                    "type": "disease",
                    "name": topic,
                    "fetch_date": date.today().isoformat(),
                    "pmid": pmids[i] if i < len(pmids) else "unknown",
                },
            ))
        return docs

    # ═══════════════════════════════════════════════════════════════════════════
    # MedlinePlus helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _fetch_medlineplus(self, topic: str) -> list[Document]:
        """Fetch and parse health-topic summaries from MedlinePlus XML API."""
        xml_text = self._get_text(MEDLINEPLUS_URL, {
            "db": "healthTopics",
            "term": topic,
            "retmax": 3,
        })
        if not xml_text:
            return []

        docs: list[Document] = []
        try:
            soup = BeautifulSoup(xml_text, "lxml-xml")
            for doc_elem in soup.find_all("document"):
                title = self._content_text(doc_elem, "title") or topic
                summary = (
                    self._content_text(doc_elem, "FullSummary")
                    or self._content_text(doc_elem, "snippet")
                    or ""
                )
                # Strip residual HTML tags from summary
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s{2,}", " ", summary).strip()

                if len(summary) < 50:
                    continue

                docs.append(Document(
                    page_content=f"Health Topic: {title}\n\n{summary}",
                    metadata={
                        "source": "MedlinePlus",
                        "type": "health_topic",
                        "name": topic,
                        "fetch_date": self.fetch_date,
                    },
                ))
        except Exception as exc:
            logger.warning("MedlinePlus parse error for '%s': %s", topic, exc)

        return docs

    @staticmethod
    def _content_text(doc_elem, name: str) -> str:
        """Extract text from a <content name='...'> element."""
        elem = doc_elem.find("content", attrs={"name": name})
        return elem.get_text(separator=" ", strip=True) if elem else ""

    # ═══════════════════════════════════════════════════════════════════════════
    # HTTP helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET → JSON with rate limiting and error handling."""
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            time.sleep(self.delay)
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("GET JSON failed: %s — %s", url, exc)
            time.sleep(self.delay)
            return None

    def _get_text(self, url: str, params: Optional[dict] = None) -> Optional[str]:
        """GET → plain text with rate limiting and error handling."""
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            time.sleep(self.delay)
            return resp.text
        except requests.RequestException as exc:
            logger.warning("GET text failed: %s — %s", url, exc)
            time.sleep(self.delay)
            return None
