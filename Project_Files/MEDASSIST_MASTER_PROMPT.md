# MedAssist — AI-Powered Healthcare Knowledge Assistant

## Master Prompt for Claude Code

> **IMPORTANT**: Follow this document phase-by-phase. Complete each phase, run it, verify it works, then move to the next. Do NOT build everything at once.

---

## PROJECT OVERVIEW

MedAssist is a RAG (Retrieval-Augmented Generation) based healthcare assistant that **automatically fetches** medical data from free public APIs, stores it in a vector database, and provides 3 modes of interaction:

1. **Medical Document Q&A** — Ask questions about diseases, conditions, treatments. Answers are cited with sources.
2. **Symptom Checker** — User describes symptoms → system retrieves matching conditions from knowledge base → LLM provides analysis with disclaimers.
3. **Drug Interaction Checker** — User enters 2+ drug names → system checks for interactions, side effects, contraindications.

---

## TECH STACK

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Groq API (`llama-3.3-70b-versatile`) | Free tier, blazing fast |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, lightweight (~80MB) |
| Vector Store | ChromaDB (persistent, local) | Simple, no infra needed |
| RAG Framework | LangChain | Industry standard |
| Frontend | Streamlit | Fast to build, professional look |
| Language | Python 3.10+ | |

---

## ENVIRONMENT SETUP

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install streamlit langchain langchain-community langchain-groq chromadb sentence-transformers requests beautifulsoup4 lxml
```

### Environment Variable
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Also create a `.gitignore`:
```
venv/
chroma_db/
.env
__pycache__/
*.pyc
```

---

## PROJECT STRUCTURE

```
medassist/
├── app.py                          # Streamlit main application
├── config.py                       # Configuration and environment variables
├── requirements.txt                # Python dependencies
├── .env                            # API keys (gitignored)
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py             # Fetches data from public medical APIs
│   ├── ingestion.py                # Chunks and embeds fetched data
│   ├── vectorstore.py              # ChromaDB setup and management
│   ├── chains.py                   # RAG chains for each mode
│   └── prompts.py                  # Prompt templates for each mode
├── scripts/
│   └── setup_knowledge_base.py     # One-command KB builder
└── chroma_db/                      # Auto-generated persistent vector store
```

---

## PHASE 1: PROJECT SKELETON + BASIC STREAMLIT UI

### Goal: Get a running Streamlit app with 3 tabs and sidebar navigation.

Create ALL the files in the project structure. The Streamlit app should have:

- A **sidebar** with:
  - App logo/title: "🏥 MedAssist" with subtitle "AI-Powered Healthcare Assistant"
  - Navigation radio buttons for 3 modes: "📄 Medical Q&A", "🩺 Symptom Checker", "💊 Drug Interactions"
  - A medical disclaimer at the bottom: "⚠️ This tool is for educational purposes only. Always consult a healthcare professional."
  - A "Knowledge Base Status" indicator showing if DB is populated

- **Main area** based on selected mode:
  - Each mode has a chat-like interface with `st.chat_message`
  - Each mode has its own `st.session_state` chat history
  - Input area at the bottom

- **Professional styling** with custom CSS:
  - Clean medical theme (blues/whites/greens)
  - Custom font styling
  - Rounded cards for information display
  - Responsive layout

### Verification: Run `streamlit run app.py` — all 3 tabs should render with placeholder content.

---

## PHASE 2: DATA FETCHER — AUTOMATIC MEDICAL DATA COLLECTION

### Goal: Build scripts that fetch medical data from free public APIs. No manual data needed.

### API Sources to Use:

#### 1. OpenFDA API (Drug Information) — NO API KEY NEEDED
```
Base URL: https://api.fda.gov/drug/label.json

# Fetch drug labels for common medications
GET https://api.fda.gov/drug/label.json?search=openfda.brand_name:"ibuprofen"&limit=1
GET https://api.fda.gov/drug/label.json?search=openfda.brand_name:"aspirin"&limit=1

# Fields to extract:
# - openfda.brand_name
# - openfda.generic_name
# - description
# - indications_and_usage
# - warnings
# - drug_interactions
# - adverse_reactions
# - dosage_and_administration
# - contraindications
```

**Fetch data for these 30+ common drugs:**
ibuprofen, aspirin, acetaminophen, amoxicillin, metformin, lisinopril, atorvastatin, omeprazole, losartan, amlodipine, metoprolol, albuterol, gabapentin, hydrochlorothiazide, sertraline, fluoxetine, prednisone, tramadol, furosemide, pantoprazole, escitalopram, montelukast, rosuvastatin, levothyroxine, clopidogrel, warfarin, ciprofloxacin, azithromycin, diazepam, lorazepam

#### 2. Disease.sh API (Disease Statistics) — NO API KEY NEEDED
```
# COVID data (as example disease data)
GET https://disease.sh/v3/covid-19/all

# Historical data
GET https://disease.sh/v3/covid-19/historical/all?lastdays=30
```

#### 3. WHO GHO API (Global Health Data) — NO API KEY NEEDED
```
GET https://ghoapi.azureedge.net/api/indicators
# Fetch indicators for major diseases
```

#### 4. PubMed/NCBI E-Utilities (Medical Research Abstracts) — FREE
```
Base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

# Search for articles
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=diabetes+treatment&retmax=10&retmode=json

# Fetch abstracts
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=PMID1,PMID2&rettype=abstract&retmode=text
```

**Fetch abstracts for these medical topics (10 articles each):**
diabetes treatment, hypertension management, asthma therapy, heart disease prevention, depression treatment, anxiety disorders, pain management, antibiotic resistance, cancer screening, vaccination guidelines, cholesterol management, kidney disease, liver disease, thyroid disorders, allergic reactions

#### 5. MedlinePlus Connect API (Health Topic Summaries) — FREE
```
# Health topics
GET https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=diabetes&retmax=5
```

### Data Fetcher Implementation (`src/data_fetcher.py`):

- Create a class `MedicalDataFetcher` with methods:
  - `fetch_drug_data()` → fetches from OpenFDA for all 30 drugs
  - `fetch_disease_data()` → fetches from PubMed abstracts for all 15 topics
  - `fetch_health_topics()` → fetches from MedlinePlus
  - `fetch_all()` → runs all fetchers, returns list of Document objects

- Each document should have:
  - `page_content`: The actual text content
  - `metadata`: `{"source": "OpenFDA/PubMed/MedlinePlus", "type": "drug/disease/symptom", "name": "ibuprofen/diabetes/etc", "fetch_date": "2025-..."}`

- Add proper error handling: if one API call fails, log it and continue with the rest
- Add a progress bar or print statements showing fetch progress
- Add rate limiting (1 second delay between API calls) to be respectful to free APIs

### Verification: Run `python -c "from src.data_fetcher import MedicalDataFetcher; f = MedicalDataFetcher(); docs = f.fetch_all(); print(f'Fetched {len(docs)} documents')"` — should show 100+ documents fetched.

---

## PHASE 3: INGESTION + VECTOR STORE

### Goal: Chunk fetched documents and store embeddings in ChromaDB.

### Ingestion Pipeline (`src/ingestion.py`):
- Use `RecursiveCharacterTextSplitter` from LangChain
  - `chunk_size=500`
  - `chunk_overlap=50`
  - `separators=["\n\n", "\n", ". ", " ", ""]`
- Preserve metadata through chunking
- Add `chunk_index` to metadata for ordering

### Vector Store (`src/vectorstore.py`):
- Use ChromaDB with persistent storage at `./chroma_db`
- Use `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2`
- Create a single collection called `medical_knowledge`
- Implement methods:
  - `initialize_vectorstore()` → creates/loads the persistent DB
  - `add_documents(docs)` → adds chunked documents
  - `similarity_search(query, k=5, filter=None)` → retrieves relevant chunks
  - `get_retriever(search_type="similarity", k=5)` → returns a LangChain retriever
  - `is_populated()` → returns True if DB has documents

### Setup Script (`scripts/setup_knowledge_base.py`):
- Fetches all data → chunks → embeds → stores
- Shows progress with print statements
- Handles the case where DB is already populated (ask to rebuild or skip)
- Total runtime should be ~5-10 minutes

### Verification: Run `python scripts/setup_knowledge_base.py` then test retrieval:
```python
from src.vectorstore import get_retriever
retriever = get_retriever()
results = retriever.invoke("What are the side effects of ibuprofen?")
print(results)
```

---

## PHASE 4: RAG CHAINS + PROMPT TEMPLATES

### Goal: Create 3 specialized RAG chains, one for each mode.

### Prompt Templates (`src/prompts.py`):

#### Medical Q&A Prompt:
```
You are MedAssist, a knowledgeable medical information assistant. Answer the user's medical question using ONLY the provided context from verified medical sources.

Rules:
- Base your answer strictly on the provided context
- If the context doesn't contain enough information, say so clearly
- Always cite which source the information came from
- Include relevant warnings or contraindications when applicable
- End with a reminder to consult a healthcare professional
- Use clear, patient-friendly language

Context from medical knowledge base:
{context}

User Question: {question}

Provide a comprehensive, well-structured answer:
```

#### Symptom Checker Prompt:
```
You are MedAssist's Symptom Analysis module. Based on the user's reported symptoms and the medical knowledge base context, provide a helpful analysis.

Rules:
- List possible conditions that match the described symptoms, ranked by likelihood based on the context
- For each condition, briefly explain why the symptoms match
- Clearly state this is NOT a diagnosis
- Recommend specific types of medical professionals to consult
- Flag any symptoms that require immediate emergency attention (chest pain, difficulty breathing, severe bleeding, etc.)
- Use empathetic, reassuring language

Context from medical knowledge base:
{context}

Patient's Reported Symptoms: {question}

Provide your symptom analysis:
```

#### Drug Interaction Prompt:
```
You are MedAssist's Drug Interaction Checker. Analyze potential interactions between the drugs mentioned by the user using the provided medical context.

Rules:
- Identify each drug mentioned and its primary use
- Check for known interactions between the drugs based on the context
- Rate interaction severity if possible (mild/moderate/severe)
- List important warnings and contraindications for each drug
- Mention common side effects
- Recommend consulting a pharmacist or physician
- If interaction data is not found in the context, clearly state this

Context from medical knowledge base:
{context}

User's Drug Query: {question}

Provide your drug interaction analysis:
```

### RAG Chains (`src/chains.py`):
- Use `ChatGroq` with model `llama-3.3-70b-versatile`
- Temperature: 0.2 (low for medical accuracy)
- Create 3 chain functions:
  - `get_medical_qa_chain(retriever)` — uses QA prompt, retrieves from all document types
  - `get_symptom_chain(retriever)` — uses symptom prompt, filters retrieval to type="disease"
  - `get_drug_chain(retriever)` — uses drug prompt, filters retrieval to type="drug"
- Each chain should return BOTH the answer AND the source documents (for citation display)
- Use `RetrievalQA` or a custom chain with `RunnablePassthrough`
- Add error handling for Groq API failures with retry logic

### Verification: Test each chain individually:
```python
result = medical_qa_chain.invoke("What is metformin used for?")
print(result)
```

---

## PHASE 5: CONNECT EVERYTHING IN STREAMLIT UI

### Goal: Wire all components together into the full working app.

### App Flow (`app.py`):

1. On startup, check if ChromaDB is populated → show status in sidebar
2. If not populated, show a "🔄 Setup Knowledge Base" button that runs the setup script
3. Once populated, enable all 3 modes

### Mode 1: Medical Q&A
- Chat interface with `st.chat_message`
- User types a medical question
- System retrieves relevant documents, generates answer
- Display answer with expandable "📚 Sources" section showing which documents were used
- Maintain chat history in `st.session_state`

### Mode 2: Symptom Checker
- Text area for describing symptoms (not just a text input — give room for detail)
- Optional: Quick-select common symptoms as chips/buttons (fever, headache, fatigue, cough, etc.)
- Display analysis with color-coded severity indicators
- Show "🚨 Seek Immediate Care" banner if critical symptoms detected
- Maintain chat history

### Mode 3: Drug Interaction Checker
- Input fields for entering drug names (at least 2)
- "Check Interactions" button
- Display results in a structured format:
  - Drug summaries in cards
  - Interaction matrix/table
  - Severity badges (green/yellow/red)
- Maintain chat history

### UI Polish:
- Loading spinner while RAG chain processes
- Error handling with user-friendly messages
- Responsive layout
- Custom CSS for medical theme:
  ```css
  Main colors:
  - Primary: #1a73e8 (medical blue)
  - Secondary: #34a853 (health green)
  - Warning: #fbbc04 (caution yellow)
  - Danger: #ea4335 (alert red)
  - Background: #f8f9fa (light gray)
  - Card Background: #ffffff
  ```
- Confidence score based on retrieval similarity score (displayed as a small badge)

### Verification: Run `streamlit run app.py` and test all 3 modes with real queries.

---

## PHASE 6: ERROR HANDLING, POLISH, AND FINAL TESTING

### Error Handling:
- Groq API rate limit (429) → exponential backoff with max 3 retries
- API fetch failures → graceful degradation, log errors
- Empty retrieval results → "I don't have enough information" message
- ChromaDB connection issues → clear error message with fix instructions
- Network timeouts → retry with timeout of 30 seconds

### Final Polish:
- Add a "Sample Questions" section for each mode with clickable examples
- Add an "About" page in the sidebar explaining the tech stack and data sources
- Add response time display (how long the query took)
- Add a "Clear Chat" button for each mode

### Testing Queries:

**Medical Q&A:**
- "What is metformin used for?"
- "What are the side effects of ibuprofen?"
- "How is hypertension treated?"
- "What causes diabetes?"
- "Tell me about asthma medications"

**Symptom Checker:**
- "I have a persistent headache, fever of 101°F, and body aches for 3 days"
- "I'm experiencing chest tightness and shortness of breath"
- "I have joint pain, fatigue, and swelling in my hands"
- "I've been feeling dizzy and nauseous for a week"

**Drug Interaction Checker:**
- "Can I take ibuprofen and aspirin together?"
- "I'm on warfarin, is it safe to take aspirin?"
- "Check interactions between metformin and lisinopril"
- "I take sertraline and tramadol, any concerns?"

---

## REQUIREMENTS.TXT

```
streamlit>=1.31.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-groq>=0.0.1
chromadb>=0.4.22
sentence-transformers>=2.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
python-dotenv>=1.0.0
```

---

## KEY IMPLEMENTATION NOTES

1. **Groq Model**: Use `llama-3.3-70b-versatile` — it's the best balance of quality and free tier limits
2. **Embedding Model**: `all-MiniLM-L6-v2` downloads automatically on first run (~80MB)
3. **ChromaDB**: Use persistent storage so you don't re-embed every time you restart
4. **Rate Limiting**: Add 1-second delays between API calls in the data fetcher
5. **Error Messages**: Always user-friendly, never show raw tracebacks in the UI
6. **Medical Disclaimers**: Show prominently — this is for educational purposes only
7. **Windows Compatibility**: Make sure all file paths use `os.path.join()` or `pathlib.Path`
8. **Session State**: Use `st.session_state` for chat history, NOT global variables

---

## README.md TEMPLATE

Create a professional README with:
- Project title and description
- Screenshots placeholder
- Features list
- Tech stack
- Installation instructions (step by step)
- Usage guide
- API sources credited
- Disclaimer
- License (MIT)
