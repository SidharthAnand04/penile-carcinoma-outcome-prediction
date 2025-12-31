# Penile SCC Agentic RAG Learning Tool

A fast MVP educational tool for learning about penile squamous cell carcinoma using RAG (Retrieval-Augmented Generation).

## Features

- 💬 Chat interface to ask questions about penile cancer
- 🎯 Two audience modes: Family/Patient or Clinician
- ⚡ Two depth modes: Quick answers or in-depth explanations
- 📚 Retrieval-augmented generation from 10 trusted sources
- 📖 Citations and source tracking for all answers
- 🛡️ Safety guardrails to prevent harmful advice
- 🔒 Local vector DB (ChromaDB) - no data upload required

## Quick Start

### 1. Install Dependencies

From the `rag` folder:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy `.env.example` to `.env` and add your OpenRouter API key:

```bash
cp .env.example .env
```

Then edit `.env`:
```
OPENROUTER_API_KEY=sk_...your_key...
OPENROUTER_MODEL=openrouter/auto  # or specific model like: gpt-3.5-turbo
```

**Get your key:** https://openrouter.ai/keys

### 3. Ingest Sources (One-time setup)

```bash
python -m src.ingest
```

This will:
- Fetch the 10 trusted medical sources
- Extract and chunk the text
- Generate embeddings
- Store in local ChromaDB at `data/chroma/`
- Write results to `reports/ingest_log.jsonl`

**Verify it worked:** Check `reports/ingest_log.jsonl` for status of each URL. Most should show "OK".

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select your profile:**
   - Patient/Family: Simple language, avoided jargon
   - Clinician: Clinical terminology

2. **Choose depth:**
   - Quick: 1-2 sentence answer
   - In-depth: Full explanation with context

3. **Ask a question**, e.g.:
   - "What are the symptoms of penile cancer?"
   - "How is penile cancer staged?"
   - "What are treatment options?"

4. **Read the answer** with citations `[1]`, `[2]`, etc.

5. **View sources** by expanding the "Retrieved Sources" box

## Project Structure

```
rag/
├── app.py                 # Streamlit UI
├── requirements.txt       # Python dependencies
├── .env.example          # Template for environment variables
├── .env                  # (Create this) Your OpenRouter API key
├── src/
│   ├── __init__.py
│   ├── ingest.py        # Fetch URLs, chunk, embed, store in ChromaDB
│   ├── rag.py           # RAG logic + OpenRouter integration
│   ├── split.py         # Text chunking with overlap
│   └── safety.py        # Guardrails for unsafe queries
├── data/
│   ├── sources.txt      # 10 URLs to ingest
│   └── chroma/          # ChromaDB storage (created after ingest)
└── reports/
    └── ingest_log.jsonl # Ingestion results log
```

## How It Works

### Ingestion (`src/ingest.py`)

1. Reads 10 URLs from `data/sources.txt`
2. Fetches HTML using `requests` with fallback extraction
3. Cleans and chunks text (512 chars, 100 char overlap)
4. Generates embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
5. Stores chunks + metadata in local ChromaDB
6. Logs results to `reports/ingest_log.jsonl`

### Retrieval & Answering (`src/rag.py`)

1. User asks a question
2. Question is embedded with same model
3. Top-7 relevant chunks retrieved from ChromaDB
4. Chunks are sent to OpenRouter LLM with system prompt
5. LLM generates grounded answer with citations
6. Answer is displayed with source tracking

### Safety (`src/safety.py`)

Blocks queries asking for:
- Personal treatment decisions
- Drug dosing
- Interpretation of personal scans/labs
- Other unsafe advice

## Environment Variables

**Required:**
- `OPENROUTER_API_KEY` - Your OpenRouter API key

**Optional:**
- `OPENROUTER_MODEL` - Model to use (default: `openrouter/auto`)
- `OPENROUTER_SITE_URL` - Your site URL (used as HTTP-Referer)
- `OPENROUTER_APP_NAME` - Your app name (used as X-Title)

## Troubleshooting

### "ChromaDB not found" error
Run ingestion first: `python -m src.ingest`

### "OPENROUTER_API_KEY not set" error
Copy `.env.example` to `.env` and add your key

### Some URLs fail during ingestion
Normal. The tool continues if 1-3 fail. Check `reports/ingest_log.jsonl` for details.

### Slow ingestion
First run downloads embedding model (~60MB). Subsequent runs are fast.

## Safety Disclaimer

⚠️ **This tool is educational only.** It does NOT provide medical advice. Always consult healthcare professionals for personal treatment decisions.

## Sources

The tool retrieves information from:
1. cancer.gov (NCI) - Patient & clinician PDQ summaries
2. cancer.org (ACS) - Detection, diagnosis, staging, symptoms
3. uroweb.org (EAU) - Clinical guidelines & disease management
4. cancerresearchuk.org - TNM staging
5. oncolink.org - Staging and treatment overview
6. pmc.ncbi.nlm.nih.gov - Published research articles

## Notes

- Chunking uses character-based splitting with overlap to preserve context
- Embeddings are generated locally using `sentence-transformers` (no API calls)
- ChromaDB uses cosine similarity for retrieval
- LLM responses are temperature-controlled (0.5) for consistency
- All answers must cite retrieved sources
