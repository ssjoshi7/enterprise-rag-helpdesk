# Enterprise RAG Helpdesk Agent

An enterprise IT helpdesk assistant built using Retrieval-Augmented Generation (RAG).

## Tech Stack
- **Python** — Core language
- **ChromaDB** — Vector database for semantic search
- **all-MiniLM-L6-v2** — Embedding model
- **Claude API (Anthropic)** — Grounded answer generation
- **Streamlit** — Chat UI

## Architecture
User Query → Embedding → ChromaDB Semantic Search → Retrieved Chunks → Claude Grounded Answer

## Features
- Semantic knowledge base retrieval
- Hallucination prevention — answers only from KB
- Intelligent escalation for unknown queries
- Professional Streamlit chat interface

## Run Locally
```bash
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Built by
Swapnil Joshi — ssjoshi7.github.io