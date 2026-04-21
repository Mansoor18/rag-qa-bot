# RAG-based Document Q&A Bot

Retrieval-Augmented Generation system for intelligent document querying.
Built on the "Attention Is All You Need" transformer paper as the knowledge base.

## Architecture
User Query → FAISS Retrieval → Context + Query → LLaMA 3.1 → Answer

## Tech Stack
- LangChain LCEL for orchestration
- FAISS for vector similarity search
- HuggingFace sentence-transformers for embeddings
- Groq API (LLaMA 3.1) as LLM backend
- FastAPI for serving

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

```bash
# Ingest document
python src/ingest.py

# Run API
python src/app.py
```

## API Usage
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

## Sample Response
```json
{
  "question": "What is multi-head attention?",
  "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
  "chunks_retrieved": 4
}
```