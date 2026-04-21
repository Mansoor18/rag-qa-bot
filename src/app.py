from fastapi import FastAPI
from pydantic import BaseModel
from generate import build_qa_chain, load_vectorstore

app = FastAPI(title="RAG Q&A Bot", version="1.0")

# Load once at startup
vectorstore = load_vectorstore()
chain, retriever = build_qa_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_retrieved: int

@app.get("/")
def root():
    return {"status": "RAG Q&A Bot is running"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    answer = chain.invoke(request.question)
    sources = retriever.invoke(request.question)
    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_retrieved=len(sources)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)