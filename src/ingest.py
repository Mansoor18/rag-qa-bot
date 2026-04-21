from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_document(path):
    print(f"Loading document: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("models", exist_ok=True)
    vectorstore.save_local("models/faiss_index")
    print("✅ FAISS index saved to models/faiss_index")
    return vectorstore

if __name__ == "__main__":
    pages = load_document("data/attention.pdf")
    chunks = split_documents(pages)
    embed_and_store(chunks)