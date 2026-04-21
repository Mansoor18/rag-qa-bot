from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "models/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_chunks(query, k=4):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    print(f"\nTop {k} chunks retrieved for: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:200])
        print()
    return results

if __name__ == "__main__":
    retrieve_chunks("What is the attention mechanism?")