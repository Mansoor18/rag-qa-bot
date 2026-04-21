from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
    model="llama-3.1-8b-instant",  # updated model
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""You are an expert assistant answering questions 
based strictly on the provided document context.
If the answer is not in the context, say "I don't have enough 
information in the document to answer this."

Context:
{context}

Question: {question}

Answer:""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask(question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    chain, retriever = build_qa_chain()
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    print(f"Answer: {answer}")
    print(f"\nSources: {len(sources)} chunks retrieved")
    return answer

if __name__ == "__main__":
    questions = [
        "What is the attention mechanism?",
        "What is multi-head attention?",
        "What optimizer was used to train the transformer?"
    ]
    for q in questions:
        ask(q)
        print("\n" + "="*60 + "\n")