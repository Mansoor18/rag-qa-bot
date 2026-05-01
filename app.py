import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import streamlit as st
import tempfile
import os

# Load API key from Streamlit secrets or .env
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#load_dotenv()

# ── Page config ──
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# ── Header ──
st.title("📚 RAG Document Q&A Bot")
st.markdown("*Upload any PDF and ask questions — powered by LLaMA 3.1 + FAISS*")
st.divider()

# ── Session state ──
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# ── Embeddings loader ──
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── Build RAG chain ──
def build_chain(retriever):
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

    prompt = PromptTemplate.from_template("""You are an expert assistant 
answering questions based strictly on the provided document context.
If the answer is not in the context, say: 
"I don't have enough information in the document to answer this."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Sidebar ──
st.sidebar.header("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.sidebar.button("🔄 Process Document", use_container_width=True):
        with st.spinner("Processing document..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name

            # Load and chunk
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)

            # Embed and store
            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Build chain
            chain = build_chain(retriever)

            # Save to session
            st.session_state.qa_chain = chain
            st.session_state.retriever = retriever
            st.session_state.chat_history = []
            st.session_state.doc_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)

            os.unlink(tmp_path)

        st.sidebar.success(f"✅ Processed {len(pages)} pages → {len(chunks)} chunks")

# ── Sidebar stats ──
if st.session_state.doc_name:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Document Stats")
    st.sidebar.markdown(f"**File:** {st.session_state.doc_name}")
    st.sidebar.markdown(f"**Chunks:** {st.session_state.num_chunks}")
    st.sidebar.markdown(f"**Retrieval:** Top 4 chunks per query")
    st.sidebar.markdown(f"**Model:** LLaMA 3.1 8B via Groq")

    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main chat area ──
if st.session_state.qa_chain is None:
    st.info("👈 Upload a PDF and click **Process Document** to get started")

    st.subheader("About this App")
    st.markdown("""
    This app implements a **Retrieval-Augmented Generation (RAG)** pipeline:

    **How it works:**
    1. 📄 **Ingest** — PDF is loaded and split into overlapping chunks
    2. 🔢 **Embed** — Each chunk converted to a vector using HuggingFace embeddings
    3. 🔍 **Retrieve** — Your question finds the top 4 most relevant chunks via FAISS
    4. 🤖 **Generate** — LLaMA 3.1 answers using only the retrieved context

    **Key Features:**
    - ✅ Hallucination guardrails — bot says "I don't know" if answer not in doc
    - ✅ Source grounding — answers based strictly on your document
    - ✅ Multi-turn chat — ask follow-up questions
    - ✅ Works on any PDF — research papers, manuals, reports, policies
    """)

else:
    st.subheader(f"💬 Chat with: {st.session_state.doc_name}")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(question)

                # Get source chunks
                sources = st.session_state.retriever.invoke(question)

            st.markdown(answer)

            # Show sources in expander
            with st.expander(f"📎 {len(sources)} source chunks retrieved"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', 'N/A')+1})")
                    st.text(doc.page_content[:300])
                    st.divider()

        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })