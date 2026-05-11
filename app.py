__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import tempfile
import os
from dotenv import load_dotenv

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="StudyBuddy Assistant",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #0f172a;
    color: white;
}

.hero-container {
    padding: 20px 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    text-align: center;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}

.hero-tagline {
    font-size: 1.2rem;
    color: #cbd5e1;
    margin: 10px auto 0;
    max-width: 800px;
}

.chat-user {
    padding: 20px;
    border-radius: 15px;
    background: #2563eb;
    margin-bottom: 15px;
}

.chat-ai {
    padding: 20px;
    border-radius: 15px;
    background: #1e293b;
    border: 1px solid #334155;
    margin-bottom: 15px;
}

.footer-name {
    text-align: center;
    margin-top: 50px;
    padding-bottom: 20px;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #30bdf8, #800cf8, #f492b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
}

/* Clear Chat Button Specific Styling */
.clear-btn > div > button {
    background: transparent !important;
    border: 1px solid #ef4444 !important;
    color: #ef4444 !important;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("🚀 Process Document")
    
    st.markdown("---")
    # Clear Chat Functionality
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# PROCESS DOCUMENTS
# =========================================================
if process_btn and uploaded_files:
    with st.spinner("Analyzing your study materials..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            try:
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        
        embedding_model = MistralAIEmbeddings(model="mistral-embed")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_ready = True
        st.success("✅ Ready to Study!")

# =========================================================
# HEADER DISPLAY
# =========================================================
st.markdown("""
<div class="hero-container">
    <span class="hero-title">📚 StudyBuddy Assistant</span>
    <p class="hero-tagline">
      <center>Your AI-powered learning companion for chatting with PDFs, 
        extracting knowledge, and simplifying study sessions instantly.
      </center>
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHAT LOGIC
# =========================================================
llm = ChatMistralAI(model="mistral-small-2506")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use ONLY the context provided to answer. If not found, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

query = st.chat_input("Ask questions from your uploaded PDFs...")

if query:
    if not st.session_state.vectorstore_ready:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        response = (prompt_template | llm).invoke({"context": context_text, "question": query})
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.content
        })

# =========================================================
# DISPLAY CHAT HISTORY (Removed View Sources)
# =========================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user"><b>🧑 You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai"><b>🤖 StudyBuddy:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown('<div class="footer-name">Developed By ❤️ Amit Kumar Rana</div>', unsafe_allow_html=True)
