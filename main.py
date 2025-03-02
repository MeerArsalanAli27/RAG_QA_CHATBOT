import streamlit as st
import os
import openai
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Global Variables
vector_db = None
retriever = None

# Set Page Config
st.set_page_config(page_title="📄 RAG-based PDF Q&A", layout="wide")

# Custom CSS for Beautiful UI
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        h1 {
            color: #4A90E2;
            text-align: center;
        }
        .sidebar .stTextInput {
            font-size: 16px;
        }
        .chat-container {
            background-color: #1e1e1e;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #4A90E2 !important;
            color: white !important;
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API Key
st.sidebar.title("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your API Key", type="password")

# Main Title
st.title("📄 RAG-based PDF Q&A")
st.markdown("Upload a **PDF**, enter your **OpenAI API key**, and ask questions!")

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")


# Function to Process PDF and Store in FAISS
def process_pdf(api_key, pdf_file):
    global vector_db, retriever
    
    if not api_key:
        st.error("❌ Please enter an API Key!")
        return

    os.environ["OPENAI_API_KEY"] = api_key

    with st.spinner("⏳ Processing PDF... Please wait!"):
        # Load PDF and Extract Text
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()

        # Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Create Vector Store
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(docs, embeddings)

        # Save Index
        with open("faiss_index.pkl", "wb") as f:
            pickle.dump(vector_db, f)

        # Create Retriever
        retriever = vector_db.as_retriever()
    
    st.success("✅ PDF processed successfully! You can now ask questions.")


# Process PDF Button
if uploaded_file and st.button("🚀 Process PDF"):
    process_pdf(api_key, uploaded_file)


# Function to Answer Questions
def answer_question(api_key, question):
    global retriever

    if not api_key:
        st.error("❌ Please enter an API Key!")
        return

    os.environ["OPENAI_API_KEY"] = api_key

    if retriever is None:
        st.warning("⚠️ No PDF processed yet. Please upload a file first.")
        return

    # Load FAISS Index
    with open("faiss_index.pkl", "rb") as f:
        vector_db = pickle.load(f)
    retriever = vector_db.as_retriever()

    # Create LLM Model
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Create Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Get Answer
    result = qa_chain.run(question)
    
    return result


# Chat-style Q&A Section
st.markdown("### 💬 Ask a Question")

question = st.text_input("❓ Enter your question")

if st.button("🤖 Get Answer"):
    if question:
        with st.spinner("💭 Thinking..."):
            answer = answer_question(api_key, question)
            st.markdown(f"""
            <div class="chat-container">
                <b>👤 You:</b> {question}  
                <br>
                <b>🤖 AI:</b> {answer}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question!")
