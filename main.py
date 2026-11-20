import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)
CHAT_MODEL = os.getenv(
    "NEMOTRON_CHAT_MODEL",
    "nvidia/nemotron-3.5-super-vl",
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)

st.set_page_config(
    page_title="PDF chat",
    page_icon=":material/auto_awesome:",
    layout="centered",
)

st.title("Chat with your PDF", icon=":material/auto_awesome:")
st.markdown(
    "Turn a document into a conversation. Upload a PDF, then ask questions "
    "about what it contains."
)

with st.sidebar:
    st.header("Workspace", icon=":material/tune:")
    if OPENROUTER_API_KEY:
        st.badge("OpenRouter connected", icon=":material/check_circle:", color="green")
    else:
        st.badge("API key not configured", icon=":material/warning:", color="orange")
    st.caption("Language model")
    st.code(CHAT_MODEL, language=None, wrap_lines=True)
    st.caption("Embedding model")
    st.code(EMBEDDING_MODEL, language=None, wrap_lines=True)
    st.caption("Upload a PDF and process it to start a new conversation.")
    api_key = OPENROUTER_API_KEY


def format_documents(documents):
    return "\n\n".join(document.page_content for document in documents)


def process_pdf(pdf_file, api_key):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name

    try:
        pages = PyPDFLoader(temp_path).load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        ).split_documents(pages)

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        vector_db = FAISS.from_documents(documents, embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 4})
    finally:
        os.unlink(temp_path)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(border=True):
    st.subheader("Your document", icon=":material/description:")
    st.caption("PDF files work best when their text is selectable.")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        accept_multiple_files=False,
    )

    if uploaded_file:
        file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("processed_file_id") != file_id:
            st.session_state.pop("retriever", None)
            st.session_state.pop("processed_file_id", None)

        st.caption(
            f":material/attach_file: {uploaded_file.name} · "
            f"{uploaded_file.size / 1024:.0f} KB"
        )
        if st.button(
            "Process document",
            icon=":material/auto_awesome:",
            type="primary",
            width="stretch",
        ):
            if not api_key:
                st.error("Please configure an OpenRouter API key.")
            else:
                with st.spinner("Reading and indexing your PDF..."):
                    try:
                        st.session_state.retriever = process_pdf(uploaded_file, api_key)
                        st.session_state.processed_file_id = file_id
                        st.session_state.messages = []
                        st.success("Document ready. Ask a question below.")
                    except Exception as exc:
                        st.error(f"Could not process the PDF: {exc}")

        if st.session_state.get("processed_file_id") == file_id:
            st.badge("Ready to chat", icon=":material/check_circle:", color="green")

if not uploaded_file and "processed_file_id" in st.session_state:
    st.session_state.pop("retriever", None)
    st.session_state.pop("processed_file_id", None)
    st.session_state.messages = []

st.subheader("Conversation", icon=":material/forum:")
if not st.session_state.messages and "retriever" not in st.session_state:
    with st.container(border=True):
        st.markdown(":material/chat_bubble_outline:")
        st.markdown("**Your conversation starts here**")
        st.caption("Upload and process a PDF, then ask a question in the chat box.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask a question about the PDF",
    submit_mode="disable",
)

if prompt:
    if not api_key:
        st.error("Please enter an API key.")
    elif "retriever" not in st.session_state:
        st.warning("Upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's question using only the provided PDF context. "
                    "If the context does not contain the answer, say that it is not "
                    "available in the document.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{question}",
                ),
            ]
        )
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            streaming=True,
            max_tokens=4096,
        )
        retriever = st.session_state.retriever
        chain = (
            {
                "context": retriever | format_documents,
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | llm
        )

        with st.chat_message("assistant"):
            response = st.write_stream(
                (chunk.text() for chunk in chain.stream(prompt))
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
