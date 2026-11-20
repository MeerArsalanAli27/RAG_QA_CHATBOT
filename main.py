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
    return "\n\n".join(
        f"[Source: {document.metadata.get('source', 'PDF')}, "
        f"page {document.metadata.get('page', 0) + 1}]\n"
        f"{document.page_content}"
        for document in documents
    )


def get_pdf_data(pdf_file, index):
    if isinstance(pdf_file, (bytes, bytearray, memoryview)):
        return f"document_{index}.pdf", bytes(pdf_file)
    if not hasattr(pdf_file, "getvalue") or not hasattr(pdf_file, "name"):
        raise TypeError("Each uploaded PDF must be a file upload or PDF byte string.")
    return pdf_file.name, pdf_file.getvalue()


def normalize_pdf_uploads(upload_value):
    if upload_value is None:
        return []
    if isinstance(upload_value, (list, tuple)):
        return list(upload_value)
    return [upload_value]


@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.limit_side_len": 680,
            "Det.limit_type": "max",
        }
    )


def add_ocr_for_image_pages(pdf_path, pages, progress_callback=None):
    pages_needing_ocr = [
        index for index, page in enumerate(pages)
        if len(page.page_content.strip()) < 50
    ]
    if not pages_needing_ocr:
        return pages

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(
            "OCR support is required for scanned PDFs. Install the project "
            "requirements, including rapidocr and pypdfium2."
        ) from exc

    ocr = get_ocr_engine()
    pdf_document = pdfium.PdfDocument(pdf_path)
    try:
        for page_index in pages_needing_ocr:
            pdf_page = pdf_document[page_index]
            try:
                bitmap = pdf_page.render(scale=0.85)
                try:
                    result = ocr(bitmap.to_numpy())
                    recognized_text = (
                        "\n".join(result.txts).strip()
                        if result and result.txts
                        else ""
                    )
                finally:
                    bitmap.close()
            finally:
                pdf_page.close()

            if recognized_text:
                pages[page_index].page_content = "\n".join(
                    part
                    for part in (
                        pages[page_index].page_content.strip(),
                        recognized_text,
                    )
                    if part
                )
            if progress_callback:
                progress_callback(page_index + 1, len(pages))
    finally:
        pdf_document.close()

    return pages


def load_pdf_documents(pdf_files, progress_callback=None):
    pages = []
    for index, pdf_file in enumerate(pdf_files, start=1):
        filename, pdf_data = get_pdf_data(pdf_file, index)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name

        try:
            loaded_pages = PyPDFLoader(temp_path).load()
            def report_page_progress(page_index, page_count):
                if progress_callback:
                    progress_callback(
                        0.85
                        * (index - 1 + page_index / page_count)
                        / len(pdf_files),
                        f"Reading {filename}: page {page_index} of {page_count}",
                    )

            loaded_pages = add_ocr_for_image_pages(
                temp_path,
                loaded_pages,
                progress_callback=report_page_progress,
            )
            for page in loaded_pages:
                page.metadata["source"] = filename
            pages.extend(loaded_pages)
            if progress_callback:
                progress_callback(
                    0.85 * index / len(pdf_files),
                    f"Finished reading {filename}",
                )
        finally:
            os.unlink(temp_path)

    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "No readable text was found in the selected PDFs. OCR could not "
            "recognize text in their pages."
        )

    documents = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    ).split_documents(pages)
    if not documents:
        raise ValueError("No readable text was found in the selected PDFs.")
    return documents


def process_pdfs(pdf_files, api_key, progress_callback=None):
    documents = load_pdf_documents(pdf_files, progress_callback=progress_callback)
    if progress_callback:
        progress_callback(0.9, "Creating the searchable index...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    vector_db = FAISS.from_documents(documents, embeddings)
    if progress_callback:
        progress_callback(1.0, "Documents are ready")
    return vector_db.as_retriever(search_kwargs={"k": 4})


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(border=True):
    st.subheader("Your document", icon=":material/description:")
    st.caption(
        "Add one or more PDFs. Scanned and image-based pages are processed with OCR."
    )
    uploaded_value = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    uploaded_files = normalize_pdf_uploads(uploaded_value)

    if uploaded_files:
        file_digest = hashlib.sha256()
        pdf_details = [
            get_pdf_data(pdf_file, index)
            for index, pdf_file in enumerate(uploaded_files, start=1)
        ]
        for name, pdf_data in pdf_details:
            file_digest.update(name.encode("utf-8"))
            file_digest.update(pdf_data)
        file_id = file_digest.hexdigest()

        if st.session_state.get("processed_file_id") != file_id:
            st.session_state.pop("retriever", None)
            st.session_state.pop("processed_file_id", None)
            st.session_state.messages = []

        pdf_count = len(pdf_details)
        st.caption(f"{pdf_count} PDF file(s) selected")
        for name, pdf_data in pdf_details:
            st.markdown(
                f"- :material/description: **{name}** "
                f"({len(pdf_data) / 1024:.0f} KB)"
            )

        if st.button(
            "Process documents",
            icon=":material/auto_awesome:",
            type="primary",
            width="stretch",
        ):
            if not api_key:
                st.error("Please configure an OpenRouter API key.")
            else:
                progress = st.progress(0, text="Preparing documents...")

                def update_progress(fraction, message):
                    progress.progress(fraction, text=message)

                try:
                    with st.spinner("Processing documents..."):
                        st.session_state.retriever = process_pdfs(
                            uploaded_files,
                            api_key,
                            progress_callback=update_progress,
                        )
                        st.session_state.processed_file_id = file_id
                        st.session_state.messages = []
                    progress.empty()
                    st.success(
                        f"{pdf_count} document(s) ready. "
                        "Ask a question below."
                    )
                except Exception as exc:
                    progress.empty()
                    st.error(f"Could not process the selected PDFs: {exc}")

        if st.session_state.get("processed_file_id") == file_id:
            st.badge("Ready to chat", icon=":material/check_circle:", color="green")

if not uploaded_files and "processed_file_id" in st.session_state:
    st.session_state.pop("retriever", None)
    st.session_state.pop("processed_file_id", None)
    st.session_state.messages = []

st.subheader("Conversation", icon=":material/forum:")
if not st.session_state.messages and "retriever" not in st.session_state:
    with st.container(border=True):
        st.markdown(":material/chat_bubble_outline:")
        st.markdown("**Your conversation starts here**")
        st.caption("Upload and process one or more PDFs, then ask a question.")

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
