# RAG-based PDF Q&A with Streamlit

## 📌 Overview

**Ferrell Solutions** is a RAG-based PDF Q&A chatbot built with Streamlit, LangChain, and FAISS. Upload one or more PDFs, then ask questions grounded in their contents. The app extracts text (with OCR for pages that have little extractable text), indexes document chunks for retrieval, and sends relevant context to the configured language model to generate answers.

## ✨ Features

- 📂 **Upload up to 3 PDF files at a time** and search them together
- 🔎 **Extract PDF text with OCR fallback** for scanned or image-based pages
- 🧠 **RAG-powered Q&A** using LangChain, FAISS retrieval, and a configurable model
- 📄 **Source-aware context** with document and page metadata
- 📘 **Introduction page** explaining the chatbot and its RAG workflow

## 🏗️ Tech Stack

- **Python**: Primary language
- **Streamlit**: Frontend UI
- **OpenRouter API**: Chat and embedding model endpoint
- **FAISS**: Vector store for fast similarity search
- **LangChain**: Document processing and retrieval
- **PyPDFLoader**: Extracting text from PDFs
- **RapidOCR / PDFium**: OCR for image-based PDF pages

## 📂 Project Structure

📦 RAG-PDF-QA
┣ 📜 app.py # Multipage navigation entry point
┣ 📂 app_pages
┃ ┗ 📜 introduction.py # Chatbot introduction and RAG workflow
┣ 📜 main.py # PDF chat workspace
┣ 📜 requirements.txt # Dependencies list
┗ 📜 README.md # Documentation

bash
Copy
Edit

# 🚀 Installation & Setup

## 1️⃣ Clone the Repository
git clone https://github.com/your-username/RAG-PDF-QA.git
cd RAG-PDF-QA

## 2️⃣ Create a Virtual Environment (Optional but Recommended)

# On macOS/Linux
python -m venv venv
source venv/bin/activate  

# On Windows
python -m venv venv
venv\Scripts\activate

## 3️⃣ Install Dependencies
pip install -r requirements.txt

## 4️⃣ Run the Application
streamlit run app.py
🛠️ How It Works
1. Upload up to 3 PDF files in the chat workspace. This limit helps keep retrieved context within the language model's context capacity.
2. LangChain extracts page text and applies OCR when a page has little extractable text.
3. Extracted text is split into overlapping chunks, embedded, and indexed in FAISS.
4. For a question, the app retrieves the four most relevant chunks and provides them as context to the configured language model.
5. The model answers from the supplied context and is instructed to say when the answer is not present.
🎨 User Interface
🔹 Introduction
Learn what the chatbot does and how its retrieval-augmented generation workflow works.
🔹 PDF Chat
Upload and process documents, then ask questions in the conversation.
🔑 API Key Setup
Configure `OPENROUTER_API_KEY` in the local `.env` file. The `.env` file is excluded from Git; never commit API keys.
📝 Example Usage
plaintext
Copy
Edit
User: What is the main topic of this document?
AI: The document discusses...
⚡ Enhancements & Future Scope
 Support for multiple PDFs
 Fine-tuned models for domain-specific Q&A
 UI improvements and advanced query suggestions
 Persistent database storage for better scalability
🤝 Contribution
Contributions are welcome! Feel free to submit a pull request or open an issue.

📜 License
This project is licensed under the MIT License. See LICENSE for details.

🌟 If you find this project useful, consider giving it a ⭐ on GitHub!
