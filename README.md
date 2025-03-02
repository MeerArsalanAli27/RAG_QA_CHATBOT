# RAG-based PDF Q&A with Streamlit

## 📌 Overview

This project is a **RAG-based PDF Q&A System** built using **Streamlit, OpenAI API, FAISS, and LangChain**. The application allows users to upload a **PDF file**, process it into a **vector database** (FAISS), and then interact with the document by asking **questions**. The system retrieves relevant content and generates AI-powered answers using OpenAI's GPT-4 model.

## ✨ Features

- 📂 **Upload PDF Files** and extract text for processing
- 🔑 **Secure API Key Input** via Streamlit Sidebar
- 🧠 **AI-Powered Q&A** using OpenAI GPT-4 and FAISS-based retrieval
- 🔍 **Efficient Text Chunking** to enhance retrieval accuracy
- 🚀 **Interactive and User-Friendly Interface** with Styled Chat UI
- 💾 **Persistent Vector Database Storage** for efficient retrieval

## 🏗️ Tech Stack

- **Python**: Primary language
- **Streamlit**: Frontend UI
- **OpenAI API**: Language model for answering queries
- **FAISS**: Vector store for fast similarity search
- **LangChain**: Document processing and retrieval
- **PyPDFLoader**: Extracting text from PDFs
- **Pickle**: Storing and loading FAISS index

## 📂 Project Structure

📦 RAG-PDF-QA ┣ 📜 app.py # Main application file ┣ 📜 requirements.txt # Dependencies list ┣ 📜 README.md # Documentation (this file) ┣ 📦 data/ # Folder for storing FAISS index ┗ 📦 assets/ # Static assets (if needed)

bash
Copy
Edit

🚀 Installation & Setup
1️⃣ Clone the Repository
To get started, clone the repository to your local machine using the following command:

bash
Copy
Edit
git clone https://github.com/your-username/RAG-PDF-QA.git
cd RAG-PDF-QA
2️⃣ Create a Virtual Environment (Optional but Recommended)
It is recommended to create a virtual environment to manage dependencies separately.

For macOS/Linux:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate
For Windows:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
Once inside the virtual environment, install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the Application
Start the Streamlit application by running:

bash
Copy
Edit
streamlit run app.py
🛠️ How It Works
Upload a PDF: Select and upload a PDF file through the Streamlit interface.
Process the PDF: The app extracts text, chunks it, and stores it in a FAISS vector database.
Ask Questions: Enter your OpenAI API key and type your question related to the document.
Receive AI Answers: The system retrieves the most relevant information from the PDF and generates an answer using GPT-4.
🎨 User Interface
🔹 Sidebar
Enter your OpenAI API key securely.
🔹 Main Page
Upload a PDF file.
Process and store document embeddings.
Input questions and receive AI-generated answers.
🔑 API Key Setup
To use the OpenAI API, you need an API key. You can get it from OpenAI.

Enter the key in the sidebar input field.
The key is securely used during the session but not stored permanently.
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
