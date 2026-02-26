# Smart PDF Question Answering Systwm (RAG-based)
A Retrieval-Augmented Generation (RAG) based PDF question Answering system built using **FastAPI, FAISS, Sentence Transformers and HuggingFace Transformers**.
This system allows users to upload a PDF document, ask contextual questions ans generate structured answers and summaries using Large Language Model.

---

## 🚀 Features

- 📂 Upload any text-based PDF
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answer generation
- 📑 Structured summary generation
- ⚡ FastAPI backend with Jinja2 templates
- 🎨 Bootstrap-based responsive UI
- 💾 Local model execution (No external API required)

---

## 🧠 Architecture (RAG Pipeline)

The system follows a *Retrieval-Augmented Generation (RAG)* approach:

### 1️⃣ PDF Processing
- Extracts text using PyPDF2
- Cleans and prepares raw text

### 2️⃣ Chunking
- Splits text into overlapping chunks  
- Configurable chunk size and overlap

### 3️⃣ Embedding
- Converts chunks into semantic vectors using:
sentence-transformers/all-MiniLM-L6-v2

### 4️⃣ Vector Storage
- Stores embeddings in a FAISS index
- Uses cosine similarity for retrieval

### 5️⃣ Retrieval
- Retrieves top-K relevant chunks based on semantic similarity
- Filters using similarity threshold

### 6️⃣ Generation
- Uses HuggingFace LLM:
google/flan-t5-base (or large)
- Structured prompt engineering
- Deterministic generation for stability

---

## 🏗️ Project Structure
pdf_qa_project/ 
├── static/    
    ├── style.css   
    ├── favicon files
├── templates/ 
    └── index.html
├── uploads/
├── app.py 
├── rag_engine.py 
├── requirements.txt 
└── README.md

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/Aayushi-777/pdf-qa-project.git
cd pdf-qa-project
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependancies

```bash
pip install -r requirements.txt
```

## Technologies used

python
FastAPI
Jinja2
FAISS
Sentence Transformers
HuggingFace Transformers
PyPDF2
Bootstrap 5