import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 8
SIM_THRESHOLD = 0.35

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(DEVICE)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
    
def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE-CHUNK_OVERLAP):
        chunk = " ".join(words[i:i+CHUNK_SIZE]).strip()
        if len(chunk)>200:
            chunks.append(chunk)
    unique_chunks = list(dict.fromkeys(chunks))
    return unique_chunks

def build_vector_store(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(question, index, chunks):
    q_embedding = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_embedding)
    scores, indices = index.search(q_embedding, TOP_K)
    filtered_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= SIM_THRESHOLD:
            filtered_chunks.append(chunks[idx])
    if not filtered_chunks:
        filtered_chunks = [chunks[i] for i in indices[0]]
    return filtered_chunks

def generate_answer(question, context):
    prompt = f""" 
You are an expert AI assistant.
Using ONLY the context below, provide a detailed explanation.
Avoid repitition of the same point in the answer.
If the answer is short in the context, expand logically but stay grounded.
Write a detailed answer including:
- A clear definition
- 5-6 bullet points explaining key concepts
- A short conclusion
- Use structured formatting

Context:
{context}

Question:
{question}

Detailed Answer:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = llm_model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_summary(context):
    prompt = f"""
You are a professional technical summarizer.

Summarize the document into:

1. 10 clear bullet points
2. Organized logically
3. Cover major topics in the document
4. Avoid repitition of the same point in summary
5. Keep it structured and coherent

Document:
{context}

Structured Summary:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = llm_model.generate(
        inputs["input_ids"],
        max_length=400,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_pdf(pdf_path):
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    index = build_vector_store(chunks)
    return{
        "chunks": chunks,
        "index": index
    }