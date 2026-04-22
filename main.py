from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import PyPDF2
import numpy as np
import io
import requests
import faiss
import os
import json
import pickle
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIG
# ==========================================

app = FastAPI(title="Auto-Routing RAG")

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "phi3"

DATA_DIR = "data"
MAP_FILE = "file_map.pkl"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384

os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# HELPERS
# ==========================================

def get_paths(filename):
    folder = os.path.join(DATA_DIR, filename.replace(".pdf", ""))
    os.makedirs(folder, exist_ok=True)
    return (
        os.path.join(folder, "index.bin"),
        os.path.join(folder, "chunks.pkl")
    )

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    return embed_model.encode([text])[0]

def load_file_map():
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_file_map(file_map):
    with open(MAP_FILE, "wb") as f:
        pickle.dump(file_map, f)

def generate(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    return data.get("response", "")

# ==========================================
# TRAIN
# ==========================================

@app.post("/train")
async def train(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    index_file, text_file = get_paths(file.filename)

    index = faiss.IndexFlatL2(dimension)
    texts = []

    contents = await file.read()
    pdf = PyPDF2.PdfReader(io.BytesIO(contents))

    raw_text = ""
    for page in pdf.pages:
        t = page.extract_text()
        if t:
            raw_text += t + "\n"

    if not raw_text.strip():
        raise HTTPException(400, "Empty PDF")

    chunks = chunk_text(raw_text)

    vectors = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        vectors.append(emb)
        texts.append(chunk)

    vectors = np.array(vectors).astype("float32")
    index.add(vectors)

    # Save index + chunks
    faiss.write_index(index, index_file)
    with open(text_file, "wb") as f:
        pickle.dump(texts, f)

    # 🔥 Save file-level embedding (for auto routing)
    file_map = load_file_map()

    summary = raw_text[:2000]  # simple summary
    file_map[file.filename] = get_embedding(summary)

    save_file_map(file_map)

    return {"status": "trained", "file": file.filename, "chunks": len(chunks)}

# ==========================================
# ASK (AUTO FILE DETECTION)
# ==========================================

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    file_map = load_file_map()

    if not file_map:
        raise HTTPException(400, "No documents trained")

    query_emb = get_embedding(query.question)

    #  Step 1: Find best file
    best_file = None
    best_score = float("inf")

    for fname, emb in file_map.items():
        score = np.linalg.norm(query_emb - emb)
        if score < best_score:
            best_score = score
            best_file = fname

    if not best_file:
        raise HTTPException(400, "No relevant file found")

    #  Step 2: Load that file's index
    index_file, text_file = get_paths(best_file)

    if not os.path.exists(index_file):
        raise HTTPException(400, "Index missing")

    index = faiss.read_index(index_file)
    with open(text_file, "rb") as f:
        texts = pickle.load(f)

    q_emb = np.array([query_emb]).astype("float32")

    distances, indices = index.search(q_emb, 3)

    context = "\n\n---\n\n".join([texts[i] for i in indices[0]])

    prompt = f"""
Answer ONLY from context.
If not found, say "I don't know".

CONTEXT:
{context}

QUESTION:
{query.question}
"""

    answer = generate(prompt)

    return {
        "file_used": best_file,
        "answer": answer if answer else "No response generated"
    }

# ==========================================
# LIST FILES
# ==========================================

@app.get("/files")
def list_files():
    file_map = load_file_map()
    return {"files": list(file_map.keys())}

# ==========================================
# DELETE FILE
# ==========================================

@app.delete("/delete/{filename}")
def delete_file(filename: str):
    file_map = load_file_map()

    if filename not in file_map:
        raise HTTPException(404, "File not found")

    folder = os.path.join(DATA_DIR, filename.replace(".pdf", ""))

    # delete folder
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        os.rmdir(folder)

    # remove from map
    del file_map[filename]
    save_file_map(file_map)

    return {"status": "deleted", "file": filename}

# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():
    return {"msg": "Auto-Routing RAG Running..."}