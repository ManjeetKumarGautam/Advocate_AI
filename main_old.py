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

app = FastAPI(title="Optimized LLaMA RAG")

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "phi3"  # ⚡ faster than llama3.1

INDEX_FILE = "faiss_index.bin"
TEXT_FILE = "chunks.pkl"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384  # embedding size
index = faiss.IndexFlatL2(dimension)

texts = []

# ==========================================
# LOAD EXISTING DATA (PERSISTENCE)
# ==========================================

if os.path.exists(INDEX_FILE) and os.path.exists(TEXT_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(TEXT_FILE, "rb") as f:
        texts = pickle.load(f)

# ==========================================
# HELPERS
# ==========================================

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def get_embedding(text):
    return embed_model.encode([text])[0]


# def generate_stream(prompt):
#     response = requests.post(
#         OLLAMA_URL,
#         json={
#             "model": LLAMA_MODEL,
#             "prompt": prompt,
#             "stream": True
#         },
#         stream=True
#     )

#     for line in response.iter_lines():
#         if line:
#             try:
#                 data = eval(line.decode("utf-8"))
#                 if "response" in data:
#                     yield data["response"]
#             except:
#                 continue


def generate_stream(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    full_response = ""

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_response += data["response"]
            except:
                continue

    return full_response
# ==========================================
# TRAIN
# ==========================================

@app.post("/train")
async def train(file: UploadFile = File(...)):
    global index, texts

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

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

    # Save to disk
    faiss.write_index(index, INDEX_FILE)
    with open(TEXT_FILE, "wb") as f:
        pickle.dump(texts, f)

    return {"status": "trained", "chunks": len(chunks)}

# ==========================================
# ASK
# ==========================================

class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(query: Query):
    if index.ntotal == 0:
        raise HTTPException(400, "Train first")

    q_emb = np.array([get_embedding(query.question)]).astype("float32")

    k = 2  # ⚡ reduce context
    distances, indices = index.search(q_emb, k)

    context = "\n\n---\n\n".join([texts[i] for i in indices[0]])

    prompt = f"""
Answer ONLY from context.
If not found, say "I don't know".

CONTEXT:
{context}

QUESTION:
{query.question}
"""

    answer = generate_stream(prompt)

    return {
       "answer": answer if answer else "No response generated"
    }

# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():
    return {"msg": "Optimized RAG Running "}