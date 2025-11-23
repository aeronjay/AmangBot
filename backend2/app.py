import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import uvicorn

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust paths based on the workspace structure provided
CHUNKS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../BART/Dataset/Chunk2.json"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))

app = FastAPI(title="AmangBot Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
chunks = []
index = None
embedder = None
llm = None

class ChatRequest(BaseModel):
    message: str
    k: Optional[int] = 3

class ChatResponse(BaseModel):
    response: str
    context: List[str]

@app.on_event("startup")
async def startup_event():
    global chunks, index, embedder, llm
    
    # 1. Load Chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        # Extract content from chunks. The file structure is a list of dicts.
        # Based on attachment, items have "content" field.
        chunks = [item.get("content", "") for item in chunks_data if "content" in item]
    
    print(f"Loaded {len(chunks)} chunks.")

    # 2. Initialize Embedder
    print("Initializing Nomic Embedder (nomic-ai/nomic-embed-text-v1.5)...")
    # trust_remote_code=True is needed for Nomic
    finetuned_model_path = "../Models/nomic-finetuned/nomic-finetuned-final"
    embedder = SentenceTransformer(finetuned_model_path, trust_remote_code=True)

    # 3. Build FAISS Index
    print("Building FAISS index...")
    # Nomic v1.5 uses "search_document: " prefix for documents
    chunks_for_embedding = [f"search_document: {c}" for c in chunks]
    
    # Encode and normalize for Cosine Similarity via L2
    embeddings = embedder.encode(chunks_for_embedding, convert_to_numpy=True, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 4. Initialize LLM
    print(f"Initializing Llama model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # Offload all layers to GPU
        n_batch=4096,         
        n_ctx=4096,           # Match your model's context window
        verbose=False         # Logs off for production speed
    )
    print("Llama model initialized.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global index, chunks, embedder, llm
    
    if not index or not embedder or not llm:
        raise HTTPException(status_code=503, detail="Models not initialized")

    # 1. Embed query
    # Nomic v1.5 uses "search_query: " prefix for queries
    query_text = f"search_query: {request.message}"
    query_embedding = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    
    # 2. Search FAISS
    D, I = index.search(query_embedding, request.k)
    
    # 3. Retrieve context
    # I[0] contains the indices of the neighbors
    retrieved_chunks = [chunks[i] for i in I[0]]
    context_str = "\n\n".join(retrieved_chunks)
    
    # 4. Construct Prompt (Mistral Instruct Format)
    # <s>[INST] Instruction [/INST]
    system_prompt = "You are a helpful assistant for EARIST students. Answer the question based only on the provided context."
    prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {request.message} [/INST]"
    
    # 5. Generate
    output = llm(
        prompt,
        max_tokens=512,
        stop=["</s>"],
        echo=False
    )
    
    response_text = output["choices"][0]["text"].strip()
    
    return ChatResponse(
        response=response_text,
        context=retrieved_chunks
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
