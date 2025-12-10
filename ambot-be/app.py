import os
import json
import glob
import time
import asyncio
import numpy as np
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
from rank_bm25 import BM25Okapi

# Configuration
MODEL_PATH = "../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DATASET_PATH = "../Dataset/300TokenDataset"
INDEX_FILE = "faiss_index_finetuned.bin"
BM25_INDEX_FILE = "bm25_index.pkl"
METADATA_FILE = "chunks_metadata.json"
EMBEDDING_MODEL_NAME = "../Models/nomic-finetuned/nomic-finetuned-final"
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

# Global variables
llm = None
embedder = None
index = None
bm25 = None
chunks_metadata = []
reranker_model = None
reranker_tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

def load_resources():
    global llm, embedder, index, chunks_metadata, reranker_model, reranker_tokenizer, bm25
    
    # Determine device for embedding model
    # If index exists, we only need CPU for inference (saving GPU for LLM)
    # If index doesn't exist, we need GPU for faster embedding creation
    index_exists = os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)
    device = 'cpu' if index_exists else 'cuda'
    
    print(f"Loading Embedding Model on {device}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device, trust_remote_code=True)
    
    print("Loading LLM...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # Offload all layers to GPU
        n_batch=1024,
        n_ctx=6144,
        verbose=False
    )

    print("Loading Reranker on GPU...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, trust_remote_code=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_NAME, 
        dtype="auto", 
        trust_remote_code=True
    )
    reranker_model.to('cuda')
    reranker_model.eval()

    if index_exists:
        print("Loading FAISS index and metadata...")
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)
            
        if os.path.exists(BM25_INDEX_FILE):
            print("Loading BM25 index...")
            with open(BM25_INDEX_FILE, 'rb') as f:
                bm25 = pickle.load(f)
        else:
            print("BM25 index missing. Recreating indexes...")
            create_index()
    else:
        print("Creating FAISS index...")
        create_index()
        # After creating index on GPU, we could potentially reload on CPU or just keep it.
        # Since the process is already running, we'll keep it as is for this session.
        # Next restart will pick up the index and use CPU.

def create_index():
    global index, chunks_metadata, bm25
    
    chunks_metadata = []
    texts = []
    
    # Load all JSON files
    json_files = glob.glob(os.path.join(DATASET_PATH, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'content' in item:
                            chunks_metadata.append(item)
                            # Embed necessary metadata along with content
                            # Format: Source: [Source]; Category: [Category]; Topic: [Topic]; Content: [Content]
                            source = item.get('source', 'Unknown')
                            category = item.get('category', 'Unknown')
                            topic = item.get('topic', 'Unknown')
                            content = item.get('content', '')
                            
                            text_to_embed = f"Source: {source}; Category: {category}; Topic: {topic}; Content: {content}"
                            texts.append(text_to_embed)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not texts:
        print("No chunks found!")
        return

    print(f"Encoding {len(texts)} chunks...")
    # Add prefix for documents as required by nomic-embed-text-v1.5
    texts_with_prefix = ["search_document: " + t for t in texts]
    embeddings = embedder.encode(texts_with_prefix, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Create BM25 index
    print("Creating BM25 index...")
    # Simple whitespace tokenization for BM25
    tokenized_corpus = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(BM25_INDEX_FILE, 'wb') as f:
        pickle.dump(bm25, f)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, ensure_ascii=False, indent=2)
    
    print("Index created and saved.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_resources()
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def construct_prompt(query: str, chunks: List[dict]) -> str:
    context_text = ""
    for i, chunk in enumerate(reversed(chunks)):
        context_text += f"[Source {i+1}: {chunk.get('source', 'Unknown')}]\n"
        context_text += f"Category: {chunk.get('category', 'Unknown')}\n"
        context_text += f"Topic: {chunk.get('topic', 'Unknown')}\n"
        context_text += f"{chunk.get('content', '')}\n\n"

    prompt = f"""[INST] You are "AmangBot", a friendly and knowledgeable AI student assistant for EARIST (Eulogio "Amang" Rodriguez Institute of Science and Technology).

YOUR PERSONALITY:
- Be warm, approachable, and conversational - like a helpful senior student or advisor
- Use natural language and be encouraging
- If this is a follow-up question, acknowledge the connection to the previous topic naturally

RESPONSE STRUCTURE:
1. FIRST, directly answer the student's question based on the provided context sources
2. THEN, provide additional related information that might be helpful to the student

RESPONSE RULES:
1. ALWAYS base your answer ONLY on the provided context sources - never make up information
2. Start your direct answer with "According to [Source Name], ..." citing the specific source
3. If multiple sources are relevant, cite each one: "According to [Source 1], ... Additionally, [Source 2] states that..."
4. After answering the main question, add a section like "You might also find this helpful:" or "Related information:" to share additional relevant details from the sources (such as deadlines, requirements, procedures, tips, or related topics)
5. Use bullet points or numbered lists for multiple items, steps, or requirements
6. If the information is NOT in the provided context, respond: "I'm sorry, I don't have specific information about that in my current sources. You may want to check with the EARIST registrar or relevant office for the most accurate details."
7. End with a helpful follow-up when appropriate, like "Is there anything else you'd like to know about this?" or "Would you like more details about any specific part?"

Context from EARIST Sources:
{context_text}

Student Question: {query} [/INST]"""
    return prompt

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    query = request.message
    
    # Embed query with prefix as required by nomic-embed-text-v1.5
    query_embedding = embedder.encode(["search_query: " + query]).astype('float32')
    
    # Search FAISS
    k = 20 # Retrieve more candidates for reranking
    D, I = index.search(query_embedding, k)
    faiss_indices = I[0]
    
    # Search BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top k indices from BM25
    bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    
    # Combine indices (Union)
    combined_indices = list(set(faiss_indices) | set(bm25_indices))
    
    initial_chunks = []
    for idx in combined_indices:
        if idx < len(chunks_metadata) and idx >= 0:
            initial_chunks.append(chunks_metadata[idx])
            
    # Reranking
    if not initial_chunks:
        retrieved_chunks = []
    else:
        # Prepare pairs for reranking
        pairs = [[query, chunk.get('content', '')] for chunk in initial_chunks]
        
        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            scores = reranker_model(**inputs).logits.squeeze(-1)
            
        # Sort by score descending
        sorted_indices = scores.argsort(descending=True)
        
        # Select top k
        k_final = 5
        top_indices = sorted_indices[:k_final].tolist()
        
        retrieved_chunks = [initial_chunks[i] for i in top_indices]
            
    # Token limit check
    # n_ctx (6144) - max_tokens (1660) = 4484 available for prompt
    MAX_PROMPT_TOKENS = 4484
    
    while True:
        prompt = construct_prompt(query, retrieved_chunks)
        tokens = llm.tokenize(prompt.encode('utf-8'))
        
        if len(tokens) <= MAX_PROMPT_TOKENS:
            break
            
        if not retrieved_chunks:
            break
            
        # Remove least relevant chunk (last one in the list)
        retrieved_chunks.pop()
    
    async def event_generator():
        # Send metadata first
        metadata = {
            "type": "metadata",
            "chunks": retrieved_chunks,
            "retrieved_chunks": retrieved_chunks, # Sending same for now as we don't have a separate reranking step yet
            "prompt": prompt
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Stream tokens
        stream = llm(
            prompt,
            temperature=0.1,       # <--- ADD THIS (0.1 to 0.3 is best for factual bots)
            top_p=0.9,
            max_tokens=1660,
            stop=["</s>", "[/INST]"],
            echo=False,
            stream=True
        )
        
        for output in stream:
            token = output['choices'][0]['text']
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            await asyncio.sleep(0)
            
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
