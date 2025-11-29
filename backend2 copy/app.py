import os
import json
import faiss
import numpy as np
import contextlib
import time
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import uvicorn

# Import auth-related modules
# Assuming these exist in your file structure
from config.database import connect_to_mongo, close_mongo_connection
from routes.auth import router as auth_router

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Dataset/300TokenDataset"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))
NOMIC_MODEL_PATH = "../Models/nomic-finetuned/nomic-finetuned-final"

# --- Tuning Parameters ---
# MS-MARCO CrossEncoder outputs raw logits (not 0-1 probability).
# > 0 is usually relevant. < -4 is usually irrelevant.
# We set a threshold: if the top chunk scores lower than this, we don't know the answer.
RERANKER_THRESHOLD = -8.9

# --- Globals State Storage ---
class GlobalState:
    chunks: List[dict] = []
    index = None
    embedder: Optional[SentenceTransformer] = None
    llm: Optional[Llama] = None
    reranker: Optional[CrossEncoder] = None

state = GlobalState()

# --- Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    k: Optional[int] = 4 

class ChunkData(BaseModel):
    content: str
    source: str
    category: str
    topic: str

class ChatResponse(BaseModel):
    response: str
    context: List[str]
    chunks: List[ChunkData] = []
    prompt: str = ""
    retrieved_chunks: List[ChunkData] = []

# --- Lifespan Manager (Modern Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect DB
    await connect_to_mongo()
    
    # 2. Load Dataset
    print(f"Loading chunks from {DATASET_DIR}...")
    if os.path.exists(DATASET_DIR):
        loaded_chunks = []
        chunks_for_embedding = []
        for filename in os.listdir(DATASET_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(DATASET_DIR, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                if "content" in item:
                                    chunk_data = {
                                        "content": item["content"],
                                        "source": item.get("source", "EARIST Database"),
                                        "category": item.get("category", ""),
                                        "topic": item.get("topic", "")
                                    }
                                    loaded_chunks.append(chunk_data)
                                    
                                    # Prepare metadata string for embedding
                                    metadata_parts = [f"{k}: {v}" for k, v in item.items() if k not in ["id", "keywords", "content"]]
                                    metadata_str = ", ".join(metadata_parts)
                                    chunks_for_embedding.append(f"search_document: {metadata_str}. {item['content']}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        state.chunks = loaded_chunks
        print(f"Loaded {len(state.chunks)} chunks.")
    else:
        print("Warning: Dataset directory not found.")

    # 3. Load Embedder
    print("Initializing Nomic Embedder...")
    state.embedder = SentenceTransformer(NOMIC_MODEL_PATH, trust_remote_code=True, device='cpu')

    # 4. Build/Load Index
    if chunks_for_embedding:
        print("Building FAISS index...")
        # Run blocking encode in threadpool even during startup to prevent loop lag
        embeddings = await run_in_threadpool(
            state.embedder.encode, 
            chunks_for_embedding, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        dimension = embeddings.shape[1]
        state.index = faiss.IndexFlatL2(dimension)
        state.index.add(embeddings)
        print(f"FAISS index built: {state.index.ntotal} vectors.")

    # 5. Load Reranker
    try:
        print("Loading Reranker...")
        state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    except Exception as e:
        print(f"Failed to load Reranker: {e}")
        state.reranker = None

    # 6. Load LLM
    print(f"Loading Mistral Model...")
    # n_gpu_layers=-1 attempts to offload all layers to GPU if available
    state.llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_batch=1024,        
        n_ctx=6144,          
        verbose=False        
    )
    
    yield # Server runs here
    
    # Shutdown logic
    print("Shutting down...")
    await close_mongo_connection()

# --- App Init ---
app = FastAPI(title="AmangBot Backend", lifespan=lifespan)

app.include_router(auth_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions (Logic) ---

def is_conversational_filler(text: str) -> bool:
    fillers = ["hi", "hello", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye"]
    return text.lower().strip() in fillers

def needs_contextualization(message: str) -> bool:
    msg_lower = message.lower().strip()
    # Simple check: short queries or those starting with pronouns
    triggers = ["it", "that", "this", "they", "them", "he", "she", "similar", "also", "and"]
    
    word_count = len(msg_lower.split())
    if word_count < 3 and not any(w in msg_lower for w in ["who", "what", "where", "when", "why", "how"]):
        return True
    
    if any(f" {t} " in f" {msg_lower} " for t in triggers):
        return True
        
    return False

# --- Async Wrappers for Blocking Calls ---

async def async_embed(texts: List[str]):
    """Non-blocking wrapper for embedding generation."""
    return await run_in_threadpool(
        state.embedder.encode, 
        texts, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

async def async_search_index(query_vector, k: int):
    """Non-blocking wrapper for FAISS search."""
    return await run_in_threadpool(state.index.search, query_vector, k)

async def async_rerank(pairs):
    """Non-blocking wrapper for CrossEncoder."""
    if not state.reranker:
        return []
    return await run_in_threadpool(state.reranker.predict, pairs)

async def async_llm_generate(prompt: str, max_tokens=1024, stop=None, temp=0.1):
    """Non-blocking wrapper for LlamaCPP generation."""
    return await run_in_threadpool(
        state.llm,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temp,
        stop=stop or [],
        echo=False
    )

async def async_llm_stream(prompt: str, max_tokens=1024, stop=None, temp=0.1) -> AsyncGenerator[str, None]:
    """Streaming wrapper for LlamaCPP generation that yields tokens."""
    def generate_stream():
        return state.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop=stop or [],
            echo=False,
            stream=True
        )
    
    # Get the generator in a thread pool
    stream = await run_in_threadpool(generate_stream)
    
    # Iterate over the stream
    for output in stream:
        token = output["choices"][0]["text"]
        yield token

# --- Core Logic ---

async def get_contextualized_query(history: List[Message], latest_message: str) -> str:
    """Refined prompt to handle topic switching better."""
    if not history or not needs_contextualization(latest_message):
        return latest_message

    conversation_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-4:]])
    
    prompt = f"""[INST] You are a query reformulator for a university chatbot.
    
Task: Rewrite the "User Question" to be standalone based on the "Chat History" ONLY IF it refers to previous context.

Rules:
1. If the User Question is a NEW TOPIC or unrelated to the history, output it EXACTLY AS IS.
2. If the User Question uses pronouns (it, that, they) referring to the history, replace them with the specific nouns.
3. Output ONLY the reformulated question. No explanations.

Chat History:
{conversation_str}

User Question: {latest_message}
Output: [/INST]"""

    # Low max_tokens for speed, temp 0 for determinism
    output = await async_llm_generate(prompt, max_tokens=60, stop=["\n", "[/INST]"], temp=0.0)
    new_query = output["choices"][0]["text"].strip().strip('"')
    
    print(f"Contextualizer: '{latest_message}' -> '{new_query}'")
    return new_query if len(new_query) > 2 else latest_message

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    total_start = time.perf_counter()
    
    # 1. Fast fail for fillers
    if is_conversational_filler(request.message):
        return ChatResponse(
            response="Hello! I am AmangBot. How can I help you with your EARIST concerns?",
            context=[]
        )

    # 2. Contextualize (Non-blocking)
    context_start = time.perf_counter()
    standalone_query = await get_contextualized_query(request.history, request.message)
    context_time = time.perf_counter() - context_start

    # 3. Embedding (Non-blocking)
    embed_start = time.perf_counter()
    query_text = f"search_query: {standalone_query}"
    query_embedding = await async_embed([query_text])
    embed_time = time.perf_counter() - embed_start

    # 4. Retrieval (Hybrid Strategy)
    retrieval_start = time.perf_counter()
    # Fetch 3x candidates to allow Reranker to filter bad ones
    initial_k = request.k * 3
    D, I = await async_search_index(query_embedding, initial_k)
    
    retrieved_chunks = [state.chunks[i] for i in I[0] if i < len(state.chunks)]
    retrieval_time = time.perf_counter() - retrieval_start

    final_chunks = []
    
    # 5. Reranking & Thresholding (The Guardrail)
    rerank_start = time.perf_counter()
    if state.reranker:
        pairs = [[standalone_query, chunk["content"]] for chunk in retrieved_chunks]
        scores = await async_rerank(pairs)
        
        print(f"\n--- QUERY: {standalone_query} ---")
        for doc, score in zip(retrieved_chunks, scores):
            print(f"Score: {score:.4f} | Text: {doc['content'][:50]}...")

        # Zip chunks with scores
        scored_chunks = list(zip(retrieved_chunks, scores))
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # CHECK THRESHOLD on the very best match
        best_score = scored_chunks[0][1] if scored_chunks else -999
        print(f"Top Reranker Score: {best_score}")

        if best_score < RERANKER_THRESHOLD:
            rerank_time = time.perf_counter() - rerank_start
            total_time = time.perf_counter() - total_start
            
            # Build retrieved chunks data for frontend even on failure
            all_retrieved_chunks = [
                ChunkData(
                    content=c["content"], 
                    source=c["source"], 
                    category=c.get("category", ""), 
                    topic=c.get("topic", "")
                ) 
                for c in retrieved_chunks
            ]
            
            print(f"\n{'='*50}")
            print(f"⏱️  TIMING BREAKDOWN (Query rejected - below threshold)")
            print(f"{'='*50}")
            print(f"  Contextualization: {context_time*1000:.2f} ms")
            print(f"  Embedding:         {embed_time*1000:.2f} ms")
            print(f"  Retrieval (FAISS): {retrieval_time*1000:.2f} ms")
            print(f"  Reranking:         {rerank_time*1000:.2f} ms")
            print(f"  Generation:        N/A (query rejected)")
            print(f"{'='*50}")
            print(f"  TOTAL:             {total_time*1000:.2f} ms")
            print(f"{'='*50}\n")
            
            # Print retrieved chunks even on failure
            print(f"\n{'='*50}")
            print(f"📦 RETRIEVED CHUNKS (Query rejected - below threshold)")
            print(f"{'='*50}")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\nChunk {i}:")
                print(f"  Source: {chunk.get('source', 'N/A')}")
                print(f"  Category: {chunk.get('category', 'N/A')}")
                print(f"  Topic: {chunk.get('topic', 'N/A')}")
                print(f"  Content: {chunk['content'][:200]}...")
            print(f"{'='*50}\n")
            
            # If the best match is garbage, return "Unknown" immediately
            return ChatResponse(
                response="I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database.",
                context=[],
                chunks=[],
                prompt="(No prompt generated - query rejected due to low relevance score)",
                retrieved_chunks=all_retrieved_chunks
            )

        # Take top K
        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        # Fallback if reranker fails to load
        final_chunks = retrieved_chunks[:request.k]
    rerank_time = time.perf_counter() - rerank_start

    # 6. Construct Prompt
    context_parts = []
    sources_used = set()
    for i, chunk in enumerate(final_chunks, 1):
        src = chunk.get("source", "EARIST Database")
        sources_used.add(src)
        context_parts.append(f"[Source {i}: {src}]\n{chunk['content']}")
    
    context_str = "\n\n".join(context_parts)
    sources_str = ", ".join(sources_used)

    system_prompt = f"""You are "AmangBot", a helpful AI student assistant for EARIST.

GUIDELINES:
1. Answer the question using ONLY the provided context.
2. CITATION IS MANDATORY: Start your answer with "According to [Source Name]...".
3. If the answer is not in the context, say "I don't know."
4. Be polite and concise.
5. Format with bullet points for lists.

Sources: {sources_str}
"""
    full_prompt = f"[INST] {system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {standalone_query} [/INST]"

    # Check token count and drop 4th chunk if too high
    MAX_PROMPT_TOKENS = 3072  # Leave room for response within 4096 context
    token_count = len(state.llm.tokenize(full_prompt.encode('utf-8')))
    print(f"Initial prompt token count: {token_count}")
    if token_count > MAX_PROMPT_TOKENS and len(final_chunks) >= 4:
        print(f"⚠️ Token count ({token_count}) exceeds limit ({MAX_PROMPT_TOKENS}). Dropping 4th chunk...")
        final_chunks = final_chunks[:3]
        
        # Rebuild prompt with reduced chunks
        context_parts = []
        sources_used = set()
        for i, chunk in enumerate(final_chunks, 1):
            src = chunk.get("source", "EARIST Database")
            sources_used.add(src)
            context_parts.append(f"[Source {i}: {src}]\n{chunk['content']}")
        
        context_str = "\n\n".join(context_parts)
        sources_str = ", ".join(sources_used)
        
        system_prompt = f"""You are "AmangBot", a helpful AI student assistant for EARIST.

GUIDELINES:
1. Answer the question using ONLY the provided context.
2. CITATION IS MANDATORY: Start your answer with "According to [Source Name]...".
3. If the answer is not in the context, say "I don't know."
4. Be polite and concise.
5. Format with bullet points for lists.

Sources: {sources_str}
"""
        full_prompt = f"[INST] {system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {standalone_query} [/INST]"
        token_count = len(state.llm.tokenize(full_prompt.encode('utf-8')))
        print(f"✅ New token count after dropping chunk: {token_count}")

    # Print the final prompt
    print(f"\n{'='*50}")
    print(f"📝 FINAL PROMPT TO LLM")
    print(f"{'='*50}")
    print(full_prompt)
    print(f"{'='*50}\n")
    
    # Print retrieved chunks
    print(f"\n{'='*50}")
    print(f"📦 RETRIEVED CHUNKS (used for generation)")
    print(f"{'='*50}")
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Source: {chunk.get('source', 'N/A')}")
        print(f"  Category: {chunk.get('category', 'N/A')}")
        print(f"  Topic: {chunk.get('topic', 'N/A')}")
        print(f"  Content: {chunk['content'][:200]}...")
    print(f"{'='*50}\n")

    # 7. Generate Response (Non-blocking)
    gen_start = time.perf_counter()
    output = await async_llm_generate(
        full_prompt, 
        stop=["</s>", "[/INST]", "User:"], 
        temp=0.1
    )
    gen_time = time.perf_counter() - gen_start
    
    response_text = output["choices"][0]["text"].strip()
    
    # Calculate total time and print timing breakdown
    total_time = time.perf_counter() - total_start
    
    print(f"\n{'='*50}")
    print(f"⏱️  TIMING BREAKDOWN")
    print(f"{'='*50}")
    print(f"  Contextualization: {context_time*1000:.2f} ms")
    print(f"  Embedding:         {embed_time*1000:.2f} ms")
    print(f"  Retrieval (FAISS): {retrieval_time*1000:.2f} ms")
    print(f"  Reranking:         {rerank_time*1000:.2f} ms")
    print(f"  Generation (LLM):  {gen_time*1000:.2f} ms ({gen_time:.2f} s)")
    print(f"{'='*50}")
    print(f"  TOTAL:             {total_time*1000:.2f} ms ({total_time:.2f} s)")
    print(f"{'='*50}\n")

    # 8. Return
    chunks_data = [
        ChunkData(
            content=c["content"], 
            source=c["source"], 
            category=c.get("category", ""), 
            topic=c.get("topic", "")
        ) 
        for c in final_chunks
    ]
    
    # All retrieved chunks (before final selection)
    all_retrieved_chunks = [
        ChunkData(
            content=c["content"], 
            source=c["source"], 
            category=c.get("category", ""), 
            topic=c.get("topic", "")
        ) 
        for c in retrieved_chunks
    ]

    return ChatResponse(
        response=response_text,
        context=[c["content"] for c in final_chunks],
        chunks=chunks_data,
        prompt=full_prompt,
        retrieved_chunks=all_retrieved_chunks
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint that returns tokens as Server-Sent Events."""
    
    # 1. Fast fail for fillers
    if is_conversational_filler(request.message):
        async def filler_response():
            response = "Hello! I am AmangBot. How can I help you with your EARIST concerns?"
            # Send metadata first
            yield f"data: {json.dumps({'type': 'metadata', 'chunks': [], 'prompt': ''})}\n\n"
            # Stream the response word by word
            for word in response.split(' '):
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(filler_response(), media_type="text/event-stream")

    # 2. Contextualize (Non-blocking)
    standalone_query = await get_contextualized_query(request.history, request.message)

    # 3. Embedding (Non-blocking)
    query_text = f"search_query: {standalone_query}"
    query_embedding = await async_embed([query_text])

    # 4. Retrieval (Hybrid Strategy)
    initial_k = request.k * 3
    D, I = await async_search_index(query_embedding, initial_k)
    retrieved_chunks = [state.chunks[i] for i in I[0] if i < len(state.chunks)]

    final_chunks = []
    
    # 5. Reranking & Thresholding
    if state.reranker:
        pairs = [[standalone_query, chunk["content"]] for chunk in retrieved_chunks]
        scores = await async_rerank(pairs)
        
        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        best_score = scored_chunks[0][1] if scored_chunks else -999

        if best_score < RERANKER_THRESHOLD:
            # Build retrieved chunks data for frontend even on failure
            all_retrieved_chunks = [
                {"content": c["content"], "source": c["source"], "category": c.get("category", ""), "topic": c.get("topic", "")}
                for c in retrieved_chunks
            ]
            
            async def rejection_response():
                yield f"data: {json.dumps({'type': 'metadata', 'chunks': [], 'retrieved_chunks': all_retrieved_chunks, 'prompt': '(No prompt generated - query rejected due to low relevance score)'})}\n\n"
                response = "I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database."
                for word in response.split(' '):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(rejection_response(), media_type="text/event-stream")

        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        final_chunks = retrieved_chunks[:request.k]

    # 6. Construct Prompt
    context_parts = []
    sources_used = set()
    for i, chunk in enumerate(final_chunks, 1):
        src = chunk.get("source", "EARIST Database")
        sources_used.add(src)
        context_parts.append(f"[Source {i}: {src}]\n{chunk['content']}")
    
    context_str = "\n\n".join(context_parts)
    sources_str = ", ".join(sources_used)

    system_prompt = f"""You are "AmangBot", a helpful AI student assistant for EARIST.

GUIDELINES:
1. Answer the question using ONLY the provided context.
2. CITATION IS MANDATORY: Start your answer with "According to [Source Name]...".
3. If the answer is not in the context, say "I don't know."
4. Be polite and concise.
5. Format with bullet points for lists.

Sources: {sources_str}
"""
    full_prompt = f"[INST] {system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {standalone_query} [/INST]"

    # Check token count and drop 4th chunk if too high
    MAX_PROMPT_TOKENS = 3072
    token_count = len(state.llm.tokenize(full_prompt.encode('utf-8')))
    if token_count > MAX_PROMPT_TOKENS and len(final_chunks) >= 4:
        final_chunks = final_chunks[:3]
        
        context_parts = []
        sources_used = set()
        for i, chunk in enumerate(final_chunks, 1):
            src = chunk.get("source", "EARIST Database")
            sources_used.add(src)
            context_parts.append(f"[Source {i}: {src}]\n{chunk['content']}")
        
        context_str = "\n\n".join(context_parts)
        sources_str = ", ".join(sources_used)
        
        system_prompt = f"""You are "AmangBot", a helpful AI student assistant for EARIST.

GUIDELINES:
1. Answer the question using ONLY the provided context.
2. CITATION IS MANDATORY: Start your answer with "According to [Source Name]...".
3. If the answer is not in the context, say "I don't know."
4. Be polite and concise.
5. Format with bullet points for lists.

Sources: {sources_str}
"""
        full_prompt = f"[INST] {system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {standalone_query} [/INST]"

    # Prepare metadata
    chunks_data = [
        {"content": c["content"], "source": c["source"], "category": c.get("category", ""), "topic": c.get("topic", "")}
        for c in final_chunks
    ]
    all_retrieved_chunks = [
        {"content": c["content"], "source": c["source"], "category": c.get("category", ""), "topic": c.get("topic", "")}
        for c in retrieved_chunks
    ]

    async def stream_response():
        # Send metadata first
        metadata = {
            'type': 'metadata',
            'chunks': chunks_data,
            'retrieved_chunks': all_retrieved_chunks,
            'prompt': full_prompt
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Stream tokens from LLM
        async for token in async_llm_stream(
            full_prompt, 
            stop=["</s>", "[/INST]", "User:"], 
            temp=0.1
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)