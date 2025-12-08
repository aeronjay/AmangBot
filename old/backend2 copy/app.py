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
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../Dataset/300TokenDataset"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))
NOMIC_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../Models/nomic-finetuned/nomic-finetuned-final"))

# --- Embeddings Cache Paths ---
EMBEDDINGS_CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, "embeddings_cache"))
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, "faiss_index.bin")
CHUNKS_CACHE_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, "chunks.json")

# --- Tuning Parameters ---
# MS-MARCO CrossEncoder outputs raw logits (not 0-1 probability).
# > 0 is usually relevant. < -4 is usually irrelevant.
# We set a threshold: if the top chunk scores lower than this, we don't know the answer.
RERANKER_THRESHOLD = -5
MAX_PROMPT_TOKENS = 4500  

# --- System Prompt Template ---
SYSTEM_PROMPT_TEMPLATE = """You are "AmangBot", a friendly and knowledgeable AI student assistant for EARIST (Eulogio "Amang" Rodriguez Institute of Science and Technology).

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

Sources Available: {sources_str}
"""

# --- Globals State Storage ---
class GlobalState:
    chunks: List[dict] = []
    index = None
    embedder: Optional[SentenceTransformer] = None
    llm: Optional[Llama] = None
    reranker: Optional[CrossEncoder] = None

state = GlobalState()

# --- Embedding Cache Functions ---
def save_embeddings_cache(index, chunks: List[dict]):
    """Save FAISS index and chunks to disk for faster startup   ."""
    os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    
    # Save chunks
    with open(CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to {CHUNKS_CACHE_PATH}")

def load_embeddings_cache():
    """Load FAISS index and chunks from disk. Returns (index, chunks) or (None, None) if not found."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_CACHE_PATH):
        return None, None
    
    try:
        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"FAISS index loaded from {FAISS_INDEX_PATH} ({index.ntotal} vectors)")
        
        # Load chunks
        with open(CHUNKS_CACHE_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunks loaded from {CHUNKS_CACHE_PATH} ({len(chunks)} chunks)")
        
        return index, chunks
    except Exception as e:
        print(f"Error loading embeddings cache: {e}")
        return None, None

# --- Uncomment this function and call it to re-embed the dataset ---
# async def rebuild_embeddings():
#     """
#     Force rebuild embeddings from the dataset.
#     Call this function when you update the dataset and need to re-embed.
#     """
#     print("Rebuilding embeddings from dataset...")
#     
#     if not os.path.exists(DATASET_DIR):
#         print("Error: Dataset directory not found.")
#         return False
#     
#     loaded_chunks = []
#     chunks_for_embedding = []
#     
#     for filename in os.listdir(DATASET_DIR):
#         if filename.endswith(".json"):
#             file_path = os.path.join(DATASET_DIR, filename)
#             try:
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     file_data = json.load(f)
#                     if isinstance(file_data, list):
#                         for item in file_data:
#                             if "content" in item:
#                                 chunk_data = {
#                                     "content": item["content"],
#                                     "source": item.get("source", "EARIST Database"),
#                                     "category": item.get("category", ""),
#                                     "topic": item.get("topic", "")
#                                 }
#                                 loaded_chunks.append(chunk_data)
#                                 
#                                 metadata_parts = [f"{k}: {v}" for k, v in item.items() if k not in ["id", "keywords", "content"]]
#                                 metadata_str = ", ".join(metadata_parts)
#                                 chunks_for_embedding.append(f"search_document: {metadata_str}. {item['content']}")
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
#     
#     if not chunks_for_embedding:
#         print("Error: No chunks to embed.")
#         return False
#     
#     print(f"Embedding {len(chunks_for_embedding)} chunks...")
#     embeddings = state.embedder.encode(
#         chunks_for_embedding, 
#         convert_to_numpy=True, 
#         normalize_embeddings=True
#     )
#     
#     dimension = embeddings.shape[1]
#     state.index = faiss.IndexFlatL2(dimension)
#     state.index.add(embeddings)
#     state.chunks = loaded_chunks
#     
#     # Save to cache
#     save_embeddings_cache(state.index, state.chunks)
#     
#     print(f"Embeddings rebuilt and saved! {state.index.ntotal} vectors.")
#     return True

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
    
    # 2. Load Embedder (needed for both cached and fresh embeddings)
    print("Initializing Nomic Embedder...")
    state.embedder = SentenceTransformer(NOMIC_MODEL_PATH, trust_remote_code=True, device='cpu')
    
    # 3. Try to load embeddings from cache
    cached_index, cached_chunks = load_embeddings_cache()
    
    if cached_index is not None and cached_chunks is not None:
        # Use cached embeddings
        state.index = cached_index
        state.chunks = cached_chunks
        print("Using cached embeddings - skipping embedding generation.")
    else:
        # Build embeddings from scratch
        print(f"No cache found. Loading chunks from {DATASET_DIR}...")
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
            
            # Build FAISS index
            if chunks_for_embedding:
                print("Building FAISS index...")
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
                
                # Save to cache for next startup
                save_embeddings_cache(state.index, state.chunks)
        else:
            print("Warning: Dataset directory not found.")

    # 4. Load Reranker
    try:
        print("Loading Reranker...")
        state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    except Exception as e:
        print(f"Failed to load Reranker: {e}")
        state.reranker = None

    # 5. Load LLM
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
    # Check for pronouns and follow-up indicators
    pronoun_triggers = ["it", "that", "this", "they", "them", "he", "she", "those", "these", "its", "their"]
    follow_up_triggers = ["similar", "also", "and", "more", "else", "another", "other", "same", "too", 
                          "about that", "about this", "about it", "what about", "how about", 
                          "tell me more", "any other", "anything else", "related"]
    
    word_count = len(msg_lower.split())
    
    # Short queries often need context
    if word_count < 4 and not any(w in msg_lower for w in ["who", "what", "where", "when", "why", "how"]):
        return True
    
    # Check for pronoun triggers
    if any(f" {t} " in f" {msg_lower} " or msg_lower.startswith(f"{t} ") or msg_lower.endswith(f" {t}") for t in pronoun_triggers):
        return True
    
    # Check for follow-up phrase triggers
    if any(trigger in msg_lower for trigger in follow_up_triggers):
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

async def async_llm_generate(prompt: str, max_tokens=1600, stop=None, temp=0.1):
    """Non-blocking wrapper for LlamaCPP generation."""
    return await run_in_threadpool(
        state.llm,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temp,
        stop=stop or [],
        echo=False
    )

async def async_llm_stream(prompt: str, max_tokens=1600, stop=None, temp=0.1) -> AsyncGenerator[str, None]:
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

# --- Helper Functions for Prompt Building ---

def build_context_from_chunks(chunks: List[dict]) -> tuple[str, set]:
    """Build context string and collect sources from chunks."""
    context_parts = []
    sources_used = set()
    
    for i, chunk in enumerate(chunks, 1):
        src = chunk.get("source", "EARIST Database")
        category = chunk.get("category", "")
        topic = chunk.get("topic", "")
        sources_used.add(src)
        
        # Build metadata header for this chunk
        metadata_lines = [f"[Source {i}: {src}]"]
        if category:
            metadata_lines.append(f"Category: {category}")
        if topic:
            metadata_lines.append(f"Topic: {topic}")
        metadata_header = "\n".join(metadata_lines)
        
        context_parts.append(f"{metadata_header}\n{chunk['content']}")
    
    context_str = "\n\n".join(context_parts)
    return context_str, sources_used

def build_full_prompt(chunks: List[dict], query: str) -> str:
    """Build the full prompt for the LLM."""
    context_str, sources_used = build_context_from_chunks(chunks)
    sources_str = ", ".join(sources_used)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(sources_str=sources_str)
    return f"[INST] {system_prompt}\n\nContext from EARIST Sources:\n{context_str}\n\nStudent Question: {query} [/INST]"

def optimize_chunks_for_token_limit(chunks: List[dict], query: str, max_tokens: int = MAX_PROMPT_TOKENS) -> List[dict]:
    """Drop chunks if prompt exceeds token limit."""
    full_prompt = build_full_prompt(chunks, query)
    token_count = len(state.llm.tokenize(full_prompt.encode('utf-8')))
    print(f"Initial prompt token count: {token_count}")
    
    if token_count > max_tokens and len(chunks) >= 4:
        print(f"⚠️ Token count ({token_count}) exceeds limit ({max_tokens}). Dropping 4th chunk...")
        chunks = chunks[:3]
        new_prompt = build_full_prompt(chunks, query)
        new_token_count = len(state.llm.tokenize(new_prompt.encode('utf-8')))
        print(f"✅ New token count after dropping chunk: {new_token_count}")
    
    return chunks

def chunks_to_chunk_data(chunks: List[dict]) -> List[ChunkData]:
    """Convert chunk dicts to ChunkData objects."""
    return [
        ChunkData(
            content=c["content"],
            source=c["source"],
            category=c.get("category", ""),
            topic=c.get("topic", "")
        )
        for c in chunks
    ]

def chunks_to_dict_list(chunks: List[dict]) -> List[dict]:
    """Convert chunks to a list of dicts for JSON serialization."""
    return [
        {"content": c["content"], "source": c["source"], "category": c.get("category", ""), "topic": c.get("topic", "")}
        for c in chunks
    ]

def print_retrieved_chunks(chunks: List[dict], header: str = "RETRIEVED CHUNKS"):
    """Print retrieved chunks for debugging."""
    print(f"\n{'='*50}")
    print(f"📦 {header}")
    print(f"{'='*50}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Source: {chunk.get('source', 'N/A')}")
        print(f"  Category: {chunk.get('category', 'N/A')}")
        print(f"  Topic: {chunk.get('topic', 'N/A')}")
        print(f"  Content: {chunk['content'][:200]}...")
    print(f"{'='*50}\n")

def print_timing_breakdown(context_time: float, embed_time: float, retrieval_time: float, 
                           rerank_time: float, gen_time: Optional[float], total_time: float):
    """Print timing breakdown for debugging."""
    print(f"\n{'='*50}")
    print(f"⏱️  TIMING BREAKDOWN")
    print(f"{'='*50}")
    print(f"  Contextualization: {context_time*1000:.2f} ms")
    print(f"  Embedding:         {embed_time*1000:.2f} ms")
    print(f"  Retrieval (FAISS): {retrieval_time*1000:.2f} ms")
    print(f"  Reranking:         {rerank_time*1000:.2f} ms")
    if gen_time is not None:
        print(f"  Generation (LLM):  {gen_time*1000:.2f} ms ({gen_time:.2f} s)")
    else:
        print(f"  Generation:        N/A (query rejected)")
    print(f"{'='*50}")
    print(f"  TOTAL:             {total_time*1000:.2f} ms ({total_time:.2f} s)")
    print(f"{'='*50}\n")

async def retrieve_and_rerank(query: str, k: int) -> tuple[List[dict], List[dict], Optional[float]]:
    """
    Retrieve chunks and rerank them. Returns (final_chunks, all_retrieved_chunks, best_score).
    best_score is None if reranker is not available.
    """
    # Embedding
    query_text = f"search_query: {query}"
    query_embedding = await async_embed([query_text])
    
    # Retrieval - Fetch 3x candidates to allow Reranker to filter bad ones
    initial_k = k * 3
    D, I = await async_search_index(query_embedding, initial_k)
    retrieved_chunks = [state.chunks[i] for i in I[0] if i < len(state.chunks)]
    
    final_chunks = []
    best_score = None
    
    # Reranking
    if state.reranker:
        pairs = [[query, chunk["content"]] for chunk in retrieved_chunks]
        scores = await async_rerank(pairs)
        
        print(f"\n--- QUERY: {query} ---")
        for doc, score in zip(retrieved_chunks, scores):
            print(f"Score: {score:.4f} | Text: {doc['content'][:50]}...")
        
        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        best_score = scored_chunks[0][1] if scored_chunks else -999
        print(f"Top Reranker Score: {best_score}")
        
        final_chunks = [chunk for chunk, score in scored_chunks[:k]]
    else:
        final_chunks = retrieved_chunks[:k]
    
    return final_chunks, retrieved_chunks, best_score

# --- Core Logic ---

async def get_contextualized_query(history: List[Message], latest_message: str) -> str:
    """Refined prompt to handle topic switching and follow-up questions better."""
    if not history or not needs_contextualization(latest_message):
        return latest_message

    conversation_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-6:]])
    
    prompt = f"""[INST] You are a query reformulator for an EARIST university chatbot.
    
Task: Rewrite the "User Question" to be a complete, standalone question based on the "Chat History".

Rules:
1. If the User Question is clearly a NEW TOPIC unrelated to the history, output it EXACTLY AS IS.
2. If the User Question is a FOLLOW-UP (uses pronouns like "it", "that", "they", "this", "those", "more", "else", "also", or asks for more details), rewrite it to include the full context from the previous conversation.
3. Include relevant details from the conversation that help clarify what the user is asking about.
4. Output ONLY the reformulated question. No explanations, no quotes.

Examples:
- History: "User: What are the requirements for enrollment?" / Follow-up: "What about for transferees?" -> "What are the enrollment requirements for transferee students?"
- History: "User: Who is the president of EARIST?" / Follow-up: "When did they start?" -> "When did the president of EARIST start their term?"
- History: "User: Tell me about scholarships" / Follow-up: "How do I apply?" -> "How do I apply for scholarships at EARIST?"

Chat History:
{conversation_str}

User Question: {latest_message}
Reformulated Question: [/INST]"""

    # Low max_tokens for speed, temp 0 for determinism
    output = await async_llm_generate(prompt, max_tokens=80, stop=["\n", "[/INST]"], temp=0.0)
    new_query = output["choices"][0]["text"].strip().strip('"').strip()
    
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

    # 3-5. Embedding, Retrieval, and Reranking
    embed_start = time.perf_counter()
    query_text = f"search_query: {standalone_query}"
    query_embedding = await async_embed([query_text])
    embed_time = time.perf_counter() - embed_start

    retrieval_start = time.perf_counter()
    initial_k = request.k * 3
    D, I = await async_search_index(query_embedding, initial_k)
    retrieved_chunks = [state.chunks[i] for i in I[0] if i < len(state.chunks)]
    retrieval_time = time.perf_counter() - retrieval_start

    rerank_start = time.perf_counter()
    final_chunks = []
    
    if state.reranker:
        pairs = [[standalone_query, chunk["content"]] for chunk in retrieved_chunks]
        scores = await async_rerank(pairs)
        
        print(f"\n--- QUERY: {standalone_query} ---")
        for doc, score in zip(retrieved_chunks, scores):
            print(f"Score: {score:.4f} | Text: {doc['content'][:50]}...")

        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        best_score = scored_chunks[0][1] if scored_chunks else -999
        print(f"Top Reranker Score: {best_score}")

        if best_score < RERANKER_THRESHOLD:
            rerank_time = time.perf_counter() - rerank_start
            total_time = time.perf_counter() - total_start
            
            print_timing_breakdown(context_time, embed_time, retrieval_time, rerank_time, None, total_time)
            print_retrieved_chunks(retrieved_chunks, "RETRIEVED CHUNKS (Query rejected - below threshold)")
            
            return ChatResponse(
                response="I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database.",
                context=[],
                chunks=[],
                prompt="(No prompt generated - query rejected due to low relevance score)",
                retrieved_chunks=chunks_to_chunk_data(retrieved_chunks)
            )

        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        final_chunks = retrieved_chunks[:request.k]
    rerank_time = time.perf_counter() - rerank_start

    # 6. Optimize chunks for token limit and build prompt
    final_chunks = optimize_chunks_for_token_limit(final_chunks, standalone_query)
    full_prompt = build_full_prompt(final_chunks, standalone_query)

    # Print the final prompt
    print(f"\n{'='*50}")
    print(f"📝 FINAL PROMPT TO LLM")
    print(f"{'='*50}")
    print(full_prompt)
    print(f"{'='*50}\n")
    
    print_retrieved_chunks(final_chunks, "RETRIEVED CHUNKS (used for generation)")

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
    print_timing_breakdown(context_time, embed_time, retrieval_time, rerank_time, gen_time, total_time)

    # 8. Return
    return ChatResponse(
        response=response_text,
        context=[c["content"] for c in final_chunks],
        chunks=chunks_to_chunk_data(final_chunks),
        prompt=full_prompt,
        retrieved_chunks=chunks_to_chunk_data(retrieved_chunks)
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint that returns tokens as Server-Sent Events."""
    
    # 1. Fast fail for fillers
    if is_conversational_filler(request.message):
        async def filler_response():
            response = "Hello! I am AmangBot. How can I help you with your EARIST concerns?"
            yield f"data: {json.dumps({'type': 'metadata', 'chunks': [], 'prompt': ''})}\n\n"
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

    # 4. Retrieval
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
            async def rejection_response():
                yield f"data: {json.dumps({'type': 'metadata', 'chunks': [], 'retrieved_chunks': chunks_to_dict_list(retrieved_chunks), 'prompt': '(No prompt generated - query rejected due to low relevance score)'})}\n\n"
                response = "I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database."
                for word in response.split(' '):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(rejection_response(), media_type="text/event-stream")

        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        final_chunks = retrieved_chunks[:request.k]

    # 6. Optimize chunks for token limit and build prompt
    final_chunks = optimize_chunks_for_token_limit(final_chunks, standalone_query)
    full_prompt = build_full_prompt(final_chunks, standalone_query)

    async def stream_response():
        # Send metadata first
        metadata = {
            'type': 'metadata',
            'chunks': chunks_to_dict_list(final_chunks),
            'retrieved_chunks': chunks_to_dict_list(retrieved_chunks),
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