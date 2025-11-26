import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import uvicorn

# Import auth-related modules
from config.database import connect_to_mongo, close_mongo_connection
from routes.auth import router as auth_router

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "./Dataset/Integrated/Latest"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))
BART_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Models/my_bart_model/"))
NOMIC_MODEL_PATH = "../Models/nomic-finetuned/nomic-finetuned-final"

# Thresholds
SIMILARITY_THRESHOLD = 1.4  # FAISS L2 distance. Lower is better. Tune this! 
                            # If distance > 1.4, the context is likely irrelevant.

app = FastAPI(title="AmangBot Backend")

# Include auth router
app.include_router(auth_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
chunks = []  # List of dicts with 'content', 'source', 'category', 'topic'
index = None
embedder = None
llm = None
reranker = None

# --- Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    k: Optional[int] = 4  # Increased k slightly to get broader context for clarification

class ChatResponse(BaseModel):
    response: str
    context: List[str]
    is_clarification_needed: bool = False

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    global chunks, index, embedder, llm, reranker
    
    # Connect to MongoDB
    await connect_to_mongo()
    
    # 1. Load Chunks
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found at {DATASET_DIR}")
    
    chunks = []
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
                                # Store chunk with metadata for source attribution
                                chunk_data = {
                                    "content": item["content"],
                                    "source": item.get("source", "EARIST Database"),
                                    "category": item.get("category", ""),
                                    "topic": item.get("topic", "")
                                }
                                chunks.append(chunk_data)
                                
                                # Prepare embedding with metadata (excluding id and keywords)
                                metadata_parts = []
                                for k, v in item.items():
                                    if k not in ["id", "keywords", "content"]:
                                        metadata_parts.append(f"{k}: {v}")
                                metadata_str = ", ".join(metadata_parts)
                                
                                chunks_for_embedding.append(f"search_document: {metadata_str}. {item['content']}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(chunks)} chunks from {DATASET_DIR}.")

    # 2. Initialize Embedder
    print("Initializing Nomic Embedder...")
    embedder = SentenceTransformer(NOMIC_MODEL_PATH, trust_remote_code=True)

    # 3. Build FAISS Index
    print("Building FAISS index...")
    # chunks_for_embedding is already prepared
    embeddings = embedder.encode(chunks_for_embedding, convert_to_numpy=True, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 4. Initialize LLM
    print(f"Initializing model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_batch=4096,         
        n_ctx=4096,           
        verbose=False         
    )

    # 5. Initialize Reranker
    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Reranker initialized.")
    except Exception:
        reranker = None


# --- Shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# --- Helper Functions ---

def is_conversational_filler(text: str) -> bool:
    """Simple check to skip RAG for 'Hi', 'Thanks', etc."""
    fillers = ["hi", "hello", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye"]
    return text.lower().strip() in fillers

def needs_contextualization(message: str) -> bool:
    """
    Heuristic check to determine if a message likely needs context from history.
    Returns True if the message contains pronouns/references that need resolution.
    """
    # Lowercase for comparison
    msg_lower = message.lower().strip()
    
    # Reference words that indicate the message depends on previous context
    context_indicators = [
        "it", "that", "this", "they", "them", "those", "these",
        "the same", "similar", "also", "too", "as well",
        "what about", "how about", "and for", "for the",
        "more about", "tell me more", "explain more",
        "can you", "could you",  # Often follow-ups
    ]
    
    # Check if message starts with context indicators
    starts_with_indicators = [
        "what about", "how about", "and ", "but ", "also ",
        "what's", "how's", "where's", "who's",
        "the ", "its ", "their "
    ]
    
    # If message is very short (1-3 words), it likely needs context
    word_count = len(msg_lower.split())
    if word_count <= 3 and not any(q in msg_lower for q in ["who", "what", "where", "when", "why", "how"]):
        return True
    
    # Check for starts-with indicators
    for indicator in starts_with_indicators:
        if msg_lower.startswith(indicator):
            return True
    
    # Check for embedded context indicators (but be more careful)
    # Only trigger if the message seems incomplete
    if word_count < 8:
        for indicator in context_indicators:
            # Check if indicator appears as a standalone word
            if f" {indicator} " in f" {msg_lower} ":
                return True
    
    return False

def contextualize_query(history: List[Message], latest_message: str) -> str:
    """
    Rewrites the question to be standalone only when necessary.
    Uses heuristics first to avoid unnecessary LLM calls and incorrect rewrites.
    """
    if not history:
        return latest_message
    
    # First, check if contextualization is likely needed
    if not needs_contextualization(latest_message):
        print(f"Contextualization skipped (standalone query): '{latest_message}'")
        return latest_message
    
    # Get relevant history (last 2 exchanges)
    conversation_str = ""
    for msg in history[-4:]:  # Last 4 messages (2 exchanges)
        conversation_str += f"{msg.role}: {msg.content}\n"
        
    prompt = f"""[INST] You are a query reformulator for an EARIST university chatbot.

Task: Determine if the "User Question" needs context from the "Chat History" to be understood.

Rules:
1. If the User Question contains pronouns (it, that, this, they) or references (the same, similar) that refer to something in the Chat History, rewrite it as a complete standalone question.
2. If the User Question is already a complete, standalone question about a NEW topic, output it EXACTLY as-is without any changes.
3. Do NOT add information that wasn't implied by the user.
4. Output ONLY the final question, nothing else.

Examples:
- History: "What are the requirements for freshmen?" -> User: "what about transferees?" -> Output: "What are the admission requirements for transferees?"
- History: "Who is the president?" -> User: "What courses does EARIST offer?" -> Output: "What courses does EARIST offer?"
- History: "Tell me about BSIT" -> User: "how many years?" -> Output: "How many years is the BSIT program?"

Chat History:
{conversation_str}
User Question: {latest_message}
Output: [/INST]"""

    output = llm(prompt, max_tokens=100, stop=["\n", "</s>", "[/INST]"], temperature=0.0, echo=False)
    new_query = output["choices"][0]["text"].strip()
    
    # Fallback: if the LLM returns empty or just punctuation, use original
    if not new_query or len(new_query) < 3:
        new_query = latest_message
    
    # Remove any quotation marks the LLM might have added
    new_query = new_query.strip('"').strip("'")
    
    print(f"Contextualized: '{latest_message}' -> '{new_query}'")
    return new_query

# --- Core Endpoint ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global index, chunks, embedder, llm, reranker
    
    # 1. Check for simple greetings (Speed optimization)
    if is_conversational_filler(request.message):
        return ChatResponse(
            response="Hello there! I am Amang Bot. How can I help you with your EARIST concerns today?",
            context=[]
        )

    # 2. Contextualize
    standalone_query = contextualize_query(request.history, request.message)

    # 3. Search
    query_text = f"search_query: {standalone_query}"
    query_embedding = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Fetch candidates
    fetch_k = request.k * 3 if reranker else request.k
    D, I = index.search(query_embedding, fetch_k)
    
    # Check distance threshold (Guardrail for "Unknown Info")
    # FAISS L2: Lower is better. If the closest chunk is very far, we don't know the answer.
    # Note: exact threshold depends on your embeddings. Start with 1.3 or 1.4 for normalized vectors.
    if D[0][0] > SIMILARITY_THRESHOLD:
        return ChatResponse(
            response="I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database. I can only answer questions related to EARIST policies, staff, and guidelines.",
            context=[]
        )

    retrieved_chunks = [chunks[i] for i in I[0]]

    # 4. Rerank (Refining the search)
    if reranker:
        pairs = [[standalone_query, chunk["content"]] for chunk in retrieved_chunks]
        scores = reranker.predict(pairs)
        scored_chunks = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        final_chunks = retrieved_chunks[:request.k]

    # Build context string with source attribution for the LLM
    context_parts = []
    sources_used = set()
    for i, chunk in enumerate(final_chunks, 1):
        source = chunk.get("source", "EARIST Database")
        topic = chunk.get("topic", "")
        sources_used.add(source)
        source_label = f"[Source {i}: {source}" + (f" - {topic}]" if topic else "]")
        context_parts.append(f"{source_label}\n{chunk['content']}")
    
    context_str = "\n\n".join(context_parts)
    sources_list = list(sources_used)
    
    # 5. Construct Prompt (The "Persona" & "Clarification" Logic)
    sources_str = ", ".join(sources_list) if sources_list else "EARIST Database"
    
    system_prompt = f"""You are "Amang Bot", a friendly and helpful AI student assistant for EARIST (Eulogio "Amang" Rodriguez Institute of Science and Technology).

GUIDELINES:
1. **Scope:** Answer ONLY using the provided Context. If the answer is not there, say you don't know.

2. **Source Attribution (CRITICAL):**
   - You MUST cite your source in your response.
   - Start your answer with "According to the [Source Name],..." or "Based on the [Source Name],...".
   - The sources available are: {sources_str}

3. **Use Latest Information:** Prioritize the MOST RECENT information. If multiple versions exist, use only the latest.

4. **Tone:** Be polite, encouraging, and conversational.

5. **Ambiguity Handling:** If the question is vague, list the available options and ask the user to clarify.

6. **Formatting (IMPORTANT - Follow these rules strictly):**
   - Use **bold** for important terms, names, or key information.
   - Use bullet points (•) for lists with 3+ items.
   - Use numbered lists (1. 2. 3.) for sequential steps or procedures.
   - Add line breaks between different sections for readability.
   - For requirements or documents, format as a clean list.
   - Keep paragraphs short (2-3 sentences max).
   - Example format for requirements:
     
     **Documentary Requirements:**
     • Form 138 (Report Card)
     • Certificate of Good Moral Character
     • PSA Birth Certificate
     
   - Example format for steps:
     
     **Steps to Enroll:**
     1. Submit your application online.
     2. Take the entrance exam.
     3. Attend the interview.

7. **No Hallucination:** If the context does not contain the answer, say "I'm sorry, but I couldn't find specific information about that in my sources."

Remember: Answer Based ONLY on the provided Sources, cite your source, and format your response for easy reading.
"""

    prompt_content = f"""{system_prompt}

Context from Sources:
{context_str}

User Question: {standalone_query}

Amang Bot Response:"""

    prompt = f"[INST] {prompt_content} [/INST]"

    # 6. Generate
    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.2, # Keep temperature low to reduce hallucinations, but >0 for natural flow
        stop=["</s>", "[/INST]", "User:"],
        echo=False
    )
    
    response_text = output["choices"][0]["text"].strip()
    
    # Extract content strings for the response context
    context_contents = [chunk["content"] for chunk in final_chunks]
    
    return ChatResponse(
        response=response_text,
        context=context_contents
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)