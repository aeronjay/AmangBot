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
                            # If distance > 1.4x`x`, the context is likely irrelevant.

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
chunks = []
index = None
embedder = None
llm = None

# --- Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    k: Optional[int] = 3  # Number of top chunks to return after reranking

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
                                chunks.append(item["content"])
                                
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
    embedder = SentenceTransformer(NOMIC_MODEL_PATH, trust_remote_code=True, device='cuda')

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
        n_gpu_layers=999, 
        n_batch=512,         
        n_ctx=2048,           
        verbose=False         
    )


# --- Shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# --- Helper Functions ---

def is_conversational_filler(text: str) -> bool:
    """Simple check to skip RAG for 'Hi', 'Thanks', etc."""
    fillers = ["hi", "hello", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye"]
    return text.lower().strip() in fillers

def contextualize_query(history: List[Message], latest_message: str) -> str:
    """
    Rewrites the question to be standalone.
    Crucial change: Added instruction to detect topic shifts.
    """
    if not history:
        return latest_message
    
    # Filter to only user/assistant messages and get last 3 turns for better context
    conversation_str = ""
    relevant_history = [msg for msg in history if msg.role in ["user", "assistant"]][-3:]
    for msg in relevant_history:
        role_label = "User" if msg.role == "user" else "Assistant"
        conversation_str += f"{role_label}: {msg.content}\n"
        
    prompt = f"""<s>[INST] You are a query reformulator for an EARIST university chatbot.

TASK: Rewrite the "Follow Up Input" to be a clear, standalone question.

RULES:
1. REFERENCE RESOLUTION: If the Follow Up Input uses pronouns or references (e.g., "what about for IT?", "how much is that?", "what are its requirements?"), resolve them using the Chat History context.
2. TOPIC SHIFT: If the Follow Up Input is a completely NEW topic unrelated to the history, return it unchanged.
3. CONTEXT PRESERVATION: Include relevant details from history (department names, specific programs, etc.) when rewriting.
4. OUTPUT: Return ONLY the rewritten question, nothing else.

EXAMPLES:
- History mentions "BSIT program" + Follow up "what are the requirements?" → "What are the requirements for the BSIT program?"
- History mentions "enrollment fees" + Follow up "how about for graduate students?" → "What are the enrollment fees for graduate students?"
- New topic "Who is the president?" → "Who is the president?" (unchanged)

Chat History:
{conversation_str}
Follow Up Input: {latest_message}
Standalone question: [/INST]"""

    output = llm(prompt, max_tokens=128, stop=["</s>", "\n\n"], temperature=0.1, echo=False)
    new_query = output["choices"][0]["text"].strip()
    
    # Fallback: if the output is empty or too short, use original
    if len(new_query) < 3:
        new_query = latest_message
        
    print(f"Contextualized: '{latest_message}' -> '{new_query}'")
    return new_query

# --- Core Endpoint ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global index, chunks, embedder, llm
    
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
    
    D, I = index.search(query_embedding, request.k)
    
    # Check distance threshold (Guardrail for "Unknown Info")
    # FAISS L2: Lower is better. If the closest chunk is very far, we don't know the answer.
    # Note: exact threshold depends on your embeddings. Start with 1.3 or 1.4 for normalized vectors.
    if D[0][0] > SIMILARITY_THRESHOLD:
        return ChatResponse(
            response="I'm sorry, but I couldn't find specific information about that in the EARIST handbook or my database. I can only answer questions related to EARIST policies, staff, and guidelines.",
            context=[]
        )

    final_chunks = [chunks[i] for i in I[0]]

    context_str = "\n\n".join(final_chunks)
    
    # 5. Construct Prompt (The "Persona" & "Clarification" Logic)
    system_prompt = """You are "Amang Bot", a friendly and helpful AI student assistant for EARIST (Eulogio "Amang" Rodriguez Institute of Science and Technology).

GUIDELINES:
1. **Scope:** Answer ONLY using the provided Context. If the answer is not there, say you don't know.
2. **Tone:** Be polite, encouraging, and conversational. You can use "we" when referring to the school community.
3. **Ambiguity Handling (CRITICAL):** - If the user asks a vague question (e.g., "What are the requirements?" or "Who is the head?"), CHECK THE CONTEXT.
   - If the context contains requirements for multiple things (e.g., "Admission" AND "Graduation"), DO NOT guess.
   - Instead, list the options you see and ask the user to clarify. (e.g., "I see requirements for both Freshmen and Transferees. Which one would you like to know about?")
4. **Formatting:** Use bullet points for lists to make it readable.
5. **No Hallucination:** If the context does not contain the answer, respond with "I'm sorry, but I couldn't find specific information about that in my Sources.."

If the context does not contain the specific contact details requested, apologize and suggest they visit the EARIST registrar personally.

Remember: Your Goal is to Answer The Question Based ONLY On the provided Sources.
"""

    prompt_content = f"""{system_prompt}

Context from Sources:
{context_str}

User Question: {standalone_query}

Amang Bot Response:"""

    prompt = f"<s>[INST] {prompt_content} [/INST]"

    # 6. Generate
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.1, # Keep temperature low to reduce hallucinations, but >0 for natural flow
        stop=["</s>", "[/INST]", "User:"],
        echo=False
    )
    
    response_text = output["choices"][0]["text"].strip()
    
    return ChatResponse(
        response=response_text,
        context=final_chunks
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)