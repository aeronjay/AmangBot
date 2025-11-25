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

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "../BART/Dataset/Latest"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))
NOMIC_MODEL_PATH = "../Models/nomic-finetuned/nomic-finetuned-final"

# Thresholds
SIMILARITY_THRESHOLD = 1.4  # FAISS L2 distance. Lower is better. Tune this! 
                            # If distance > 1.4, the context is likely irrelevant.

app = FastAPI(title="AmangBot Backend")

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
    
    conversation_str = ""
    for msg in history[-2:]: # Only look at last 2 turns to prevent confusion
        conversation_str += f"{msg.role}: {msg.content}\n"
        
    prompt = f"""<s>[INST] You are a query reformulator.
Task: Rewrite the "Follow Up Input" to be a standalone question based on the "Chat History".
Rules:
1. If the Follow Up Input refers to previous context (e.g., "what about for IT?", "how much is that?"), rewrite it to include the context.
2. If the Follow Up Input is a NEW topic unrelated to the history, DO NOT change it. Just output the input as is.
3. Output ONLY the rewritten question.

Chat History:
{conversation_str}
Follow Up Input: {latest_message}
Standalone question: [/INST]"""

    output = llm(prompt, max_tokens=64, stop=["\n", "</s>"], temperature=0.1, echo=False)
    new_query = output["choices"][0]["text"].strip()
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
        pairs = [[standalone_query, doc] for doc in retrieved_chunks]
        scores = reranker.predict(pairs)
        scored_chunks = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score in scored_chunks[:request.k]]
    else:
        final_chunks = retrieved_chunks[:request.k]

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