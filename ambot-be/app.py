import os
import json
import glob
import time
import asyncio
import numpy as np
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
from rank_bm25 import BM25Okapi
from database import get_database, db
from auth_utils import verify_password, get_password_hash, create_access_token, decode_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import datetime, timedelta
import PyPDF2
import io
import shutil
from bson import ObjectId
import re

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

BART_MODEL = os.path.join(PROJECT_ROOT, "Models/finetuned-BART")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset/Default AMBOT Knowledge Base")
EMBEDDING_MODEL_NAME = os.path.join(PROJECT_ROOT, "Models/nomic-finetuned/nomic-finetuned-final")

INDEX_FILE = "faiss_index_finetuned.bin"
BM25_INDEX_FILE = "bm25_index.pkl"
METADATA_FILE = "chunks_metadata.json"
DISABLED_DATASETS_FILE = "disabled_datasets.json"

RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RELEVANCE_THRESHOLD = -2  # Threshold for query relevance (adjust as needed)
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

def get_disabled_datasets():
    if os.path.exists(DISABLED_DATASETS_FILE):
        try:
            with open(DISABLED_DATASETS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_disabled_datasets(disabled_list):
    with open(DISABLED_DATASETS_FILE, 'w') as f:
        json.dump(disabled_list, f)

def load_resources():
    global llm, embedder, index, chunks_metadata, reranker_model, reranker_tokenizer, bm25
    
    # Cleanup existing resources if any
    if llm is not None:
        del llm
        import gc
        gc.collect()
        llm = None

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
    
    disabled_files = get_disabled_datasets()
    # Normalize disabled files for comparison
    disabled_files_norm = [d.replace("\\", "/") for d in disabled_files]

    # Load all JSON files
    json_files = glob.glob(os.path.join(DATASET_PATH, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files.")
    
    for file_path in json_files:
        # Check if file is disabled
        rel_path = os.path.relpath(file_path, DATASET_PATH)
        rel_path_norm = rel_path.replace("\\", "/")
        
        if rel_path_norm in disabled_files_norm:
            print(f"Skipping disabled file: {rel_path}")
            continue

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
    
    # Create default admin if not exists
    try:
        admin = await db.users.find_one({"email": "admin@ambot.com"})
        if not admin:
            hashed_pw = get_password_hash("admin123")
            await db.users.insert_one({
                "email": "admin@ambot.com",
                "hashed_password": hashed_pw,
                "is_admin": True,
                "is_active": True,
                "username": "Admin",
                "created_at": datetime.utcnow()
            })
            print("Default admin created: admin@ambot.com / admin123")
    except Exception as e:
        print(f"Error creating default admin: {e}")

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

# Auth Setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = await db.users.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user = Depends(get_current_active_user)):
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Not authorized")
    return current_user

# Routers
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

# ... Auth Endpoints ...
@auth_router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.users.find_one({
        "$or": [
            {"email": form_data.username},
            {"username": form_data.username}
        ]
    })
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/me")
async def read_users_me(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "username": current_user.get("username", ""),
        "is_active": current_user.get("is_active", True),
        "is_admin": current_user.get("is_admin", False),
        "created_at": current_user.get("created_at", "")
    }

@auth_router.post("/verify-token")
async def verify_token(current_user = Depends(get_current_user)):
    return {
        "valid": True,
        "user_id": str(current_user["_id"]),
        "email": current_user["email"],
        "is_admin": current_user.get("is_admin", False)
    }

@auth_router.post("/logout")
async def logout():
    return {"message": "Logged out"}

# ... Admin Endpoints ...
@admin_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    source: str = Form(...),
    category: str = Form(...),
    topic: str = Form(...),
    current_user = Depends(get_current_admin_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    content = await file.read()
    
    # Save to MongoDB
    file_doc = {
        "filename": file.filename,
        "source": source,
        "category": category,
        "topic": topic,
        "content": content, # Binary
        "uploaded_at": datetime.utcnow(),
        "uploaded_by": current_user["email"],
        "size": len(content)
    }
    result = await db.files.insert_one(file_doc)
    
    # Process PDF
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        
        # Clean text to fix newlines in words and formatting
        # 1. Remove hyphens at line endings (e.g. "process-\ning" -> "processing")
        text = re.sub(r'-\n', '', text)
        # 2. Replace newlines with spaces
        text = text.replace('\n', ' ')
        # 3. Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    
    # Chunking (Increased to 1200 tokens, 200 overlap)
    chunks = []
    CHUNK_SIZE = 1200
    OVERLAP = 200
    STRIDE = CHUNK_SIZE - OVERLAP
    
    if llm:
        try:
            tokens = llm.tokenize(text.encode('utf-8'))
            token_chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + CHUNK_SIZE, len(tokens))
                chunk_tokens = tokens[start:end]
                token_chunks.append(chunk_tokens)
                if end == len(tokens):
                    break
                start += STRIDE
                
            chunks = [llm.detokenize(c).decode('utf-8', errors='ignore') for c in token_chunks]
        except Exception as e:
            print(f"Tokenization error: {e}. Falling back to char split.")
            # Fallback: ~4 chars per token. 1200 tokens ~= 4800 chars.
            CHAR_CHUNK_SIZE = 4800
            CHAR_STRIDE = 4000
            chunks = [text[i:i+CHAR_CHUNK_SIZE] for i in range(0, len(text), CHAR_STRIDE)]
    else:
        # Fallback approximation
        CHAR_CHUNK_SIZE = 4800
        CHAR_STRIDE = 4000
        chunks = [text[i:i+CHAR_CHUNK_SIZE] for i in range(0, len(text), CHAR_STRIDE)]

    # Save chunks to JSON
    chunk_data = []
    for i, chunk_text in enumerate(chunks):
        chunk_data.append({
            "source": source,
            "category": category,
            "topic": topic,
            "content": chunk_text
        })
        
    json_path = os.path.join(DATASET_PATH, f"{file.filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
    return {"message": "File uploaded and processed", "id": str(result.inserted_id)}

@admin_router.get("/files")
async def list_files(current_user = Depends(get_current_admin_user)):
    files = []
    cursor = db.files.find({}, {"content": 0}) # Exclude content
    async for doc in cursor:
        files.append({
            "id": str(doc["_id"]),
            "name": doc["filename"],
            "source": doc.get("source", "AdminDB"),
            "size": f"{doc['size'] / 1024:.1f} KB",
            "lastModified": doc["uploaded_at"].isoformat().split('T')[0],
            "status": "indexed"
        })
    return files

@admin_router.delete("/files/{file_id}")
async def delete_file(file_id: str, current_user = Depends(get_current_admin_user)):
    # Get filename
    try:
        doc = await db.files.find_one({"_id": ObjectId(file_id)})
    except:
        raise HTTPException(status_code=404, detail="Invalid ID")
        
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Delete from MongoDB
    await db.files.delete_one({"_id": ObjectId(file_id)})
    
    # Delete JSON file
    json_path = os.path.join(DATASET_PATH, f"{doc['filename']}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
        
    return {"message": "File deleted"}

@admin_router.post("/restart")
async def restart_system(current_user = Depends(get_current_admin_user)):
    # Delete existing index files to force re-indexing
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
    if os.path.exists(BM25_INDEX_FILE):
        os.remove(BM25_INDEX_FILE)
        
    load_resources()
    return {"message": "System resources reloaded and index updated"}

class ToggleKBRequest(BaseModel):
    filename: str
    enabled: bool

@admin_router.get("/knowledge-base")
async def list_knowledge_base(current_user = Depends(get_current_admin_user)):
    json_files = glob.glob(os.path.join(DATASET_PATH, "**/*.json"), recursive=True)
    disabled_files = get_disabled_datasets()
    # Normalize disabled files for comparison
    disabled_files_norm = [d.replace("\\", "/") for d in disabled_files]
    
    kb_files = []
    for file_path in json_files:
        rel_path = os.path.relpath(file_path, DATASET_PATH)
        rel_path_norm = rel_path.replace("\\", "/")
        
        kb_files.append({
            "filename": rel_path_norm,
            "enabled": rel_path_norm not in disabled_files_norm
        })
    return kb_files

@admin_router.post("/knowledge-base/toggle")
async def toggle_knowledge_base(request: ToggleKBRequest, current_user = Depends(get_current_admin_user)):
    disabled_files = get_disabled_datasets()
    # Normalize disabled files
    disabled_files = [d.replace("\\", "/") for d in disabled_files]
    
    filename_norm = request.filename.replace("\\", "/")
    
    if request.enabled:
        if filename_norm in disabled_files:
            disabled_files.remove(filename_norm)
    else:
        if filename_norm not in disabled_files:
            disabled_files.append(filename_norm)
            
    save_disabled_datasets(disabled_files)
    return {"message": f"File {'enabled' if request.enabled else 'disabled'}", "filename": request.filename}

app.include_router(auth_router)
app.include_router(admin_router)

def construct_prompt(query: str, chunks: List[dict]) -> str:
    context_text = ""
    for i, chunk in enumerate(reversed(chunks)):
        context_text += f"[Source {i+1}: {chunk.get('source', 'Unknown')}]\n"
        context_text += f"Category: {chunk.get('category', 'Unknown')}\n"
        context_text += f"Topic: {chunk.get('topic', 'Unknown')}\n"
        context_text += f"{chunk.get('content', '')}\n\n"

    prompt = f"""[INST] You are "AmangBot," a helpful senior student assistant at EARIST. Your goal is to guide freshmen by explaining policies thoroughly, warmly, and clearly.

### YOUR GUIDE ON HOW TO ANSWER:

1.  **Speak in the First Person ("I"):**
    * **BAD:** "According to the provided context..." or "The document states..."
    * **GOOD:** "Based on my data from the **[Insert Source Name Here]**,..." or "I checked the **[Insert Source Name Here]**, and here is what I found..."
    * Always extract the specific *Source Name* (e.g., Student Handbook 2021) from the text below.

2.  **Be Expansive & Proactive (Don't just answer—Explain):**
    * Do not give short, one-sentence answers.
    * **Expand on the topic:** If the student asks about "Failing," do not just define it. Look at the text and explain the *consequences* (like warnings) or the *process* to fix it.
    * **Connect the dots:** Treat the provided text as a whole concept. Explain the rules like you are teaching them to a friend.

3.  **Formatting Matters:**
    * Use **Bold** for emphasis.
    * Use Bullet points for lists/steps.
    * Use a warm, encouraging tone.

4.  **If You Don't Know:**
    * If the specific answer is NOT in the text below, say: "I don't have that specific information in my current data."
    * **Then guide them:**
        * For enrollment/grades -> Point them to the **Registrar's Office** or their **College's Official Facebook Page**.
        * For conduct/orgs -> Point them to the **Office of Student Affairs (OSAS)**.

---
### CONTEXT DATA:
{context_text}

### STUDENT QUESTION:
{query}

### YOUR RESPONSE (As a Senior Student):
[/INST]"""
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
    top_score = -float('inf')

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
        
        if len(sorted_indices) > 0:
            top_score = scores[sorted_indices[0]].item()
        
        print(f"Query: '{query}' | Top Score: {top_score}")

        # Select top k
        k_final = 5
        top_indices = sorted_indices[:k_final].tolist()
        
        retrieved_chunks = [initial_chunks[i] for i in top_indices]
            
    # Guardrail: Check relevance threshold
    if top_score < RELEVANCE_THRESHOLD:
        async def refusal_generator():
            refusal_message = "I'm sorry, but I can't help you with that. I only answer student academic queries related to EARIST."
            yield f"data: {json.dumps({'type': 'token', 'content': refusal_message})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(refusal_generator(), media_type="text/event-stream")

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
            "retrieved_chunks": initial_chunks,
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
