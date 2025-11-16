import json
import os
import torch
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from types import SimpleNamespace
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# 1. Initialize Flask App
# --------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------------------------------------------------------------
# 2. Load Data and Models (Done once on startup)
# --------------------------------------------------------------------------
print("🚀 Initializing backend...")

# --- Load Pre-processed Chunks ---
print("📚 Loading pre-processed chunks...")
try:
    with open('data/Chunks.json', 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    chunks = []
    for item in chunks_data:
        chunk_obj = SimpleNamespace(
            page_content=item['content'],
            metadata={'chunk_id': item['chunk_id'], **item['metadata']}
        )
        chunks.append(chunk_obj)
    print(f"✅ Loaded {len(chunks)} chunks.")
except FileNotFoundError:
    print("❌ ERROR: 'data/Chunks.json' not found. Please ensure the data file is in the correct directory.")
    chunks = []

# --- Initialize Embedding Model ---
print("🔧 Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True
    },
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded: nomic-ai/nomic-embed-text-v1.5")

# --- Generate Embeddings and Build FAISS Index ---
print("🔄 Generating embeddings and building FAISS index...")
if chunks:
    chunk_texts = [chunk.page_content for chunk in chunks]
    all_embeddings = embedding_model.embed_documents(chunk_texts)
    
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    embeddings_array = np.array(all_embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    print(f"✅ FAISS index ready: {index.ntotal} vectors.")
else:
    index = None
    print("⚠️ FAISS index not built because no chunks were loaded.")

# --- Load Language Model ---
print("🤖 Loading language model...")
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print(f"✅ Model loaded: {model_name}")
print("🎉 Backend initialization complete!")

# --------------------------------------------------------------------------
# 3. Core RAG Functions
# --------------------------------------------------------------------------
def retrieve_relevant_chunks(query, top_k=5):
    if not index or not chunks:
        return []
    query_embedding = embedding_model.embed_query(query)
    query_vector = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                'text': chunk.page_content,
                'score': float(score),
                'chunk_id': chunk.metadata.get('chunk_id', int(idx)),
            })
    return results

def generate_answer(query, context_chunks, max_new_tokens=350):
    MAX_TOTAL_TOKENS = 4096
    MAX_OUTPUT_TOKENS = max_new_tokens
    MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS - 32

    working_chunks = list(context_chunks)
    while True:
        if not working_chunks:
            combined_context = "No information found in the handbook."
            break

        best_chunk = working_chunks[0]
        other_chunks = working_chunks[1:]
        other_chunks.reverse()

        context_parts = [f"<HANDBOOK_SECTION_{i+1}>\n{chunk['text']}\n</HANDBOOK_SECTION_{i+1}>" for i, chunk in enumerate(other_chunks)]
        context_parts.append(f"<HANDBOOK_SECTION_MOST_RELEVANT>\n{best_chunk['text']}\n</HANDBOOK_SECTION_MOST_RELEVANT>")
        combined_context = "\n\n".join(context_parts)

        prompt = f"""
<s>[INST]
You are **Amang Bot (Ambot)**, a precise and helpful university advisor. 
You must answer the user's question **using only the information provided in the INFORMATION SECTIONS**. 

### Your Answer Must Follow These Rules:
1. **Start with a direct answer** to the user's question.
2. After the direct answer, **provide additional relevant details**, background, or context based strictly on the provided information.
3. If the information sections **do not include the answer**, explicitly say:
   "The provided information does not contain the answer to this question."
4. Do NOT invent or assume facts that are not in the information sections.
5. Maintain a clear, formal, and student-friendly tone.

---

### INFORMATION SECTIONS:
{combined_context}

---

### USER QUESTION:
{query}

Provide your answer now, following the rules above.
[/INST]
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = inputs['input_ids'].shape[1]

        if input_length <= MAX_INPUT_TOKENS:
            break
        elif len(working_chunks) > 1:
            working_chunks.pop(-1)
        else:
            break

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    input_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer

# --------------------------------------------------------------------------
# 4. Flask Routes
# --------------------------------------------------------------------------
@app.route('/ask', methods=['POST'])
def ask_handbook_route():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid request. "question" field is required.'}), 400

    question = data['question']
    print(f"\n❓ Received question: {question}")

    try:
        # Retrieve
        relevant_chunks = retrieve_relevant_chunks(question, top_k=5)
        if not relevant_chunks:
            return jsonify({
                'question': question,
                'answer': "The provided information does not contain the answer to this question.",
                'sources': []
            })

        # Generate
        answer = generate_answer(question, relevant_chunks)

        response = {
            'question': question,
            'answer': answer,
            'sources': relevant_chunks
        }
        
        print(f"💡 Sending answer: {answer[:100]}...")
        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR during processing: {str(e)}")
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify that the service is running."""
    return jsonify({'status': 'ok', 'message': 'Backend is running.'})

# --------------------------------------------------------------------------
# 5. Run Flask App
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Use a production-ready WSGI server like Gunicorn or Waitress in a real deployment
    app.run(host='0.0.0.0', port=5000, debug=False)
