# --- Configuration ---
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Paths (adjust if needed)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend2"))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "Dataset/Integrated/Latest"))

# Model options:
# Option 1: Use local fine-tuned model
NOMIC_MODEL_PATH = "../../Models/nomic-finetuned/nomic-finetuned-final"
# Option 2: Use official Nomic model from HuggingFace (nomic-ai/nomic-embed-text-v1.5)
USE_HUGGINGFACE_NOMIC = True  # Set to True to use official HuggingFace model

# Thresholds
SIMILARITY_THRESHOLD = 1.4  # FAISS L2 distance. Lower is better.

print(f"Dataset directory: {DATASET_DIR}")
if USE_HUGGINGFACE_NOMIC:
    print("Using official Nomic model from HuggingFace: nomic-ai/nomic-embed-text-v1.5")
else:
    print(f"Using local Nomic model path: {NOMIC_MODEL_PATH}")

# --- Load Chunks from Dataset ---
chunks = []
chunks_for_embedding = []

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found at {DATASET_DIR}")

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

# --- Initialize Embedder ---
print("Initializing Nomic Embedder...")
if USE_HUGGINGFACE_NOMIC:
    # Use official Nomic model from HuggingFace
    embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
else:
    # Use local fine-tuned model
    embedder = SentenceTransformer(NOMIC_MODEL_PATH, trust_remote_code=True)
print("Embedder initialized.")

# --- Build FAISS Index ---
print("Building FAISS index...")
embeddings = embedder.encode(chunks_for_embedding, convert_to_numpy=True, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")
print(f"Embedding dimension: {dimension}")

# --- Initialize Reranker ---
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Reranker initialized.")
except Exception as e:
    reranker = None
    print(f"Reranker failed to initialize: {e}")

# --- Retrieval Function ---
def retrieve(query: str, k: int = 4):
    """
    Retrieve relevant chunks for a given query.
    
    Args:
        query: The search query
        k: Number of results to return
    
    Returns:
        Tuple of (final_chunks, distances, is_relevant)
    """
    # Encode query
    query_text = f"search_query: {query}"
    query_embedding = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Fetch candidates (3x for reranking)
    fetch_k = k * 3 if reranker else k
    D, I = index.search(query_embedding, fetch_k)
    
    # Check if results are relevant based on distance threshold
    is_relevant = D[0][0] <= SIMILARITY_THRESHOLD
    
    retrieved_chunks = [chunks[i] for i in I[0]]
    distances = D[0].tolist()
    
    # Rerank if available
    if reranker:
        pairs = [[query, doc] for doc in retrieved_chunks]
        scores = reranker.predict(pairs)
        scored_chunks = sorted(zip(retrieved_chunks, scores, distances), key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score, dist in scored_chunks[:k]]
        final_scores = [score for chunk, score, dist in scored_chunks[:k]]
        return final_chunks, final_scores, is_relevant
    else:
        return retrieved_chunks[:k], distances[:k], is_relevant

print("Retrieval function defined.")

# --- Helper function to display results ---
def display_results(query: str, results: list, scores: list, is_relevant: bool):
    """Display retrieval results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"🔍 Query: {query}")
    print('='*70)
    print(f"📊 Relevance Check (within threshold {SIMILARITY_THRESHOLD}): {'✅ YES' if is_relevant else '❌ NO'}")
    print(f"\n📚 Top {len(results)} Retrieved Chunks:\n")
    
    for i, (chunk, score) in enumerate(zip(results, scores), 1):
        print(f"┌{'─'*68}┐")
        print(f"│ Result {i} │ Score: {score:.4f}")
        print(f"├{'─'*68}┤")
        # Word wrap the chunk for better display
        chunk_display = chunk[:500] + "..." if len(chunk) > 500 else chunk
        for line in chunk_display.split('\n'):
            print(f"│ {line[:66]}")
        print(f"└{'─'*68}┘")
        print()

# --- Test Retrieval ---
# Example queries to test
test_queries = [
    "What are the admission requirements?",
    "Who is the president of EARIST?",
    "What are the school fees?",
]

print("\n" + "="*70)
print("🧪 RUNNING TEST QUERIES")
print("="*70)

for query in test_queries:
    results, scores, is_relevant = retrieve(query, k=3)
    display_results(query, results, scores, is_relevant)

# --- Interactive Mode ---
def interactive_mode():
    """
    Interactive mode for querying the retrieval system.
    Type 'quit', 'exit', or 'q' to exit.
    Type 'help' for available commands.
    """
    print("\n" + "="*70)
    print("🤖 INTERACTIVE RETRIEVAL MODE")
    print("="*70)
    print("Type your questions to see what chunks get retrieved.")
    print("Commands:")
    print("  • 'quit', 'exit', or 'q' - Exit interactive mode")
    print("  • 'help' or 'h' - Show this help message")
    print("  • 'k=N' - Change number of results (e.g., 'k=5')")
    print("  • 'threshold=N' - Change similarity threshold (e.g., 'threshold=1.2')")
    print("  • 'full' - Toggle full chunk display (no truncation)")
    print("="*70 + "\n")
    
    k = 4  # Default number of results
    show_full = False  # Toggle for full chunk display
    global SIMILARITY_THRESHOLD
    
    while True:
        try:
            user_input = input("📝 Enter your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting interactive mode. Goodbye!")
                break
            
            if user_input.lower() in ['help', 'h']:
                print("\nCommands:")
                print("  • 'quit', 'exit', or 'q' - Exit interactive mode")
                print("  • 'help' or 'h' - Show this help message")
                print("  • 'k=N' - Change number of results (e.g., 'k=5')")
                print("  • 'threshold=N' - Change similarity threshold (e.g., 'threshold=1.2')")
                print("  • 'full' - Toggle full chunk display (no truncation)")
                print(f"\nCurrent settings: k={k}, threshold={SIMILARITY_THRESHOLD}, full_display={show_full}\n")
                continue
            
            if user_input.lower().startswith('k='):
                try:
                    new_k = int(user_input.split('=')[1])
                    if new_k > 0:
                        k = new_k
                        print(f"✅ Number of results changed to {k}\n")
                    else:
                        print("❌ k must be a positive integer\n")
                except ValueError:
                    print("❌ Invalid format. Use 'k=N' where N is an integer\n")
                continue
            
            if user_input.lower().startswith('threshold='):
                try:
                    new_threshold = float(user_input.split('=')[1])
                    if new_threshold > 0:
                        SIMILARITY_THRESHOLD = new_threshold
                        print(f"✅ Similarity threshold changed to {SIMILARITY_THRESHOLD}\n")
                    else:
                        print("❌ Threshold must be a positive number\n")
                except ValueError:
                    print("❌ Invalid format. Use 'threshold=N' where N is a number\n")
                continue
            
            if user_input.lower() == 'full':
                show_full = not show_full
                print(f"✅ Full display mode: {'ON' if show_full else 'OFF'}\n")
                continue
            
            # Perform retrieval
            results, scores, is_relevant = retrieve(user_input, k=k)
            
            # Display results
            print(f"\n{'='*70}")
            print(f"🔍 Query: {user_input}")
            print('='*70)
            print(f"📊 Relevance Check (threshold={SIMILARITY_THRESHOLD}): {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT'}")
            print(f"📈 Top Score: {scores[0]:.4f}" if scores else "No results")
            print(f"\n📚 Retrieved {len(results)} Chunks (k={k}):\n")
            
            for i, (chunk, score) in enumerate(zip(results, scores), 1):
                print(f"┌{'─'*68}┐")
                print(f"│ 📄 Result {i} │ Score: {score:.4f}")
                print(f"├{'─'*68}┤")
                
                # Display chunk (with or without truncation)
                if show_full:
                    chunk_display = chunk
                else:
                    chunk_display = chunk[:500] + "..." if len(chunk) > 500 else chunk
                
                for line in chunk_display.split('\n'):
                    # Word wrap long lines
                    while len(line) > 66:
                        print(f"│ {line[:66]}")
                        line = line[66:]
                    print(f"│ {line}")
                
                print(f"└{'─'*68}┘")
                print()
            
            print("-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

# --- Main Execution ---
if __name__ == "__main__":
    # Run interactive mode
    interactive_mode()