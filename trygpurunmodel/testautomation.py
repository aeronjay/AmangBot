import sys
import os
import json
import time
import pandas as pd
import numpy as np
import torch
import re

# Add backend to path so we can import app and its dependencies
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ambot-be"))
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import app module from ambot-be
import app as ambot_app

def run_test_automation():
    # Capture script directory for saving output later
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize resources (Load Models, Indices, etc.)
    print("Loading AMBOT backend resources...")
    
    # Change working directory to backend path so app.py can find the index files
    # This prevents re-embedding everything
    if os.getcwd() != backend_path:
        print(f"Changing working directory to {backend_path}")
        os.chdir(backend_path)

    ambot_app.load_resources()
    print("Resources loaded successfully.")

    # Path to the dataset
    # Using the path from the conversation context
    dataset_path = r"c:\Users\tebats\Baste\Projects\AmangBot\Dataset\goldends\100qds.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    results = []
    
    print(f"Starting evaluation of {len(questions_data)} questions to generate output.csv")

    for i, item in enumerate(questions_data):
        original_question = item.get('question')
        ground_truth_answer = item.get('ground_truth_answer')
        ground_truth_context_id = item.get('context_id')
        
        print(f"Processing {i+1}/{len(questions_data)}: {original_question[:50]}...")
        
        start_time = time.time()
        
        # --- LOGIC COPIED/ADAPTED FROM app.py chat_stream ---
        
        query = original_question
        # Capitalize 'earist' -> 'EARIST'
        query = re.sub(r'earist', 'EARIST', query, flags=re.IGNORECASE)
        
        # Contextualize
        # For single question evaluation without history, contextualized_query usually equals query
        # But we call the function to be consistent with the pipeline
        contextualized_query = ambot_app.contextualize_query(query, [])
        
        # Embed query
        # Add prefix as required by nomic-embed-text-v1.5
        query_embedding = ambot_app.embedder.encode(["search_query: " + contextualized_query]).astype('float32')
        
        # Search FAISS
        k = 20
        D, I = ambot_app.index.search(query_embedding, k)
        faiss_indices = I[0]
        
        # Search BM25
        tokenized_query = contextualized_query.lower().split()
        bm25_scores = ambot_app.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        
        # Combine indices
        combined_indices = list(set(faiss_indices) | set(bm25_indices))
        
        initial_chunks = []
        for idx in combined_indices:
            if idx < len(ambot_app.chunks_metadata) and idx >= 0:
                initial_chunks.append(ambot_app.chunks_metadata[idx])
        
        # Rerank
        retrieved_chunks = []
        if initial_chunks:
            pairs = [[contextualized_query, chunk.get('content', '')] for chunk in initial_chunks]
            
            with torch.no_grad():
                inputs = ambot_app.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
                # app.py forces cuda for reranker, but here we check just in case, or stick to app.py's implementation
                # ambot_app.reranker_model is already on the device set in app.py (which is 'cuda')
                
                scores = ambot_app.reranker_model(**inputs).logits.squeeze(-1)
            
            sorted_indices = scores.argsort(descending=True)
            k_final = 5
            top_indices = sorted_indices[:k_final].tolist()
            retrieved_chunks = [initial_chunks[ix] for ix in top_indices]
            
        # Generate
        MAX_PROMPT_TOKENS = 4484
        prompt = ""
        
        # Make a copy to modify
        current_chunks = retrieved_chunks.copy()
        
        while True:
            prompt = ambot_app.construct_prompt(contextualized_query, current_chunks)
            tokens = ambot_app.llm.tokenize(prompt.encode('utf-8'))
            
            if len(tokens) <= MAX_PROMPT_TOKENS:
                break
            if not current_chunks:
                break
            current_chunks.pop() # Remove least relevant (last one)
        
        # Generate with Llama
        output = ambot_app.llm(
            prompt,
            temperature=0.1,
            top_p=0.5, 
            max_tokens=1660,
            stop=["</s>", "[/INST]"],
            echo=False
        )
        generated_answer = output['choices'][0]['text'].strip()
        
        latency = time.time() - start_time
        
        # Prepare result
        retrieved_ids = [chunk.get('id') for chunk in current_chunks]
        
        results.append({
            "original question": original_question,
            "answer": ground_truth_answer,
            "generated answer": generated_answer,
            "retrieved chunks id": str(retrieved_ids), # String representation as requested often for CSV or just list
            "context_id": ground_truth_context_id,
            "contextualized query": contextualized_query,
            "latency": round(latency, 4)
        })

    # Save to CSV
    # Use script_dir captured at start of function
    output_csv = os.path.join(script_dir, "output.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Test completed. Results saved to {output_csv}")

if __name__ == "__main__":
    run_test_automation()
