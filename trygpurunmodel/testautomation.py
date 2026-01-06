import pandas as pd
import json
import requests
import os
import time

def run_test_automation(
    input_json_path,
    output_excel_path="test_results.xlsx",
    api_url="http://localhost:8000/chat/stream"
):
    """
    Runs test automation against the AmangBot backend.
    
    WHY RUN AGAINST BACKEND (app.py)?
    To conserve computing power (RAM/VRAM), it is best to run the backend server (`app.py`) once 
    and treat this test script as a client. 
    
    - If we imported the model code here directly, we would load the LLM and Embedding models 
      into memory a second time (or have to shut down the backend), which is inefficient 
      and potentially leads to Out Of Memory (OOM) errors.
    - Using the API allows testing the actual serving infrastructure.
    
    Args:
        input_json_path (str): Path to the input JSON file containing questions.
        output_excel_path (str): Path where the output Excel file will be saved.
        api_url (str): The URL of the chat stream endpoint.
    """
    
    print("Starting test automation...")
    print("Configuration: Using API Client Mode (Conserves Compute Resources)")
    print(f"Target API: {api_url}")
    print(f"Reading questions from: {input_json_path}")
    
    # Load questions
    if not os.path.exists(input_json_path):
        print(f"Error: File not found at {input_json_path}")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    results = []
    
    print(f"Found {len(questions)} questions. Processing...")

    for i, item in enumerate(questions):
        question_id = item.get('id', i+1)
        question_text = item.get('question', '')
        ground_truth = item.get('ground_truth_answer', '')
        ground_truth_id = item.get('context_id', '')
        
        print(f"Processing Q{question_id}: {question_text[:50]}...")
        
        start_time = time.time()
        
        full_response = ""
        retrieved_chunks = []
        status = "Success"
        
        try:
            # Prepare payload
            payload = {
                "message": question_text,
                "history": [] 
            }
            
            # Make streaming request
            # We set stream=True to handle the Server-Sent Events (SSE)
            response = requests.post(api_url, json=payload, stream=True, timeout=120)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            try:
                                data = json.loads(data_str)
                                
                                if data.get("type") == "metadata":
                                    # Capture the chunks used for generation
                                    retrieved_chunks = data.get("chunks", [])
                                elif data.get("type") == "token":
                                    # Append tokens to form the answer
                                    full_response += data.get("content", "")
                                elif data.get("type") == "done":
                                    break
                            except json.JSONDecodeError:
                                pass
            else:
                full_response = f"HTTP Error: {response.status_code} - {response.text}"
                status = "Error"
                
        except requests.exceptions.ConnectionError:
            full_response = "Connection Error: Ensure app.py is running on port 8000."
            print("  !! Connection Error. Is the backend running?")
            status = "Connection Error"
            # Optional: break or continue? Let's continue to try next queries or stop.
            # Usually if server is down, we stop.
            print("  Stopping automation due to connection failure.")
            break
        except Exception as e:
            full_response = f"Error: {str(e)}"
            status = "Error"
            print(f"  Error: {e}")

        end_time = time.time()
        duration = end_time - start_time
        
        # Format chunks for Excel
        chunks_summary = ""
        retrieved_ids = []
        
        for idx, chunk in enumerate(retrieved_chunks):
            source = chunk.get('source', 'Unknown')
            chunk_id = chunk.get('id', 'N/A')
            retrieved_ids.append(chunk_id)
            
            # Extract a snippet
            content_snippet = chunk.get('content', '')[:100].replace("\n", " ")
            chunks_summary += f"[{idx+1}] {source} (ID: {chunk_id}): {content_snippet}...\n"

        # Check for retrieval accuracy
        retrieval_success = False
        if ground_truth_id and ground_truth_id in retrieved_ids:
            retrieval_success = True

        results.append({
            "ID": question_id,
            "Question": question_text,
            "Ground Truth": ground_truth,
            "Ground Truth ID": ground_truth_id,
            "Model Answer": full_response,
            "Retrieved Chunks Used": chunks_summary,
            "Retrieved IDs": ", ".join(str(x) for x in retrieved_ids),
            "Retrieval Accuracy": 1 if retrieval_success else 0,
            "Processing Time (s)": round(duration, 2),
            "Status": status
        })
        
        # Slight delay to be nice to the server (optional)
        # time.sleep(0.5)

    if not results:
        print("No results to save.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to Excel
    print(f"Saving results to {output_excel_path}...")
    try:
        # Requires 'openpyxl' installed
        df.to_excel(output_excel_path, index=False)
        print("Successfully saved Excel file.")
    except ImportError:
        print("Error: 'openpyxl' library is missing. Install with: pip install openpyxl")
        print("Saving as CSV instead...")
        csv_path = output_excel_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Define paths
    # Assuming script is in trygpurunmodel/ and wants to access dataset in ../Dataset/goldends/100qds.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    dataset_path = os.path.join(project_root, "Dataset", "goldends", "100qds.json")
    
    # You can change this to output elsewhere
    output_path = os.path.join(base_dir, "automation_results.xlsx")
    
    run_test_automation(dataset_path, output_path)
