import pandas as pd
import json
import os

# Define paths
base_dir = r"c:\Users\tebats\Baste\Projects\AmangBot"
output_csv_path = os.path.join(base_dir, "trygpurunmodel", "output.csv")
qds_json_path = os.path.join(base_dir, "Dataset", "goldends", "100qds.json")
chunks_metadata_path = os.path.join(base_dir, "ambot-be", "chunks_metadata.json")
new_csv_output_path = os.path.join(base_dir, "trygpurunmodel", "formatted_dataset_for_eval.csv")

def create_formatted_dataset():
    print("Starting dataset processing...")
    
    # 1. Load Data
    try:
        # Load output.csv
        df_output = pd.read_csv(output_csv_path)
        print(f"Loaded output.csv with {len(df_output)} rows.")
        
        # Load 100qds.json
        with open(qds_json_path, 'r', encoding='utf-8') as f:
            qds_data = json.load(f)
        # Create a mapping from question to context_id
        q_to_context_id = {item['question'].strip(): item['context_id'] for item in qds_data}
        print(f"Loaded 100qds.json with {len(qds_data)} items.")
        
        # Load chunks_metadata.json
        with open(chunks_metadata_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        # Create mapping from id to chunk data
        chunks_map = {item['id']: item for item in chunks_data}
        print(f"Loaded chunks_metadata.json with {len(chunks_data)} items.")
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Process Data
    new_rows = []

    for index, row in df_output.iterrows():
        question = row.get('original question', '')
        generated_answer = row.get('generated answer', '')
        ground_truth_csv = row.get('answer', '') # based on report: "ground_truth (based on answer in output.csv)"
        
        # Clean question for lookup
        question_clean = str(question).strip()
        
        # Get context_id from 100qds based on question match
        context_id = q_to_context_id.get(question_clean)
        
        # If not found in 100qds map, try the one in the CSV if it exists
        if not context_id and 'context_id' in row:
             context_id = row.get('context_id')

        contexts_list = []
        
        if context_id:
            chunk = chunks_map.get(context_id)
            if chunk:
                # Combine fields as requested: source, category, topic, content
                formatted_context = (
                    f"Source: {chunk.get('source', '')}\n"
                    f"Category: {chunk.get('category', '')}\n"
                    f"Topic: {chunk.get('topic', '')}\n"
                    f"Content: {chunk.get('content', '')}"
                )
                contexts_list.append(formatted_context)
            else:
                # Log warning but continue
                # print(f"Warning: Context ID '{context_id}' not found in metadata for q: {question[:20]}...")
                pass
        
        new_rows.append({
            'question': question,
            'answer': generated_answer,
            'ground_truth': ground_truth_csv,
            'contexts': contexts_list 
        })

    # 3. Create DataFrame and Save
    df_new = pd.DataFrame(new_rows)
    
    # Save to CSV
    df_new.to_csv(new_csv_output_path, index=False)
    print(f"Successfully created new CSV at: {new_csv_output_path}")
    print(f"Total rows: {len(df_new)}")
    print(df_new.head())

if __name__ == "__main__":
    create_formatted_dataset()
