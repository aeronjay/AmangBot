import json
import os

def process_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_path = os.path.join(base_dir, 'retriever_chunks.json')
    train_path = os.path.join(base_dir, 'train.json')
    output_path = os.path.join(base_dir, 'train_dataset.json')

    print(f"Loading chunks from {chunks_path}...")
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {chunks_path} not found.")
        return

    # Create a mapping from id_num to formatted content
    chunk_map = {}
    for chunk in chunks:
        id_num = chunk.get('id_num')
        if id_num:
            source = chunk.get('source', 'Unknown')
            category = chunk.get('category', 'Unknown')
            topic = chunk.get('topic', 'Unknown')
            content = chunk.get('content', '')
            
            # Format: Source: [Source]; Category: [Category]; Topic: [Topic]; Content: [Content]
            formatted_content = f"Source: {source}; Category: {category}; Topic: {topic}; Content: {content}"
            chunk_map[id_num] = formatted_content
    
    print(f"Loading training data from {train_path}...")
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {train_path} not found.")
        return
    
    new_train_data = []
    missing_chunks = 0
    
    for item in train_data:
        chunk_id = item.get('positive_chunk_id')
        if chunk_id and chunk_id in chunk_map:
            new_item = {
                'question': item['question'],
                'positive_document': chunk_map[chunk_id]
            }
            new_train_data.append(new_item)
        else:
            if chunk_id:
                print(f"Warning: Chunk ID {chunk_id} not found in retriever_chunks.json")
                missing_chunks += 1
            
    print(f"Processed {len(train_data)} items.")
    if missing_chunks > 0:
        print(f"Skipped {missing_chunks} items due to missing chunks.")
        
    print(f"Saving {len(new_train_data)} items to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_train_data, f, indent=4, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    process_files()
