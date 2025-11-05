import pickle
import os

# Use the correct absolute path for the pickle file
pkl_path = os.path.join(os.path.dirname(__file__), '../Retriever/saved_chunks/chunks_p50_b1.pkl')
output_txt_path = os.path.join(os.path.dirname(__file__), 'readable_chunks.txt')

if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

# Load the pickle file
with open(pkl_path, "rb") as f:
    chunks = pickle.load(f)

# Write chunks to a readable txt file
with open(output_txt_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        # Try to get the text content (adjust if your chunk object is different)
        text = getattr(chunk, 'page_content', str(chunk))
        f.write(f"--- Chunk {i+1} ---\n")
        f.write(text)
        f.write("\n\n")

print(f"Wrote {len(chunks)} chunks to {output_txt_path}")
