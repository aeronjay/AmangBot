import pickle
import os

# List all available pickle files in the saved_chunks directory
saved_chunks_dir = os.path.join(os.path.dirname(__file__), '../Retriever/saved_chunks')

print("Available chunk pickle files:")
for fname in os.listdir(saved_chunks_dir):
    if fname.endswith('.pkl'):
        print(fname)
