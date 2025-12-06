# AmangBot Backend

This is the FastAPI backend for AmangBot.

## Setup

1.  Ensure you have Python installed (3.10+ recommended).
2.  Install the required packages:
    ```powershell
    pip install -r requirements.txt
    ```
    *Note: For GPU support with `llama-cpp-python`, you might need to install it with specific flags or pre-compiled wheels. See [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python).*

## Running

Run the start script:
```powershell
.\start_backend.ps1
```

Or manually:
```powershell
python app.py
```

The server will start on `http://localhost:8000`.

## Features

*   **FAISS Indexing**: On first run, it will read JSON files from `../Dataset/300TokenDataset`, create embeddings using `nomic-ai/nomic-embed-text-v1.5`, and save a FAISS index locally.
*   **Streaming Response**: The `/chat/stream` endpoint streams the response token by token.
*   **GPU Acceleration**: Configured to use GPU for both embedding creation (if available) and LLM inference.
