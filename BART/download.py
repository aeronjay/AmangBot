import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("\n🔧 Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",  # Nomic embedding model
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True
    },
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded: nomic-embed-text-v1.5 (768 dimensions)")