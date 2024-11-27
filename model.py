import os
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")

# Load upgraded embedding model
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)

# Load upgraded QA pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if device == torch.device("mps") else -1)

# Load upgraded summarization pipeline
summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == torch.device("mps") else -1)

# Load contexts for FAISS index
def load_squad_contexts(file_path):
    with open(file_path, "r") as f:
        squad_data = json.load(f)
    documents = [p["context"] for d in squad_data["data"] for p in d["paragraphs"]]
    return documents

# Path to SQuAD JSON files
train_file = os.path.join("documents", "train-v1.1.json")
documents = load_squad_contexts(train_file)

# Index documents with FAISS
doc_embeddings = embedder.encode(documents, convert_to_tensor=True).cpu().detach().numpy()
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
