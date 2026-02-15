import os
import sys
import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DATA_FOLDER,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

def recursive_split(text, chunk_size=350, overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1 and last_newline > start + (chunk_size // 2):
                end = last_newline + 1
            else:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + (chunk_size // 2):
                    end = last_space + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def load_clean_data():
    documents = []
    
    # --- SURGICAL REMOVAL: SKIPPING FOOD EXCHANGE JSON ---
    # We no longer ingest 'food_exchange_clean.json' because 
    # math substitutions are handled by the deterministic Math Node.
    
    # 1. Nutrition Ebook (Markdown)
    md_path = os.path.join(DATA_FOLDER, "nutration-ebook_clean.md")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = recursive_split(content, chunk_size=500, overlap=150)
        for chunk in chunks:
            meta = {"source": "nutrition_ebook", "type": "text_concept"}
            documents.append({"text": chunk, "metadata": meta})
            
    # 2. Training Guide (Markdown)
    train_path = os.path.join(DATA_FOLDER, "Training-Guide-Diet-Cheat-1_clean.md")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = recursive_split(content, chunk_size=500, overlap=150)
        for chunk in chunks:
            meta = {"source": "training_guide", "type": "text_concept"}
            documents.append({"text": chunk, "metadata": meta})

    return documents

def main():
    print("--- Starting Ingestion (CLEAN VERSION: NO FOOD LIST) ---")
    
    # Initialize Client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Reset Collection (Delete old one to remove food items)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    except:
        pass
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )
    
    docs = load_clean_data()
    print(f"Loaded {len(docs)} chunks from Ebooks only.")
    
    if not docs:
        print("No data found!")
        return

    # Batch Add (Chroma handles batching better than one-by-one)
    ids = [str(uuid.uuid4()) for _ in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    
    # Simple batching to avoid hitting limits if any
    BATCH_SIZE = 100
    for i in range(0, len(docs), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(docs))
        print(f"Adding batch {i} to {end}...")
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        
    print(f"Success! Rebuilt RAG DB without Food List.")

if __name__ == "__main__":
    main()
