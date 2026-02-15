import os
import json
import re
from retriever import DietRetriever

# --- Helper to Clean Arabic Text for comparison ---
def clean_text(text):
    if not text: return ""
    # Remove special chars but keep Arabic letters and spaces
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text.strip()

# --- 1. Load Dataset ---
def load_dataset(path):
    if not os.path.exists(path):
        print(f"❌ Error: File not found at {path}")
        exit(1)
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Handle list vs dict format
    if isinstance(data, dict) and "test" in data:
        return data["test"]
    return data

# --- 2. Metric Logic (Strict) ---
def calculate_metrics(retrieved_docs, ground_truth_text):
    hits = 0
    rank = 0
    
    gt_clean = clean_text(ground_truth_text)
    gt_words = set(gt_clean.split())
    
    if len(gt_words) == 0:
        return 0, 0 

    for i, doc in enumerate(retrieved_docs):
        chunk_text = clean_text(doc.get("text", ""))
        chunk_words = set(chunk_text.split())
        
        common_words = gt_words.intersection(chunk_words)
        overlap_score = len(common_words) / len(gt_words)
        
        if overlap_score >= 0.30: 
            hits = 1
            rank = 1 / (i + 1)
            break
            
    return hits, rank

# --- 3. Main Evaluation Loop ---
def evaluate_retrieval(dataset_path):
    try:
        retriever = DietRetriever()
        print("✅ Retriever Initialized Successfully.")
    except Exception as e:
        print(f"❌ Retriever Init Failed: {e}")
        return

    data = load_dataset(dataset_path)
    
    total_hits = 0
    total_mrr = 0
    total_queries = len(data)
    
    print(f"\n🚀 Starting STRICT Retrieval Evaluation on {total_queries} queries...")
    print(f"   Mode: Keyword Overlap Check (Threshold: 30%)\n")
    
    for idx, item in enumerate(data):
        query = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        results = retriever.search(query, k=5)
        
        hit, mrr = calculate_metrics(results, ground_truth)
        
        total_hits += hit
        total_mrr += mrr
        
        if not hit:
            print(f"\n❌ [Q{idx+1}] FAILED: '{query}'")
            print(f"   Target Answer Snippet: {ground_truth[:100]}...")
            print(f"   --- RETRIEVED CHUNKS ---")
            for i, res in enumerate(results[:3]): # Show top 3 results
                src = res['meta'].get('source', 'unknown')
                print(f"   [{i+1}] ({src}): {res['text'][:150]}...")
            print("-" * 60)

    print("\n" + "="*50)
    print(f"📊 STRICT RETRIEVAL SCORE REPORT")
    print(f"Total Queries: {total_queries}")
    print(f"Hit Rate @ k=5: {(total_hits/total_queries)*100:.1f}%")
    print(f"MRR (Mean Rank): {total_mrr/total_queries:.3f}")
    print("="*50)

if __name__ == "__main__":
    path = "ragas_dataset_clean.json"
    if not os.path.exists(path):
        path = "Ragas/ragas_dataset_clean.json"
        
    evaluate_retrieval(path)
