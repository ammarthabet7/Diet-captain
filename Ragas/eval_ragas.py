import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional
import pandas as pd
from datasets import Dataset
from ragas.run_config import RunConfig

# --- 1. Project Setup ---
# Allow importing chat.py from the parent directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your actual RAG bot
try:
    from chat import run_turn
    print("✅ Successfully imported 'chat.run_turn'")
except ImportError:
    print("❌ CRITICAL: Could not import 'chat.py'. Check paths.")
    sys.exit(1)

# --- 2. Modern RAGAS Imports (v0.2+) ---
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

# --- 3. Helper Functions ---

def load_dataset_robust(path: str, split_name: str = "test") -> List[Dict[str, Any]]:
    """Loads dataset whether it's a List or a Dict with splits."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        print(f"ℹ️  Dataset is a list. Using all {len(data)} items.")
        return data
    elif isinstance(data, dict):
        if split_name in data:
            print(f"ℹ️  Loaded split '{split_name}' with {len(data[split_name])} items.")
            return data[split_name]
        else:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(data.keys())}")
    else:
        raise ValueError("Dataset must be a JSON list or dict.")

def load_cached_inputs(split_key: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(OUT_DIR, f"ragas_input_{split_key}.jsonl")
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except: return None

def save_inputs(split_key: str, rows: List[Dict[str, Any]]):
    path = os.path.join(OUT_DIR, f"ragas_input_{split_key}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- 4. Main Inference Logic ---

def run_bot_collect(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    history = [] # Reset history for each run or keep it if continuous chat is needed
    
    print(f"🚀 Running Inference on {len(examples)} queries...")
    
    for i, ex in enumerate(examples):
        q = ex.get("question")
        gt = ex.get("ground_truth")
        # Handle field mismatch (docsource vs doc_source)
        src_doc = ex.get("docsource") or ex.get("doc_source")
        
        print(f"[{i+1}/{len(examples)}] Q: {q[:40]}...")
        
        # Call your Pipeline
        # run_turn returns: {'answer': str, 'sources': [{'snippet':..., 'source':...}], ...}
        out = run_turn(q, history)
        
        answer = out.get("answer", "")
        sources_list = out.get("sources", [])
        
        # Extract Contexts from RAG Sources
        # RAGAS expects 'contexts' to be a list of strings
        contexts = [s.get("snippet", "") for s in sources_list]
        
        # If no context was retrieved, provide empty list (Metrics will punish this)
        if not contexts:
            print(f"   ⚠️ No context retrieved for this query.")

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
            "doc_source": src_doc
        })
        
        # Optional: Reset history if you want each question to be independent
        history = [] 
        
    return rows

# --- 5. RAGAS Evaluator Setup ---

def make_evaluator():
    print("⚖️ Initializing Perplexity Judge...")

    # Perplexity API Configuration (As requested)
    judge_model = "sonar-pro"
    pplx_api_key = os.getenv("PERPLEXITY_API_KEY")

    if not pplx_api_key:
        print("⚠️ Warning: PERPLEXITY_API_KEY not found in environment.")

    pplx_chat = ChatOpenAI(
        openai_api_base="https://api.perplexity.ai",
        openai_api_key=pplx_api_key,
        model_name=judge_model,
        temperature=0.0,
        request_timeout=60,
        max_retries=2,
        n=1,  # <--- Move it here!
        # model_kwargs={"n": 1}  <--- REMOVE THIS
    )


    wrapped_llm = LangchainLLMWrapper(pplx_chat)

    # Embeddings (Local HF is fine and free)
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    hf_emb = HuggingFaceEmbeddings(model_name=emb_model)
    wrapped_emb = LangchainEmbeddingsWrapper(hf_emb)

    return wrapped_llm, wrapped_emb

def evaluate_rows(split_key: str, rows: List[Dict[str, Any]]):
    # Convert to HuggingFace Dataset
    ds = Dataset.from_list(rows)
    
    llm, embeddings = make_evaluator()
    
    # Define Metrics
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm)
    ]
    
    # Run Config
    run_config = RunConfig(
        max_workers=4, # Parallel execution
        timeout=120, 
        max_retries=2
    )
    
    print(f"📉 Starting RAGAS Evaluation for '{split_key}'...")
    
    # Execute Evaluation
    result = evaluate(ds, metrics=metrics, run_config=run_config)
    
    # --- 6. Saving Results ---
    scores_df = result.to_pandas()
    
    # Save CSV
    out_path = os.path.join(OUT_DIR, f"ragas_scores_{split_key}.csv")
    scores_df.to_csv(out_path, index=False)
    
    # Summary Stats
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = scores_df[metric_cols].mean(numeric_only=True)
    
    print("\n" + "="*40)
    print("📊 Evaluation Summary:")
    print(summary)
    print("="*40)
    print(f"✅ Detailed results saved to: {out_path}")

# --- 7. Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline using RAGAS")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset")
    parser.add_argument("--split", default="test", help="Split name (if dataset is a dict)")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-run inference")
    args = parser.parse_args()
    
    # 1. Load Data
    examples = load_dataset_robust(args.dataset, args.split)
    
    # 2. Inference (Check Cache)
    cached = load_cached_inputs(args.split)
    
    if not cached or args.force:
        if args.force: print("🔄 Force flag set. Ignoring cache.")
        else: print("📂 No cache found.")
        
        inference_results = run_bot_collect(examples)
        save_inputs(args.split, inference_results)
    else:
        print(f"📦 Loaded {len(cached)} cached RAG responses. Skipping inference.")
        inference_results = cached
        
    # 3. Evaluate
    evaluate_rows(args.split, inference_results)

if __name__ == "__main__":
    main()
