import os
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv()

# --- PROJECT PATHS ---
# Base directory is the directory containing this config file (project root if config.py is at root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "ready_data")
FOOD_JSON_PATH = os.path.join(DATA_FOLDER, "food_exchange_clean.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Ensure logs exist
os.makedirs(LOG_DIR, exist_ok=True)

# --- API KEYS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in .env")

# --- MODEL NAMES ---
COLLECTION_NAME = "diet_cheat_rag"

# LLM Models
GEMINI_MODEL_NAME = 'gemma-3-27b-it'
MODEL_PRIMARY = "Qwen/Qwen2.5-7B-Instruct"
MODEL_FALLBACK = "Qwen/Qwen2.5-7B-Instruct"

# Embedding & Reranking
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# --- CACHE ---
# Minimum similarity score for semantic cache hit
CACHE_SIMILARITY_THRESHOLD = 0.90
