import os
import re
import json
import time
import uuid
import requests
import numpy as np
import google.generativeai as genai
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
from retriever import DietRetriever
from session_manager import SessionStore, SemanticCache
from config import (
    GEMINI_API_KEY, 
    GEMINI_MODEL_NAME, 
    HF_TOKEN, 
    MODEL_PRIMARY, 
    MODEL_FALLBACK, 
    FOOD_JSON_PATH, 
    LOG_DIR
)

# --- NEW IMPORTS FOR HUGGING FACE ---
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# -----------------------
# CONFIG
# -----------------------

# GEMINI CONFIG (Gemma 3 27B) - USED ONLY FOR ROUTER
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Initialize Client
client = InferenceClient(token=HF_TOKEN)

class TraceLogger:
    def __init__(self, request_id):
        self.request_id = request_id
        self.events = []
        self.summary = {}

    def log(self, node, t0, m0, extra=None):
        self.events.append({
            "node": node,
            "duration": round((time.perf_counter() - t0)*1000, 2),
            "extra": extra or {}
        })
        if extra and "final" in extra and node == "intent_router":
            self.summary["intent"] = extra["final"]
        if extra and "norm" in extra:
            self.summary["translated_query"] = extra["norm"]

    def write_jsonl(self):
        path = os.path.join(LOG_DIR, f"trace_{time.strftime('%Y-%m-%d')}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                rec = {"req_id": self.request_id, "summary": self.summary, "events": self.events}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except: pass

def gemini_chat_router(prompt):
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Router Error: {e}")
        return "GENERAL_QA" 

def hf_chat(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Robust HF Chat: Tries 72B -> Retries -> Falls back to 7B."""
    
    def try_generate(model_id, retries=3):
        for attempt in range(retries):
            try:
                response = client.chat_completion(
                    model=model_id, messages=messages, max_tokens=1024,
                    temperature=temperature, top_p=0.9
                )
                return response.choices[0].message.content.strip()
            except HfHubHTTPError as e:
                if e.response.status_code in [429, 503]:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ HF Busy ({model_id}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else: raise e
            except Exception as e:
                print(f"⚠️ Error with {model_id}: {e}")
                break
        return None

    # 1. Try Primary
    result = try_generate(MODEL_PRIMARY)
    if result:
        print(f"✅ Generated with: {MODEL_PRIMARY}")
        return result

    # 2. Fallback
    print(f"⚠️ Primary failed. Switching to Fallback ({MODEL_FALLBACK})...")
    result = try_generate(MODEL_FALLBACK)
    if result:
        print(f"✅ Generated with: {MODEL_FALLBACK}")
        return result
    
    return "Error: AI Service Unavailable."

def extract_json_from_text(text):
    if not text: return {} # Handle None input
    try:
        # Try to find JSON block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        # Try to find raw JSON object
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match: return json.loads(match.group(0))
        
        return {}
    except: return {}


# -----------------------
# DOMAIN LOGIC
# -----------------------
def normalize_arabic_light(text: str) -> str:
    text = re.sub(r"[أإآ]", "ا", text)
    return re.sub(r"ـ", "", text)

def load_food_index(path):
    if not os.path.exists(path): return {}, {}, [], None, None
    
    with open(path, "r", encoding="utf-8") as f: 
        data = json.load(f)
        
    all_items = []
    aliases = {}
    
    for cat in data:
        for item in cat.get("items", []):
            item["category_en"] = cat.get("category_en")
            all_items.append(item)
            
            # 1. Safe Canon (Arabic Name)
            canon = item.get("item_ar") or "" 
            if canon:
                aliases[normalize_arabic_light(str(canon)).strip().lower()] = canon
            
            # 2. Safe Aliases List
            for a in item.get("aliases", []) or []:
                if a: # Filter out None or empty strings
                    aliases[normalize_arabic_light(str(a)).strip().lower()] = canon
            
            # 3. Safe English Name
            item_en = item.get("item_en")
            if item_en:
                aliases[str(item_en).lower().strip()] = canon
                
    # 4. FINAL SAFETY FILTER: Remove any empty/None keys
    alias_keys = [k for k in aliases.keys() if k and isinstance(k, str)]
    
    return {"all_items": all_items, "aliases": aliases, "keys": alias_keys, "model": None, "embeddings": None}

def get_lazy_embedding_model(food_index):
    if food_index["model"] is None:
        print("Lazy Loading Semantic Model (CPU)...")
        food_index["model"] = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
        
        print(f"Encoding {len(food_index['keys'])} food aliases...")
        
        # 🔴 CRITICAL FIX BELOW: Add show_progress_bar=False
        food_index["embeddings"] = food_index["model"].encode(
            food_index["keys"], 
            convert_to_tensor=True, 
            show_progress_bar=False  # <--- THIS STOPS THE CRASH
        )
        
    return food_index["model"], food_index["embeddings"]




def resolve_food(query: str, food_index: Dict) -> Tuple[Optional[Dict], Optional[str], Dict]:
    # 1. Handle Empty/None Input safely
    if not query:
        return None, "empty_query", {}

    # 2. Define variables in order
    raw = query.strip()
    norm = normalize_arabic_light(raw).lower()

    # 3. Exact Match Search
    if norm in food_index["aliases"]:
        canon = food_index["aliases"][norm]
        for item in food_index["all_items"]:
            if item["item_ar"] == canon:
                return item, "exact", {}

    # 4. Fuzzy Match Search
    matches = process.extract(norm, food_index["keys"], scorer=fuzz.WRatio, limit=1)
    if matches and matches[0][1] > 95:
        canon = food_index["aliases"][matches[0][0]]
        for item in food_index["all_items"]:
            if item["item_ar"] == canon:
                return item, "fuzzy", {"score": matches[0][1]}

    # 5. Vector Search (Lazy Load)
    try:
        model, embeddings = get_lazy_embedding_model(food_index)
        query_emb = model.encode(norm, convert_to_tensor=True, show_progress_bar=False)

        scores = util.cos_sim(query_emb, embeddings)[0]
        best_score_idx = np.argmax(scores.cpu().numpy())
        best_score = scores[best_score_idx].item()

        if best_score > 0.75:
            best_alias = food_index["keys"][best_score_idx]
            canon = food_index["aliases"][best_alias]
            for item in food_index["all_items"]:
                if item["item_ar"] == canon:
                    return item, "vector", {"score": best_score, "matched_alias": best_alias}
        
        return None, "Not found", {"best_vector_score": best_score}

    except Exception as e:
        print(f"Vector Search Failed: {e}")
        return None, "Not found", {}


# -----------------------
# ENGINE
# -----------------------
class DietCheatEngine:
    def __init__(self):
        self.retriever = DietRetriever()
        self.food_index = load_food_index(FOOD_JSON_PATH)
        self.sessions = SessionStore()
        # Reuse the retriever's embedding model for the cache
        self.cache = SemanticCache(self.retriever.emb_fn._model) 

    def run_turn(self, user_msg: str, session_id: str = "default") -> Dict:
        req_id = str(uuid.uuid4())
        trace = TraceLogger(req_id)
        
        # 1. Get History
        history = self.sessions.get_history(session_id)
        
        state = {
            "req_id": req_id, "raw_query": user_msg, "normalized_query": user_msg,
            "history": history, "intent": "GENERAL_QA", "sources": [], "entities": {}
        }

        self._node_franco(state, trace)
        self._node_intent_router(state, trace)
        
        # 2. Check Cache
        # Need query embedding first. 
        # Since we use the retriever's model, we can compute it here or use what the retriever does?
        # The cache expects an embedding. Let's compute it using the model we passed to it.
        # But wait, self.retriever.emb_fn is a wrapper. 
        # Actually, self.retriever uses `SentenceTransformerEmbeddingFunction` which uses `SentenceTransformer`.
        # Accessing `_model` might be fragile if library changes, but `embedding_functions` from chroma usually has standard HF model.
        # Let's just re-encode using the cache's model reference.
        
        t0_cache = time.perf_counter()
        query_emb = self.cache.model.encode(state["normalized_query"], convert_to_tensor=True)
        cached_ans = self.cache.get(query_emb, state["intent"], state.get("entities", {}))
        
        if cached_ans:
             state["final_answer"] = cached_ans
             state["source"] = "cache"
             trace.log("cache_hit", t0_cache, 0)
        else:
            trace.log("cache_miss", t0_cache, 0)
            
            if state["intent"] == "MATH_SUBSTITUTION":
                self._node_math(state, trace)
                self._node_writer(state, trace)
            elif state["intent"] == "CHIT_CHAT":
                state["final_answer"] = "اهلا بيك يا بطل! اقدر اساعدك ازاي النهاردة في الدايت او التمرين؟"
            else:
                self._node_rag(state, trace)
                self._node_writer(state, trace)
                
            # 3. Add to Cache (if answer was generated)
            if state.get("final_answer") and not state.get("math_error"):
                 self.cache.add(query_emb, state["intent"], state.get("entities", {}), state["final_answer"])

        # 4. Update History
        self.sessions.add_turn(session_id, user_msg, state["final_answer"])
        
        trace.write_jsonl()
        return {
            "answer": state.get("final_answer"),
            "sources": state.get("sources", []),
            "debug_translation": state["normalized_query"],
            "session_id": session_id
        }

    def _node_franco(self, state, trace):
        t0 = time.perf_counter()
        sys = (
            "You are a professional Translator. Task: Convert Franco-Arabic (Latin script) to Egyptian Arabic Script.\n"
            "Rules:\n1. Output ONLY the Arabic text.\n2. Do NOT chat or explain."
        )
        res = hf_chat([{"role":"system","content":sys},{"role":"user","content":state["raw_query"]}])
        state["normalized_query"] = res if res else state["raw_query"]
        trace.log("franco_check", t0, 0, extra={"norm": state["normalized_query"]})

    def _node_intent_router(self, state, trace):
        t0 = time.perf_counter()
        query = state["normalized_query"]
        prompt = (
            f"You are a Router. Classify the user query into exactly one category.\n"
            f"CATEGORIES:\n1. MATH_SUBSTITUTION\n2. CHIT_CHAT\n3. GENERAL_QA\n"
            f"User Query: {query}\nINSTRUCTION: Reply with ONLY the category name."
        )
        res = gemini_chat_router(prompt)
        if "MATH" in res.upper(): state["intent"] = "MATH_SUBSTITUTION"
        elif "CHIT" in res.upper(): state["intent"] = "CHIT_CHAT"
        else: state["intent"] = "GENERAL_QA"
        trace.log("intent_router", t0, 0, extra={"raw_llm": res, "final": state["intent"]})

    def _node_rag(self, state, trace):
        t0 = time.perf_counter()
        allowed = ["nutrition_ebook", "training_guide"]
        results_norm = self.retriever.search(state["normalized_query"], k=3, where={"source": {"$in": allowed}})
        results_raw = []
        if state["normalized_query"] != state["raw_query"]:
             results_raw = self.retriever.search(state["raw_query"], k=2, where={"source": {"$in": allowed}})
        
        seen = set()
        merged = []
        for r in results_norm + results_raw:
            if r["text"] not in seen:
                merged.append(r)
                seen.add(r["text"])
        valid = merged[:3]
        state["chunks"] = valid
        state["sources"] = [{"source": r["meta"]["source"], "snippet": r["text"][:80]} for r in valid]
        trace.log("rag_search", t0, 0, extra={"count": len(valid)})

    def _node_math(self, state, trace):
        t0 = time.perf_counter()
        sys = ("Extract JSON: {\"source_food\": \"...\", \"source_weight_g\": \"...\", \"target_food\": \"...\"}\nIf weight missing, use 100.")
        res = hf_chat([{"role":"system","content":sys},{"role":"user","content":state["normalized_query"]}])
        
        slots = extract_json_from_text(res)
        
        # CRITICAL: Default to empty strings if keys are missing to prevent NoneType error in resolve_food
        # Existing code is good, but ensure this pattern is strictly followed:
        source_food = str(slots.get("source_food") or "")
        target_food = str(slots.get("target_food") or "")

        
        if not slots.get("source_weight_g"):
            state["math_error"] = "حدد الوزن بالجرام."
            trace.log("math_logic", t0, 0, extra={"error": "no_weight"})
            return

        # Pass safe strings to resolve_food
        src_item, method_s, meta_s = resolve_food(source_food, self.food_index)
        tgt_item, method_t, meta_t = resolve_food(target_food, self.food_index)
        if src_item and tgt_item:
            try:
                res_w = (float(slots["source_weight_g"]) / src_item["weight_g"]) * tgt_item["weight_g"]
                state["math_result"] = {
                    "user_w": slots["source_weight_g"], "src": src_item["item_ar"], 
                    "tgt": tgt_item["item_ar"], "res": res_w, "debug_methods": f"{method_s} -> {method_t}"
                }
            except: state["math_error"] = "خطأ في الحساب"
        else: state["math_error"] = "مش لاقي الاكل ده في القائمة"
        trace.log("math_logic", t0, 0, extra=slots)

    def _node_writer(self, state, trace):
        t0 = time.perf_counter()
        
        # 1. Handle Math/Error Cases
        if state.get("math_error"):
            state["final_answer"] = state["math_error"]
            return
        if state.get("math_result"):
            r = state["math_result"]
            state["final_answer"] = (
                f"✅ تمام يا بطل.\n"
                f"لو عايز تبدل *{r['user_w']}* جم {r['src']}، "
                f"تاخد مكانهم *{r['res']:.0f}* جم {r['tgt']}."
            )
            return

        context = "\n".join([f"- {c['text']}" for c in state.get("chunks", [])])
        
        # 2. System Prompt (STRICT SEPARATION)
        sys = (
            "You are 'Captain', a professional Nutrition Coach.\n"
            "Your task: Answer in EGYPTIAN Arabic suitable for WhatsApp.\n"
            "Be helpful and friendly"
            "CRITICAL FORMATTING RULES:\n"
            "1. The main answer must be 100% Arabic script. Transliterate English terms (e.g., write 'بنش برس' NOT 'Bench Press').\n"
            "2. Format:\n"
            "   🎯 [One sentence in pure Arabic]\n"
            "   \n"
           
        )
        
        msg = f"Context:\n{context}\n\nUser Question: {state['normalized_query']}" if context else f"User Question: {state['normalized_query']}"

        # 3. Call LLM
        state["final_answer"] = hf_chat([{"role": "system", "content": sys}, {"role": "user", "content": msg}])
        trace.log("writer", t0, 0)



_ENGINE = None
def run_turn(msg, session_id="default"):
    global _ENGINE
    if not _ENGINE: _ENGINE = DietCheatEngine()
    return _ENGINE.run_turn(msg, session_id)
