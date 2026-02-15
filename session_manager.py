import json
import time
import torch
from typing import Dict, List, Optional, Any
from sentence_transformers import util
import numpy as np
from config import CACHE_SIMILARITY_THRESHOLD

class TraceLogger:
    """Mock TraceLogger if not passed from engine (for isolated testing)"""
    def log(self, node, t0, m0, extra=None): pass

class SemanticCache:
    def __init__(self, embedding_model):
        self.cache = [] # List of {embedding, intent, entities, answer, timestamp}
        self.model = embedding_model # Use the same model as RAG

    def get(self, query_emb, intent: str, entities: Dict[str, Any]) -> Optional[str]:
        if not self.cache:
            return None
        
        # 1. Intent Match (Hard Filter)
        candidates = [c for c in self.cache if c["intent"] == intent]
        if not candidates:
            return None

        # 2. Entity Match (Hard Filter) 
        # Only if the query has entities. If no entities, skip this check.
        # This is a simple equality check. logic can be improved.
        if entities:
             candidates = [c for c in candidates if c["entities"] == entities]
        if not candidates:
             return None

        # 3. Semantic Simularity (Vector Search)
        # We need to stack candidate embeddings first if they are individual tensors
        candidate_embs = [c["embedding"] for c in candidates]
        import torch
        if not candidate_embs:
            return None

        # Convert list of tensors to a single stacked tensor
        # Assuming all embeddings have the same shape
        try:
            candidates_tensor = torch.stack(candidate_embs)
        except Exception:
             # Fallback if shapes mismatch or other tensor issues
             return None

        # util.cos_sim returns a tensor [[score1, score2, ...]]
        # query_emb should be 1D or 2D (1, dim)
        
        scores = util.cos_sim(query_emb, candidates_tensor)[0]
        
        # Ensure we're working with CPU for numpy conversion
        scores_cpu = scores.cpu()
        best_idx = np.argmax(scores_cpu.numpy())
        best_score = scores_cpu[best_idx].item()

        if best_score >= CACHE_SIMILARITY_THRESHOLD:
            return candidates[best_idx]["answer"]
        
        return None

    def add(self, query_emb, intent, entities, answer):
        self.cache.append({
            "embedding": query_emb,
            "intent": intent,
            "entities": entities,
            "answer": answer,
            "timestamp": time.time()
        })
        # Basic cleanup: Keep last 1000
        if len(self.cache) > 1000:
            self.cache.pop(0)

class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {} # session_id -> history list

    def get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def add_turn(self, session_id: str, user_msg: str, ai_msg: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({"role": "user", "content": user_msg})
        self.sessions[session_id].append({"role": "assistant", "content": ai_msg})
        
        # Keep last 10 turns (20 messages)
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
