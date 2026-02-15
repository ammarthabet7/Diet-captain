import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
from typing import Any, Dict, List, Optional
from config import (
    CHROMA_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME, 
    RERANKER_MODEL_NAME
)


class DietRetriever:
    def __init__(self, debug: bool = False):
        self.debug = debug

        # 1) Setup Chroma
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.emb_fn
        )

        # 2) Pull full corpus for BM25
        data = self.collection.get()
        self.documents: List[str] = data.get("documents", []) or []
        self.metadatas: List[Dict[str, Any]] = data.get("metadatas", []) or []

        # 2.1) Tokenize corpus for BM25
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3) Setup reranker
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        if self.debug:
            print(f"Retriever ready. Corpus size = {len(self.documents)}")

    def _tokenize(self, text: str) -> List[str]:
        # Light normalization
        text = re.sub(r"[أإآ]", "ا", text)
        text = text.replace("ال", "")
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.lower().split()

        # Character trigrams + full words
        ngrams = []
        for w in words:
            ngrams.append(w)
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    ngrams.append(w[i:i+3])
        return ngrams

    def _allowed_indices_from_where(self, where: Optional[Dict[str, Any]]) -> Optional[List[int]]:
        """
        Currently supports your usage:
          where = {"source": {"$in": ["nutrition_ebook", "training_guide"]}}
        Returns list of document indices allowed for BM25 filtering.
        """
        if not where:
            return None

        # simple path: filter on meta['source'] using $in
        if "source" in where and isinstance(where["source"], dict) and "$in" in where["source"]:
            allowed = set(where["source"]["$in"])
            idxs = [i for i, m in enumerate(self.metadatas) if (m or {}).get("source") in allowed]
            return idxs

        # fallback: no filtering if unsupported
        return None

    def search(self, query: str, k: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        CANDIDATES_K = 15

        # A) Vector candidates (with metadata filter)
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=CANDIDATES_K,
            where=where  # Chroma supports where filters like $in. [web:31]
        )

        # B) BM25 candidates (respect metadata filter manually)
        allowed_idxs = self._allowed_indices_from_where(where)

        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        if allowed_idxs is None:
            candidate_pool = range(len(bm25_scores))
        else:
            candidate_pool = allowed_idxs

        top_indices = sorted(candidate_pool, key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATES_K]

        # C) Merge & deduplicate by text
        unique_docs: Dict[str, Dict[str, Any]] = {}

        # from vector
        docs0 = (vector_results.get("documents") or [[]])[0]
        metas0 = (vector_results.get("metadatas") or [[]])[0]
        if docs0:
            for i, doc in enumerate(docs0):
                meta = metas0[i] if i < len(metas0) else {}
                unique_docs[doc] = meta

        # from bm25
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                doc = self.documents[idx]
                unique_docs[doc] = self.metadatas[idx]

        candidates = list(unique_docs.keys())
        if not candidates:
            return []

        if self.debug:
            print("--- RETRIEVER DEBUG ---")
            print("Query:", query)
            print("Where:", where)
            print("Candidates:", len(candidates))

        # D) Rerank
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked_results = []
        for score, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True):
            ranked_results.append({
                "text": doc,
                "meta": unique_docs.get(doc, {}) or {},
                "score": float(score),
                "source": "hybrid-reranked"
            })

        return ranked_results[:k]


if __name__ == "__main__":
    retriever = DietRetriever(debug=True)
    test_q = "ya3ni eh TEF?"
    print("Searching:", test_q)
    results = retriever.search(test_q, k=3, where={"source": {"$in": ["nutrition_ebook"]}})
    for i, r in enumerate(results):
        print(f"#{i+1} score={r['score']:.4f} source={(r['meta'] or {}).get('source')} text={r['text'][:80]}...")
