# Diet & Cheat RAG Chatbot Pipeline (Final)

**Project Goal:** Build a production-ready RAG system to answer Diet & Cheat moderator FAQs using 3 PDFs (nutrition ebook, training guide, food exchange list), with hybrid retrieval, caching, and safe routing to human moderators when needed.

**Core Principle:** Learn production RAG deeply—ingestion to evaluation to logging—while keeping costs near-zero (local LLM, AWS free tier, Cloudflare free tunnel).

---

## Project & Data

Your knowledge base is 3 PDFs with different "content shapes":
- **Nutrition ebook:** explanatory text + formulas/terms (TDEE, BMR, macros, amino acids, hormones, hydration).
- **Training guide:** training concepts and terms (RPE/RIR, 1RM, HIIT, Fartlek, deload, progressive overload).
- **Food Exchange List:** dense numeric tables (foods + calories/macros per portion).

### Main Challenges
1. Different document types (tables vs narrative text) require different parsing/chunking strategies.
2. Tables and numbers retrieve better via keyword/BM25 than embeddings alone.
3. User questions are mixed-language (Arabic + English + Franco/Arabizi), which breaks retrieval if ignored.
4. Safety-first: must refuse medical questions (disease, meds, pregnancy, injury, eating disorders).
5. Moderators expect Egyptian Arabic (not MSA), short answers, and clear citations.

### Evaluation Criteria (per pipeline part)

#### A) Document Parsing & Chunking
- **Goal:** Chunks are "self-contained units" (whole recipe, whole table block, coherent paragraph).
- **Validation:** Manual spot-check ~30 chunks; label "good/bad chunk boundary."
- **Failure signs:** recipe split mid-way, table rows separated from headers, mixed topics in one chunk.
- **Admin requirement:** Build chunk viewer UI to inspect extracted text → chunk boundaries before indexing.

#### B) Embeddings (baseline)
- **Goal:** Top retrieved chunks actually contain the answer.
- **Evaluation set:** Build 30–50 real moderator-style questions (Arabic/English/mixed/Franco).
- **Metrics:** 
  - Recall@k: does the correct chunk appear in top-k?
  - MRR: how high is the first correct chunk ranked?

#### C) Retrieval (hybrid)
- **Goal:** Don't miss answers (high recall).
- **Comparison:** Dense-only vs BM25-only vs Hybrid (dense+BM25).
- **Metric:** Recall@50 should improve with hybrid, especially for table/numeric queries.

#### D) Reranking (cross-encoder)
- **Goal:** Push best evidence to top (high precision).
- **Metric:** Precision@5 (how many of top-5 are actually relevant).
- **Human check:** For 20 questions, review top-5 ordering.

#### E) Generation (LLM answer)
- **Goal:** Moderators see correct, grounded answers.
- **Metrics:** RAGAS (faithfulness/relevancy) to catch hallucination or context mismatch.
- **Feature:** Internal CoT for model only (not shown to users) for multi-step calculations.

#### F) Semantic Caching
- **Goal:** Faster + cheaper without serving wrong answers.
- **Metrics:** 
  - Cache hit rate
  - Wrong-cache-hit rate (false positives)
- **Critical:** Must evaluate on near-duplicates (e.g., "substitute open-ended" vs "specific substitute yes/no").

#### G) Router (doc-type routing)
- **Goal:** Reduce search space and improve precision.
- **Metric:** Router accuracy on labeled set of ~50 queries ("nutrition vs training vs recipe vs table").

---

## Deployment Architecture

### Where things run:
- **AWS EC2:** RAG backend (always-on) + Postgres + pgvector
- **Your laptop:** Nile-Chat-4B LLM (local) + LLM gateway (FastAPI)
- **Cloudflare:** Tunnel to expose local gateway securely

### Connectivity:
- Cloudflare Tunnel creates outbound-only connection from your laptop to Cloudflare.
- AWS backend calls your gateway via tunnel HTTPS URL without opening home network ports.

---

## Final Pipeline (A→Z)

### 0) Inputs
**User message** (can be Arabic, English, or Franco/Arabizi).

#### Language Normalization
- Create `q_original` (raw user input).
- Create `q_normalized` (cheap normalization):
  - Keep English tokens as-is.
  - Normalize Arabic letters (أ/إ/ا → ا, ة/ه → ه, ي/ى → ي).
  - For Franco/Arabizi: basic rules-based transliteration to Arabic where possible, keep original too.
- Downstream retrieval uses **both** queries and merges results (critical for Franco recall).

---

### 1) Ingestion (Offline)
Use different chunking strategies (not fixed chunk size). Minimal metadata: only doc_type label.

#### 1.1 PDF Parsing
- Extract text from all 3 PDFs page-by-page.
- For pages with images/tables: attempt text extraction; note extraction quality in logs.
- **Admin requirement:** Persist raw extraction output so you can compare "PDF page → extracted text → chunk text" (critical for debugging Food Exchange List).

#### 1.2 Chunking (Structure-Aware)
Different strategy per document type:

**Nutrition ebook:** Chunk by headings/paragraph blocks (normal text chunking).
- Keep formulas with explanatory text.
- Preserve section structure (e.g., "BMR calculation" stays together).

**Training guide:** Chunk by topic blocks (RPE/RIR, HIIT, deload, cardio, recovery, etc.).
- Keep concept + examples together.

**Food Exchange List:** Chunk by table blocks / row groups.
- Keep header + rows together.
- Avoid splitting headers away from values.
- This is critical; naive chunking will destroy table coherence.

#### 1.3 Doc-Type Labeling (Minimal)
Label each chunk with a single tag:
```
doc_type ∈ {nutrition, training, table}
```

Store per chunk:
- `chunk_id`
- `doc_id`
- `doc_type`
- `page_start`, `page_end`
- `section_title` (if any)
- `text`
- `created_at`

---

### 2) Indexing (Offline)

#### 2.1 Dense Index (Postgres + pgvector)
**Embedding model:** Start with a multilingual baseline (e.g., Swan or Sentence-Transformers) that handles Arabic + English + mixed well.

**Why Swan/multilingual:** Your corpus is code-switched (Arabic + English terms like "BMR", "RPE", "macros"), so a pure Arabic embedding model may fail on Franco/mixed queries.

**Setup:**
- Enable pgvector in Postgres: `CREATE EXTENSION vector;`
- Compute embeddings for every chunk.
- Store in `chunks` table:
  - `embedding` as `vector(N)` column (N = embedding dimension, typically 384–768)
  - Create pgvector index (IVFFLAT or HNSW depending on corpus size; start simple for MVP)

#### 2.2 Sparse Index (BM25)
Build a BM25 index over chunk texts (Python-based).

**Implementation approach:**
- Option A (simplest): Compute BM25 index in Python (e.g., using `rank-bm25` library), serialize to disk (pickle/json).
- Option B (more production later): Use a Postgres BM25 extension (not required for month 1).

**Store:** `chunk_id → text` + BM25 scores on query time.

**Why BM25:** Better for exact matches, numbers, and food names (critical for Food Exchange List queries like "بدل" / substitution, "كام سعرات").

---

### 3) Query-Time Pipeline (Online)

#### Step 3.1: Query Router (Cheap Classifier)
**Input:** User query (normalized + original).  
**Output:** Predicted `doc_type` candidate(s).

Routing heuristics (can be keyword-based or a small classifier):
- Keywords: workout/RPE/RIR/cardio/training → `training`
- Keywords: macros/calories/protein/TDEE/BMR → `nutrition`
- Keywords: exchange list/portion/بدل/substitute → `table`
- Keywords: recipe/ingredients/how to make → `recipe` (optional; may map to nutrition)

**Soft routing:** Don't hard filter; if user asks multiple topics, include all relevant doc types.

#### Step 3.2: Intent + Entities Extraction (for Caching Safety)
Run a small intent classifier:
```
intent ∈ {
  substitution_open,        # "what can I eat instead?"
  substitution_specific,    # "can I substitute X with Y?"
  allowed_food_yesno,       # "is X allowed?"
  macros_calculation,       # "how much protein in 100g X?"
  recipe_request,           # "how to make X?"
  training_advice           # "should I do X?"
}
```

Extract entities (when present):
- `food_item_from`
- `food_item_to`
- Any numeric quantity mentioned

**Safety gate (belongs here logically):**
If query contains sensitive medical keywords (disease, medication, pregnancy, surgery, eating disorder) → **force route to moderator, skip generation**.

#### Step 3.3: Cache Lookup (Semantic, but Safe)
Semantic similarity cache lookup:
- Compute query embedding.
- Find cached queries within threshold (e.g., 0.85–0.90 cosine similarity).
- **Cache hit is valid only if:**
  - Intent matches
  - Entities match (when present)
  - Doc types compatible
- If all match → return cached final answer (fast).

**What cache stores per entry:**
```json
{
  "query_embedding": [...],
  "intent": "macros_calculation",
  "entities": {"food_item": "chicken", "quantity_g": 100},
  "doc_type_used": ["nutrition", "table"],
  "retrieved_chunk_ids": [5, 12, 23],
  "answer": "100g chicken breast has ~31g protein...",
  "citations": ["chunk_5", "chunk_12"],
  "timestamp": "2026-01-03T...",
  "ttl": 86400
}
```

This structure prevents "wrong-cache-hit" by validating semantic + intent + entities match.

#### Step 3.4: Hybrid Candidate Retrieval (only if cache miss)
Run both retrievers in parallel on the routed subset:

**Dense (pgvector):**
- Query with both `q_original` and `q_normalized`.
- Return top 50 chunks for each query.
- Merge deduplicate by `chunk_id`.

**Sparse (BM25):**
- Query with both `q_original` and `q_normalized`.
- Return top 50 chunks for each query.
- Merge deduplicate by `chunk_id`.

**Merge strategy:**
- Combine dense + sparse candidates.
- Deduplicate by `chunk_id`.
- Result: 80–120 candidate chunks total.

**Why hybrid:** Increases recall so you don't miss table/numeric answers (Food Exchange List queries) and exact matches (specific food names, technical terms).

#### Step 3.5: Cross-Encoder Reranking
Rerank merged candidates with a cross-encoder model.
- Keep top 5–10 chunks after reranking.
- **Metric:** Ensure Precision@5 is good (how many of top-5 are actually relevant).

#### Step 3.6: LLM Answer Generation (RAG)
**LLM runtime (local):**
- Nile-Chat-4B running locally via Ollama (GGUF import).
- **Important clarification:** Nile-Chat-4B is not in Ollama registry by default; run it by importing a GGUF quantization via `ollama import` or `ollama create`.

**Local LLM Gateway (your laptop):**
- FastAPI service running on `localhost:PORT`.
- Endpoint: `POST /v1/generate`
- Auth: Requires `X-API-KEY` header.
- Rate limiting: Basic in-memory token bucket.
- Calls Ollama `/api/generate` or `/api/chat` under the hood.
- Returns: JSON with `text` field.

**Cloudflare Tunnel (secure exposure):**
- Run `cloudflared` on your laptop as a systemd service (auto-restart on reboot).
- Exposes gateway to AWS backend via HTTPS URL (e.g., `https://llm-gateway-xxx.trycloudflare.com`).
- No public IP / port opening needed on home network.

**Prompt engineering:**
- Include: retrieved chunks, citations, safety rules.
- Enable internal CoT (chain-of-thought) for model only; don't show reasoning to user.
- Force short, moderator-friendly output in Egyptian Arabic (not MSA).
- Instruction: "Answer in Egyptian Arabic. If answer is not in the documents, refuse and ask moderator."

**AWS backend call to gateway:**
```python
# Pseudo-code
response = requests.post(
    "https://llm-gateway-xxx.trycloudflare.com/v1/generate",
    json={"prompt": grounded_prompt, "context": top_chunks},
    headers={"X-API-KEY": os.getenv("LLM_GATEWAY_KEY")}
)
answer_text = response.json()["text"]
```

#### Step 3.7: Evaluate + Log (Learning Loop)
Log everything for debugging + observability:
- Query: original + normalized
- Routed doc_type
- Retrieval results: dense scores, BM25 scores, merged candidates
- Reranking results: top-5 order
- Generation: final answer, internal CoT
- Execution time: retrieval latency, LLM latency

**Storage:** Postgres `queries_log` table.

**Admin views (critical for learning):**
- "Top unanswered / refused questions" → shows where gaps are.
- "Queries with low evidence score" → shows hard cases.
- "Most retrieved chunks" → shows which docs are most useful.

**Evaluation on eval set:**
- Run RAGAS (faithfulness, relevancy, answer relevancy) regularly.
- Track metrics per doc_type (nutrition recall, table recall, training recall).
- Track metrics per query intent (macros_calculation accuracy, substitution_open recall, etc.).

#### Step 3.8: Cache Write
If cache miss was resolved and answer generated:
```json
{
  "query_embedding": [computed above],
  "intent": [from Step 3.2],
  "entities": [from Step 3.2],
  "doc_type_used": [routed doc types],
  "retrieved_chunk_ids": [top-5 after reranking],
  "answer": [LLM output],
  "citations": [chunk IDs referenced in answer],
  "timestamp": now(),
  "ttl": 86400  // 24 hours
}
```

Store in Postgres `semantic_cache` table.

---

## Billing / "What's Free vs What Costs What"

### Free / Near-Zero (MVP)
- **LLM inference:** Free (runs on your laptop); costs are hardware + electricity.
- **Cloudflare Tunnel:** Free tier can handle demo/pilot traffic.
- **Postgres + pgvector:** Open-source software.
- **GitHub Actions (CI/CD):** Free for public repos.

### Where AWS Can Cost (be careful)
- **EC2:** Free Tier typically covers ~750 hours/month of t2.micro for 12 months. After that or if you upgrade: billing kicks in.
- **Public IPv4:** AWS Free Tier now includes 750 hours of free public IPv4 addresses. Beyond that: can be billed.
- **Storage/Bandwidth:** PDF uploads and logs growth can add small charges.
- **Data Transfer:** Minimal within AWS; egress to internet can be billed.

**Recommendation:** Set AWS budget alerts immediately to avoid surprises.

### CI/CD Costs
- **GitHub Actions:** 
  - Public repos: unlimited free minutes.
  - Private repos (free plan): 2,000 free minutes per month, then billed.

---

## Tech Stack (Summary)

| Component | Technology | Free/Cost | Notes |
|-----------|-----------|-----------|-------|
| Embedding Model | Sentence-Transformers (Swan/multilingual) | Free (OSS) | Handles Arabic+English+Franco |
| Dense Retrieval | Postgres + pgvector | Free (OSS) + AWS cost | pgvector is a Postgres extension |
| Sparse Retrieval | Python BM25 | Free (OSS) | In-app implementation |
| Reranker | HuggingFace cross-encoder | Free (OSS) | Optional early; add if needed |
| LLM | Nile-Chat-4B (GGUF) | Free (OSS) | Via Ollama import on laptop |
| LLM Gateway | FastAPI | Free (Python) | Your laptop |
| Gateway Exposure | Cloudflare Tunnel | Free (Free Tier) | For AWS ↔ laptop communication |
| RAG Backend | FastAPI | Free (Python) | Runs on AWS EC2 |
| Database | Postgres | Free (OSS) + EC2 cost | On EC2 |
| CI/CD | GitHub Actions | Free (public repo) | Deploy to EC2 |
| Hosting | AWS EC2 | Free (Free Tier, 1st 12mo) | After: ~$9–20/mo for t2.micro |

---

## Next Steps (Execution)

1. **Prepare evaluation set:** Collect 30–50 real moderator questions (Arabic/English/Franco) + label correct answer sources.
2. **Implement parsing:** Extract text from PDFs; create admin chunk viewer.
3. **Implement ingestion:** Build chunking pipeline (structure-aware per doc type).
4. **Benchmark embeddings:** Test multilingual embedding model (Swan) on eval set; measure Recall@k.
5. **Implement dense retrieval:** pgvector setup + indexing.
6. **Implement sparse retrieval:** Python BM25 index.
7. **Benchmark hybrid:** Measure Recall@50 for dense vs BM25 vs hybrid; validate hybrid wins on table slice.
8. **Implement reranking:** Add cross-encoder if Precision@5 is low.
9. **Implement safety + answerability:** Hard medical gate + confidence thresholding.
10. **Implement LLM gateway:** Local Nile-Chat + Ollama + FastAPI.
11. **Setup Cloudflare Tunnel:** Expose gateway securely.
12. **Deploy AWS backend:** EC2 + Nginx + systemd + Postgres.
13. **Setup CI/CD:** GitHub Actions → EC2 deploy.
14. **Implement caching:** Semantic cache with intent/entity validation.
15. **Setup logging + monitoring:** Admin views for debugging.
16. **Evaluate on full set:** RAGAS + human review.

---

## Key Learning Outcomes

By completing this pipeline, you will understand:
- **Production RAG:** Ingestion, chunking, hybrid retrieval, reranking, evaluation, caching.
- **Deployment:** EC2 + Nginx + systemd + CI/CD (no Docker, pure Linux).
- **Observability:** Logging, metrics, admin dashboards for ops.
- **Multi-language:** Query normalization, Franco/Arabizi handling.
- **Safety-first AI:** Medical gating, refusal mechanisms, grounding with citations.
- **Local LLM inference:** Ollama, GGUF quantization, secure tunneling.

This is **real production AI engineering**, not classroom RAG.

---

**Document version:** 1.0  
**Last updated:** 2026-01-03  
**Project:** Diet & Cheat RAG Chatbot  
**Status:** Ready for implementation


AT THE END:
The Deal:

Now (MVP): We build the full RAG pipeline (Ingestion -> Retrieval -> Generation) using ChromaDB. This gets your chatbot working tonight.

Later (Production Layer): Once the bot works, we add Postgres specifically to handle the "Semantic Cache" and "User Logs." This is the perfect use for Postgres because it teaches you how to manage structured data alongside vector data.