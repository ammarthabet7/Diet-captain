<<<<<<< HEAD
# Diet & Cheat - AI Nutrition Coach

A RAG-powered chatbot helping moderators answer fitness & nutrition questions with scientifically backed sources (Nutrition Ebook, Training Guide, Food Exchange List).

## Features
- **Hybrid Retrieval:** Combined Vector Search (Semantic) + BM25 (Keyword) for high accuracy.
- **Franco-Arabic Support:** Understands "3amel eh ya coach" and converts it to Arabic.
- **Scientific RAG:** Retrieves answers only from trusted PDF sources.
- **Math Substitution:** Calculates food exchanges (e.g., "Substitute 100g Rice with Potato").
- **Semantic Caching:** Caches similar questions to reduce latency and cost.
- **API First:** Ready for WhatsApp/Web integration via FastAPI.

## Project Structure
```
diet-cheat/
├── api.py              # FastAPI application
├── chat.py             # Main Logic Engine (RAG pipeline)
├── config.py           # Configuration constants
├── session_manager.py  # History & Caching logic
├── retriever.py        # Search system (ChromaDB + BM25)
├── scripts/            # Setup scripts (ingestion, cleaning)
├── tests/              # Unit tests
├── chroma_db/          # Vector Database
└── logs/               # Execution traces
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```ini
   GEMINI_API_KEY=your_gemini_key
   HF_TOKEN=your_huggingface_token
   ```

3. **Ingest Data** (If running for the first time)
   ```bash
   python scripts/ingest_chroma.py
   ```

4. **Run API**
   ```bash
   uvicorn api:app --reload
   ```
   Server will start at `http://localhost:8000`.

## API Usage

**POST /chat**
```json
{
  "message": "بديل 100 جرام رز؟",
  "session_id": "user123"
}
```

**Response**
```json
{
  "answer": "✅ تمام يا بطل. لو عايز تبدل 100 جم رز...",
  "session_id": "user123",
  "sources": [{"source": "food_exchange", "snippet": "..."}]
}
```

## Testing
Run unit tests:
```bash
pytest tests/
```
=======
# Diet-captain
>>>>>>> e3422ef2b818207b675e51eeea99038b351e458d
