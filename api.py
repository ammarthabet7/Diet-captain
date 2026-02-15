from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
import uuid
from chat import run_turn

app = FastAPI(title="Diet & Cheat API", version="1.0")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[Dict] = []
    debug_translation: Optional[str] = None

# Stats (Simple in-memory)
STATS = {"requests": 0, "start_time": time.time()}

@app.get("/health")
def health_check():
    return {"status": "ok", "uptime": time.time() - STATS["start_time"]}

@app.get("/stats")
def get_stats():
    return STATS

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        STATS["requests"] += 1
        
        # Use provided session_id or generate new one
        sid = req.session_id or str(uuid.uuid4())
        
        result = run_turn(req.message, sid)
        
        return ChatResponse(
            answer=result["answer"],
            session_id=sid,
            sources=result.get("sources", []),
            debug_translation=result.get("debug_translation")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
