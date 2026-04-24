"""
main.py  —  Metasys-Chat Milestone 1

Start the server from the project root:
    uvicorn backend.main:app --reload --port 8000

Routes:
    GET  /         serve the dark chat UI
    POST /chat     SSE streaming RAG response
    POST /ingest   re-ingest openapi.json into ChromaDB
    GET  /health   quick readiness check
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from config import CHROMA_PATH, COLLECTION_NAME
from ingest import run_ingest
from rag import query_stream, _reset_collection_cache

app = FastAPI(title="metasys-chat", docs_url=None, redoc_url=None)

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    return StreamingResponse(
        query_stream(req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest():
    try:
        result = run_ingest()
        _reset_collection_cache()   # next /chat picks up fresh collection
        return {
            "status":  "success",
            "message": f"Ingested {result['chunks']} chunks into '{result['collection']}'",
            **result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    db_file = Path(CHROMA_PATH) / "chroma.sqlite3"
    return {
        "status":       "ok",
        "chroma_ready": db_file.exists(),
        "collection":   COLLECTION_NAME,
    }
