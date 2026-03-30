"""
main.py – FastAPI backend server
Exposes /chat, /health, /tools, /departments endpoints
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Global agent instance (initialised during startup) ────────────────────────
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    log.info("🚀  Booting UET MCP-RAG API …")
    from mcp_agent import UETMCPAgent
    agent = UETMCPAgent()
    log.info("✅  Agent ready – API accepting requests")
    yield
    log.info("🛑  Shutting down")


app = FastAPI(
    title="UET MCP-RAG API",
    description=(
        "Accurate, fast RAG over the UET prospectus via an MCP-based AI agent. "
        "Combines query expansion + reranking for accuracy with pre-loaded resources for speed."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str         = Field(..., min_length=1, description="User's question")
    session_id: str         = Field(default_factory=lambda: str(uuid.uuid4()))
    department: Optional[str] = Field(default="", description="Optional department filter")


class ChatResponse(BaseModel):
    answer:            str
    sources:           list[str]
    is_valid:          bool
    tools_used:        list[str]
    num_chunks:        int
    processing_time_s: float
    llm_time_s:        float
    session_id:        str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":     "UET MCP-RAG API",
        "version":     "3.0.0",
        "description": "Accurate + fast UET prospectus Q&A via MCP agent",
        "endpoints": {
            "POST /chat":       "Ask a question about UET",
            "GET  /health":     "Health check",
            "GET  /tools":      "List registered MCP tools",
            "GET  /departments":"List covered departments",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "agent_ready": agent is not None, "timestamp": time.time()}


@app.get("/tools")
def list_tools():
    if agent is None:
        raise HTTPException(503, "Agent not ready")
    return {"tools": agent.server.list_tools()}


@app.get("/departments")
def list_departments():
    if agent is None:
        raise HTTPException(503, "Agent not ready")
    result = agent._call("list_departments", query="")
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if agent is None:
        raise HTTPException(503, "Agent is initialising, please retry in a moment.")
    if not req.message.strip():
        raise HTTPException(422, "Message cannot be empty")

    result = agent.answer(query=req.message, department_filter=req.department or "")
    return ChatResponse(
        answer            = result["answer"],
        sources           = result.get("sources", []),
        is_valid          = result.get("is_valid", True),
        tools_used        = result.get("tools_used", []),
        num_chunks        = result.get("num_chunks", 0),
        processing_time_s = result.get("processing_time_s", 0),
        llm_time_s        = result.get("llm_time_s", 0),
        session_id        = req.session_id,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
