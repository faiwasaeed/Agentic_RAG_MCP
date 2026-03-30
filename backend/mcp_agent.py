"""
mcp_agent.py – Orchestrator agent that calls MCP tools in sequence

Flow for each user query:
  1. validate_query   → check scope
  2. retrieve_context → get relevant chunks (expand + rerank)
  3. generate_answer  → LLM grounded response
  4. Return structured result

This keeps the agent logic thin and tool logic inside rag_tools.py,
mirroring the MCP separation of concerns.
"""

import json
import logging
import time
from typing import Optional

from mcp_server import MCPServer, MCPTool
from rag_tools import (
    tool_validate_query,
    tool_retrieve_context,
    tool_generate_answer,
    tool_list_departments,
    _load_resources,
)

log = logging.getLogger(__name__)


def build_mcp_server() -> MCPServer:
    """Register all RAG tools with the MCP server and return it."""
    server = MCPServer(name="UET-RAG-MCP-Server")

    server.register_tool(MCPTool(
        name="validate_query",
        description="Check whether a user query is within the UET prospectus scope before processing.",
        parameters={"query": {"type": "string", "description": "User's question"}},
        handler=tool_validate_query,
    ))

    server.register_tool(MCPTool(
        name="retrieve_context",
        description=(
            "Retrieve the most relevant chunks from the UET prospectus vector database "
            "using multi-query expansion + MMR + reranking."
        ),
        parameters={
            "query":      {"type": "string",  "description": "User's question"},
            "department": {"type": "string",  "description": "Optional department filter (empty string = all)"},
        },
        handler=tool_retrieve_context,
    ))

    server.register_tool(MCPTool(
        name="generate_answer",
        description="Generate a grounded answer from retrieved context chunks using a local LLM.",
        parameters={
            "query":          {"type": "string", "description": "User's original question"},
            "context_chunks": {"type": "array",  "description": "Chunks returned by retrieve_context"},
        },
        handler=tool_generate_answer,
    ))

    server.register_tool(MCPTool(
        name="list_departments",
        description="Return the list of UET departments covered by this system.",
        parameters={"query": {"type": "string", "description": "Ignored – pass empty string"}},
        handler=tool_list_departments,
    ))

    return server


class UETMCPAgent:
    """
    High-level agent that orchestrates MCP tool calls.
    Preloads resources on __init__ to avoid per-request cold-start latency.
    """

    def __init__(self):
        log.info("[Agent] Initialising UET MCP Agent …")
        self.server = build_mcp_server()
        # Pre-load embedding model + ChromaDB + LLM so first request is fast
        try:
            _load_resources()
            log.info("[Agent] Resources pre-loaded ✓")
        except Exception as e:
            log.warning(f"[Agent] Resource pre-load failed (will retry on first query): {e}")

    # ── Internal tool call helper ─────────────────────────────────────────────

    def _call(self, tool_name: str, **kwargs) -> dict:
        result = self.server.call_tool(tool_name, kwargs)
        if result["isError"]:
            raise RuntimeError(result["content"][0]["text"])
        return json.loads(result["content"][0]["text"])

    # ── Main entry point ──────────────────────────────────────────────────────

    def answer(self, query: str, department_filter: str = "") -> dict:
        """
        Process a user query end-to-end via MCP tool chain.
        Returns a dict with keys: answer, sources, processing_time_s, is_valid, tools_used.
        """
        t_start = time.perf_counter()
        tools_used: list[str] = []

        # ── Step 1: Validate scope ────────────────────────────────────────────
        tools_used.append("validate_query")
        validation = self._call("validate_query", query=query)
        if not validation["is_valid"]:
            return {
                "answer":            validation["reason"],
                "sources":           [],
                "processing_time_s": round(time.perf_counter() - t_start, 2),
                "is_valid":          False,
                "tools_used":        tools_used,
            }

        # ── Step 2: Retrieve context ──────────────────────────────────────────
        tools_used.append("retrieve_context")
        retrieval = self._call("retrieve_context", query=query, department=department_filter)
        chunks    = retrieval.get("chunks", [])
        log.info(f"[Agent] Retrieved {len(chunks)} chunks")

        # ── Step 3: Generate grounded answer ─────────────────────────────────
        tools_used.append("generate_answer")
        generation = self._call("generate_answer", query=query, context_chunks=chunks)

        total_time = round(time.perf_counter() - t_start, 2)
        return {
            "answer":            generation["answer"],
            "sources":           generation.get("sources", []),
            "processing_time_s": total_time,
            "llm_time_s":        generation.get("llm_time_s", 0),
            "is_valid":          True,
            "tools_used":        tools_used,
            "num_chunks":        len(chunks),
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    agent = UETMCPAgent()

    test_queries = [
        "Who is the chairperson of the Computer Science department?",
        "What programs are offered by the Electrical Engineering department?",
        "What are the admission requirements for Software Engineering?",
        "What is the fee structure for BS programs?",
        "What lab facilities are available in Mechanical Engineering?",
        "What is the capital of France?",          # out-of-scope
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        result = agent.answer(q)
        print(f"Valid: {result['is_valid']}  |  Time: {result['processing_time_s']}s  |  Tools: {result['tools_used']}")
        print(f"A: {result['answer'][:400]}")
        if result.get("sources"):
            print(f"Sources: {result['sources']}")
