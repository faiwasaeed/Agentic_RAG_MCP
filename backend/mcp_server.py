"""
UET MCP Server - Exposes RAG tools via Model Context Protocol
Combines accurate RAG from UET_RAG_System with MCP speed from AI_Agent_MCP
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ─── MCP Tool Registry ────────────────────────────────────────────────────────

class MCPTool:
    """Represents a single MCP-registered tool."""
    def __init__(self, name: str, description: str, parameters: dict, handler):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }


class MCPServer:
    """
    Lightweight synchronous MCP server.
    Tools are registered and dispatched in-process (stdio/HTTP transport optional).
    """

    def __init__(self, name: str = "UET-RAG-MCP-Server"):
        self.name = name
        self._tools: dict[str, MCPTool] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_tool(self, tool: MCPTool):
        self._tools[tool.name] = tool
        logger.info(f"[MCP] Registered tool: {tool.name}")

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name not in self._tools:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool '{tool_name}' not found."}],
            }
        tool = self._tools[tool_name]
        t0 = time.perf_counter()
        try:
            result = tool.handler(**arguments)
            elapsed = round(time.perf_counter() - t0, 3)
            return {
                "isError": False,
                "elapsed_ms": int(elapsed * 1000),
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            }
        except Exception as exc:
            logger.exception(f"[MCP] Tool '{tool_name}' raised an exception")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {exc}"}],
            }

    # ── MCP JSON-RPC handler (used when running over stdio/HTTP) ─────────────

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "tools/list":
            return {"id": req_id, "result": {"tools": self.list_tools()}}

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            return {"id": req_id, "result": result}

        return {"id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
