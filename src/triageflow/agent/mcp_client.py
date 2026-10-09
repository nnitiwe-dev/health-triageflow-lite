"""Connect the agent to both MCP servers via the standard client.

  * ``triageflow``  — our own server (the hands): validate / score / deterioration / approval
  * ``filesystem``  — the borrowed official server (@modelcontextprotocol/server-filesystem),
                      read-only over the protocols/ dir, used to fetch the cited SATS text.

Why borrow the filesystem server: sandboxed, path-traversal-safe file access is a solved
problem — reusing the reference implementation is safer and faster than writing our own.

Both are spoken to through ``langchain-mcp-adapters`` so the exact same client code would
work against any other MCP server a facility already runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..core.config import settings


def parse_tool_result(res: Any) -> Any:
    """MCP tools return text content blocks; normalise to a Python object/string."""
    text = None
    if isinstance(res, str):
        text = res
    elif isinstance(res, list) and res:
        first = res[0]
        text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
    elif isinstance(res, dict):
        return res
    if text is None:
        return res
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

_SRC = str(Path(__file__).resolve().parents[2])  # .../src


def _connections() -> dict[str, dict[str, Any]]:
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return {
        "triageflow": {
            "command": sys.executable,
            "args": ["-m", "triageflow.mcp.server"],
            "transport": "stdio",
            "env": env,
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem",
                     str(settings.protocols_dir)],
            "transport": "stdio",
        },
    }


class TriageMCP:
    """Lazy wrapper around MultiServerMCPClient; indexes tools by name."""

    def __init__(self, include_filesystem: bool = True):
        self.include_filesystem = include_filesystem
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, Any] = {}
        self.connected_servers: list[str] = []

    async def connect(self) -> dict[str, Any]:
        conns = _connections()
        if not self.include_filesystem:
            conns.pop("filesystem", None)
        self._client = MultiServerMCPClient(conns)
        try:
            tools = await self._client.get_tools()
        except Exception:
            # Fall back to own-server only if the borrowed server can't start (e.g. no npx/net).
            conns.pop("filesystem", None)
            self._client = MultiServerMCPClient(conns)
            tools = await self._client.get_tools()
        self._tools = {t.name: t for t in tools}
        self.connected_servers = list(conns.keys())
        return self._tools

    @property
    def tools(self) -> dict[str, Any]:
        return self._tools

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"MCP tool not available: {name} (have: {list(self._tools)})")
        return await self._tools[name].ainvoke(args)

    async def call_json(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool and parse its content blocks into a Python object."""
        return parse_tool_result(await self.call(name, args))

    async def read_protocol(self, filename: str = "sats_adult.md") -> str:
        """Fetch protocol text via the borrowed filesystem MCP (tool name varies by version),
        falling back to a direct read so evidence retrieval never hard-fails."""
        path = str(settings.protocols_dir / filename)
        for tool_name, arg in (("read_text_file", {"path": path}),
                               ("read_file", {"path": path})):
            if self.has(tool_name):
                try:
                    res = await self.call(tool_name, arg)
                    return res if isinstance(res, str) else str(res)
                except Exception:
                    continue
        try:
            return Path(path).read_text()
        except OSError:
            return ""
