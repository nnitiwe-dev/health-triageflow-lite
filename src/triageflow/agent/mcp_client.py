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

import asyncio
import json
import os
import shutil
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
    """Lazy wrapper around MultiServerMCPClient; indexes tools by name.

    Our own server is required and connects first (fast, local). The borrowed filesystem
    server is optional: it is fetched by ``npx`` and may be slow or unreachable on a clinic
    connection, so it has a timeout and can be connected in the background — the app must
    never wait on the network before it can triage a patient.
    """

    def __init__(self, include_filesystem: bool = True, filesystem_timeout: float = 45.0):
        self.include_filesystem = include_filesystem
        self.filesystem_timeout = filesystem_timeout
        self._tools: dict[str, Any] = {}
        self._protocol_cache: dict[str, str] = {}
        self.connected_servers: list[str] = []
        self.protocol_source = "local file"

    async def connect(self, wait_for_filesystem: bool = True) -> dict[str, Any]:
        """Connect our own server; then the borrowed one (awaited unless told otherwise —
        pass ``wait_for_filesystem=False`` and schedule ``connect_filesystem()`` yourself)."""
        own = MultiServerMCPClient({"triageflow": _connections()["triageflow"]})
        tools = await own.get_tools()
        self._tools = {t.name: t for t in tools}
        self.connected_servers = ["triageflow"]
        if self.include_filesystem and wait_for_filesystem:
            await self.connect_filesystem()
        return self._tools

    async def connect_filesystem(self) -> bool:
        """Attach the borrowed filesystem MCP. Returns False (never raises) if Node/npx is
        missing, the package can't be fetched, or it doesn't answer within the timeout."""
        if "filesystem" in self.connected_servers:
            return True
        if shutil.which("npx") is None:
            return False
        try:
            fs = MultiServerMCPClient({"filesystem": _connections()["filesystem"]})
            tools = await asyncio.wait_for(fs.get_tools(), timeout=self.filesystem_timeout)
        except Exception:
            return False
        self._tools.update({t.name: t for t in tools})
        self.connected_servers.append("filesystem")
        return True

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
        falling back to a direct read so evidence retrieval never hard-fails. The text is
        static, so a successful MCP read is cached for the life of the process."""
        if filename in self._protocol_cache:
            self.protocol_source = "filesystem (borrowed MCP, cached)"
            return self._protocol_cache[filename]
        path = str(settings.protocols_dir / filename)
        for tool_name in ("read_text_file", "read_file"):
            if self.has(tool_name):
                try:
                    res = await asyncio.wait_for(self.call(tool_name, {"path": path}), timeout=20)
                    text = res if isinstance(res, str) else str(parse_tool_result(res))
                    if text.strip():
                        self._protocol_cache[filename] = text
                        self.protocol_source = "filesystem (borrowed MCP)"
                        return text
                except Exception:
                    continue
        self.protocol_source = "local file (fallback)"
        return read_protocol_file(filename)


def read_protocol_file(filename: str = "sats_adult.md") -> str:
    """Direct, UTF-8 read of a protocol file ('' if unreadable). UTF-8 is explicit because
    the Windows default codepage cannot decode the protocol text."""
    try:
        return (settings.protocols_dir / filename).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


class InProcessMCP:
    """The same tool surface as ``TriageMCP`` but calling our server's tool functions directly
    — no subprocess. For unit tests and the eval harness, so the full agent flow (validation →
    scoring → approval gate) can be exercised quickly and offline. The app itself always uses
    the real MCP transport."""

    connected_servers = ["triageflow (in-process)"]
    protocol_source = "local file"

    def __init__(self) -> None:
        from ..mcp import server
        self._fns = {name: getattr(server, name) for name in (
            "validate_triage_input", "score_triage", "check_deterioration",
            "request_clinician_approval", "suggest_discriminators", "fetch_fhir_vitals")}

    @property
    def tools(self) -> dict[str, Any]:
        return self._fns

    def has(self, name: str) -> bool:
        return name in self._fns

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        return self._fns[name](**args)

    async def call_json(self, name: str, args: dict[str, Any]) -> Any:
        return await self.call(name, args)

    async def read_protocol(self, filename: str = "sats_adult.md") -> str:
        return read_protocol_file(filename)
