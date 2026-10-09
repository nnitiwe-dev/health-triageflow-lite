"""Append-only audit log — the backbone of defensibility.

Every tool call (inputs, outputs, timestamp, errors) and every irreversible action
(a proposed priority reaching the queue, an approval/override) is recorded here.
Writes go to a JSONL file (human-readable, greppable) AND a SQLite table (queryable,
and exposed to inspection). The log is append-only: we never update or delete rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path.home() / ".triageflow"
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AuditLog:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.base_dir / "audit.jsonl"
        self.db_path = self.base_dir / "triageflow.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS audit (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    encounter_id TEXT,
                    event TEXT NOT NULL,        -- tool_call | tool_result | tool_error | approval | flag
                    actor TEXT,                 -- agent | clinician name | system
                    tool TEXT,
                    payload TEXT                -- JSON
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_audit_enc ON audit(encounter_id)")

    def record(self, event: str, *, encounter_id: str | None = None,
               actor: str = "agent", tool: str | None = None,
               payload: dict[str, Any] | None = None) -> str:
        entry = {
            "id": str(uuid.uuid4()),
            "ts": _now(),
            "encounter_id": encounter_id,
            "event": event,
            "actor": actor,
            "tool": tool,
            "payload": payload or {},
        }
        with _lock:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            with sqlite3.connect(self.db_path) as con:
                con.execute(
                    "INSERT INTO audit (id, ts, encounter_id, event, actor, tool, payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (entry["id"], entry["ts"], entry["encounter_id"], entry["event"],
                     entry["actor"], entry["tool"], json.dumps(entry["payload"], default=str)),
                )
        return entry["id"]

    def trail(self, encounter_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM audit WHERE encounter_id=? ORDER BY ts ASC", (encounter_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
            out.append(d)
        return out


# Module-level default instance for convenience.
_default: AuditLog | None = None


def get_audit(base_dir: Path | None = None) -> AuditLog:
    global _default
    if _default is None or base_dir is not None:
        _default = AuditLog(base_dir)
    return _default
