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

_lock = threading.Lock()


def _default_dir() -> Path:
    # Same directory as everything else the app stores (honours TRIAGEFLOW_HOME).
    from .config import settings
    return settings.home


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AuditLog:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else _default_dir()
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
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
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

    # ---- read-only views for the history / logs pages ---------------------------------------
    # All derived from the same append-only table; nothing here writes or mutates.

    @staticmethod
    def _complaint_from(payload: dict[str, Any]) -> str:
        """Best-effort presenting complaint out of whatever a payload carried."""
        enc = payload.get("encounter") if isinstance(payload.get("encounter"), dict) else None
        if enc and enc.get("presenting_complaint"):
            return str(enc["presenting_complaint"])
        return str(payload.get("complaint") or "")

    def _summary_from_trail(self, enc_id: str, trail: list[dict[str, Any]],
                            agg: dict[str, Any]) -> dict[str, Any]:
        complaint, proposed, approval, flags = "", None, None, 0
        for e in trail:
            p = e.get("payload") or {}
            if not complaint:
                complaint = self._complaint_from(p)
            if p.get("proposed_priority") and e["event"] != "approval":
                proposed = p["proposed_priority"]
            if e["event"] == "flag":
                flags += 1
            if e["event"] == "approval":                 # the latest one wins (sorted asc)
                approval = {
                    "confirmed_priority": p.get("confirmed_priority"),
                    "proposed_priority": p.get("proposed_priority"),
                    "decision": p.get("decision"),
                    "clinician": p.get("clinician") or e.get("actor"),
                    "reason": p.get("reason") or "",
                    "ts": e["ts"],
                }
        return {
            "encounter_id": enc_id,
            "first_ts": agg.get("first_ts"),
            "last_ts": agg.get("last_ts"),
            "events": agg.get("n"),
            "complaint": complaint,
            "proposed_priority": (approval or {}).get("proposed_priority") or proposed,
            "confirmed_priority": (approval or {}).get("confirmed_priority"),
            "clinician": (approval or {}).get("clinician"),
            "decision": (approval or {}).get("decision"),
            "reason": (approval or {}).get("reason") or "",
            "flags": flags,
            "confirmed": approval is not None,
        }

    def encounters(self, limit: int = 200) -> list[dict[str, Any]]:
        """One row per encounter ever seen, newest activity first — the patient-run history."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            groups = con.execute(
                "SELECT encounter_id, MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n "
                "FROM audit WHERE encounter_id IS NOT NULL AND encounter_id != '' "
                "GROUP BY encounter_id ORDER BY last_ts DESC LIMIT ?", (int(limit),)).fetchall()
            out = []
            for g in groups:
                rows = con.execute(
                    "SELECT * FROM audit WHERE encounter_id=? ORDER BY ts ASC",
                    (g["encounter_id"],)).fetchall()
                trail = []
                for r in rows:
                    d = dict(r)
                    d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
                    trail.append(d)
                out.append(self._summary_from_trail(g["encounter_id"], trail, dict(g)))
        return out

    def queue_history(self, limit: int = 500) -> list[dict[str, Any]]:
        """Every confirmed disposition (approval event), newest first — the queue history,
        separate from the live waiting queue and durable across restarts."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM audit WHERE event='approval' ORDER BY ts DESC LIMIT ?",
                (int(limit),)).fetchall()
        out = []
        for r in rows:
            p = json.loads(r["payload"]) if r["payload"] else {}
            out.append({
                "ts": r["ts"],
                "encounter_id": r["encounter_id"],
                "confirmed_priority": p.get("confirmed_priority"),
                "proposed_priority": p.get("proposed_priority"),
                "decision": p.get("decision"),
                "clinician": p.get("clinician") or r["actor"],
                "reason": p.get("reason") or "",
            })
        return out

    def recent_events(self, limit: int = 300,
                      encounter_id: str | None = None) -> list[dict[str, Any]]:
        """The raw append-only log, newest first — optionally filtered to one encounter."""
        q = "SELECT * FROM audit"
        args: list[Any] = []
        if encounter_id:
            q += " WHERE encounter_id=?"
            args.append(encounter_id)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(q, tuple(args)).fetchall()
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
