"""SQLite persistence for firewall verdicts (SOC / audit trail)."""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config as cfg

DB_PATH = getattr(cfg, "ALERTS_DB", os.path.join(cfg.BASE_DIR, "data", "alerts.db"))


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                action      TEXT NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                class_id    INTEGER,
                tiers_used  TEXT NOT NULL,
                probabilities TEXT,
                tier_trace  TEXT
            )
        """)
        try:
            conn.execute("ALTER TABLE alerts ADD COLUMN tier_trace TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE alerts ADD COLUMN latency_ms REAL")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_action ON alerts(action)"
        )


def log_alert(result: Dict[str, Any]) -> int:
    for attempt in range(3):
        try:
            with _connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO alerts
                        (created_at, action, label, confidence, class_id, tiers_used,
                         probabilities, tier_trace, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        result["action"],
                        result["label"],
                        float(result["confidence"]),
                        result.get("class_id"),
                        json.dumps(result.get("tiers_used", [])),
                        json.dumps(result.get("probabilities")) if result.get("probabilities") else None,
                        json.dumps(result.get("tier_trace")) if result.get("tier_trace") else None,
                        result.get("latency_ms"),
                    ),
                )
                return int(cur.lastrowid)
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_alerts(
    limit: int = 50,
    action: Optional[str] = None,
) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 500))
    q = "SELECT * FROM alerts"
    params: List[Any] = []
    if action:
        q += " WHERE action = ?"
        params.append(action.upper())
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(q, params).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "action": r["action"],
            "label": r["label"],
            "confidence": r["confidence"],
            "class_id": r["class_id"],
            "tiers_used": json.loads(r["tiers_used"]),
            "probabilities": json.loads(r["probabilities"]) if r["probabilities"] else None,
            "tier_trace": json.loads(r["tier_trace"]) if r["tier_trace"] else None,
            "latency_ms": r["latency_ms"] if "latency_ms" in r.keys() else None,
        })
    return out


def alert_counts() -> Dict[str, int]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT action, COUNT(*) AS n FROM alerts GROUP BY action"
        ).fetchall()
    return {r["action"]: r["n"] for r in rows}
