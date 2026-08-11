"""Automatic trial logging -- your DSR trial count N (§5.4).

"Self-reported N is always low by 5-10x." Every model fit should be logged (config hash,
CV Sharpe, timestamp) to a durable store so the Deflated Sharpe uses the *honest* trial
count. We use a plain SQLite table (the report's minimal recommendation) -- no MLflow
dependency required.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class TrialLog:
    """Append-only SQLite log of every configuration evaluated (§5.4)."""

    def __init__(self, path: str | Path = "trials.sqlite") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS trials ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts REAL, config_hash TEXT, cv_sharpe REAL, note TEXT, config_json TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def config_hash(config: dict) -> str:
        blob = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def log(self, config: dict, cv_sharpe: float, note: str = "") -> str:
        h = self.config_hash(config)
        self._conn.execute(
            "INSERT INTO trials (ts, config_hash, cv_sharpe, note, config_json) VALUES (?,?,?,?,?)",
            (time.time(), h, float(cv_sharpe), note, json.dumps(config, default=str)),
        )
        self._conn.commit()
        return h

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def sharpes(self) -> list[float]:
        rows = self._conn.execute("SELECT cv_sharpe FROM trials").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
