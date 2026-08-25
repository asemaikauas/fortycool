from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock

from .models import AnalysisResponse


class RunRepository:
    """Small SQLite result repository suitable for a single hackathon service."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("FORTYCOOL_DB_PATH", ".fortycool-data/runs.sqlite3")
        self.path = str(configured)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                response_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._lock = Lock()

    def save(self, response: AnalysisResponse) -> None:
        payload = response.model_dump_json()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO analysis_runs (run_id, response_json)
                VALUES (?, ?)
                ON CONFLICT(run_id) DO UPDATE SET response_json = excluded.response_json
                """,
                (response.run_id, payload),
            )
            self._connection.commit()

    def get(self, run_id: str) -> AnalysisResponse | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM analysis_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return AnalysisResponse.model_validate_json(row[0])
