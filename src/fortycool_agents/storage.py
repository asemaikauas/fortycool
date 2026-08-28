from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock

from .models import AnalysisResponse

# Each saved run is roughly 34 KB of JSON, and nothing pruned the table, so the
# database grew without limit while `/demo/verified-run` parsed up to 500 of
# those blobs on every call.
MAX_RETAINED_RUNS = int(os.getenv("FORTYCOOL_MAX_RETAINED_RUNS", 2_000))


class RunRepository:
    """Small SQLite result repository suitable for a single hackathon service."""

    def __init__(
        self, path: str | Path | None = None, *, max_rows: int = MAX_RETAINED_RUNS
    ) -> None:
        configured = path or os.getenv("FORTYCOOL_DB_PATH", ".fortycool-data/runs.sqlite3")
        self.path = str(configured)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        # Write-ahead logging plus a busy timeout so a second process reading or
        # writing the same file waits instead of raising "database is locked"
        # straight through to the client as a 500.
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                response_json TEXT NOT NULL
            )
            """
        )
        # `rowid` is implicit and cannot be indexed, but ordering by created_at
        # is the scan that `list_recent` and the retention sweep both perform.
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS analysis_runs_created_at "
            "ON analysis_runs (created_at DESC)"
        )
        self._connection.commit()
        self._lock = Lock()
        self.max_rows = max_rows

    def save(self, response: AnalysisResponse) -> None:
        payload = response.model_dump_json()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO analysis_runs (run_id, response_json)
                VALUES (?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (response.run_id, payload),
            )
            # Retention, so the file cannot grow without bound.
            self._connection.execute(
                """
                DELETE FROM analysis_runs
                WHERE rowid NOT IN (
                    SELECT rowid FROM analysis_runs
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
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

    def list_recent(self, *, limit: int = 100) -> list[tuple[str, AnalysisResponse]]:
        bounded_limit = max(1, min(limit, 500))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT created_at, response_json
                FROM analysis_runs
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [
            (created_at, AnalysisResponse.model_validate_json(payload))
            for created_at, payload in rows
        ]
