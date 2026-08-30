from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .database import Database
from .models import AnalysisResponse

# Each saved run is roughly 34 KB of JSON. Bound retention on both the local
# SQLite fallback and the Heroku Postgres database.
MAX_RETAINED_RUNS = int(os.getenv("FORTYCOOL_MAX_RETAINED_RUNS", 2_000))


class RunRepository:
    """Durable analysis results backed by Postgres or local SQLite."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        database: Database | None = None,
        max_rows: int = MAX_RETAINED_RUNS,
    ) -> None:
        if path is not None and database is not None:
            raise ValueError("pass either path or database, not both")
        self.database = database or Database(path)
        self.path = self.database.path
        self.max_rows = max_rows

    def save(self, response: AnalysisResponse) -> None:
        payload = response.model_dump_json()
        saved_at = datetime.now(timezone.utc).isoformat()
        with self.database.session() as session:
            session.execute(
                """
                INSERT INTO analysis_runs (run_id, created_at, response_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (response.run_id, saved_at, payload),
            )
            session.execute(
                """
                DELETE FROM analysis_runs
                WHERE run_id NOT IN (
                    SELECT run_id FROM analysis_runs
                    ORDER BY created_at DESC, run_id DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
            )

    def get(self, run_id: str) -> AnalysisResponse | None:
        with self.database.session() as session:
            row = session.execute(
                "SELECT response_json FROM analysis_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return AnalysisResponse.model_validate_json(row[0])

    def list_recent(self, *, limit: int = 100) -> list[tuple[str, AnalysisResponse]]:
        bounded_limit = max(1, min(limit, 500))
        with self.database.session() as session:
            rows = session.execute(
                """
                SELECT created_at, response_json
                FROM analysis_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [
            (str(created_at), AnalysisResponse.model_validate_json(payload))
            for created_at, payload in rows
        ]
