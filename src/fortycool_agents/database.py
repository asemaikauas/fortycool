from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Sequence


class DatabaseSession:
    """Expose the small SQL subset used by the service on both databases."""

    def __init__(self, connection: Any, *, postgres: bool) -> None:
        self.connection = connection
        self.postgres = postgres

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> Any:
        if self.postgres:
            sql = sql.replace("?", "%s")
        return self.connection.execute(sql, parameters)


class Database:
    """Shared persistence connection with PostgreSQL and local SQLite support.

    Heroku supplies ``DATABASE_URL`` when Postgres is attached. Local and test
    processes fall back to ``FORTYCOOL_DB_PATH`` so contributors do not need a
    database server to run the application.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        database_url: str | None = None,
    ) -> None:
        # Passing a path explicitly is an intentional request for SQLite (used
        # heavily by tests), even if the caller's shell also has DATABASE_URL.
        configured_url = database_url if database_url is not None else (
            None if path is not None else os.getenv("DATABASE_URL")
        )
        self.database_url = configured_url.strip() if configured_url else None
        self.postgres = bool(
            self.database_url
            and self.database_url.startswith(("postgres://", "postgresql://"))
        )
        if self.database_url and not self.postgres:
            raise ValueError("DATABASE_URL must use the postgres or postgresql scheme")

        self._lock = RLock()
        self._sqlite_connection: sqlite3.Connection | None = None
        if not self.postgres:
            configured_path = path or os.getenv(
                "FORTYCOOL_DB_PATH", ".fortycool-data/runs.sqlite3"
            )
            self.path = str(configured_path)
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_connection = sqlite3.connect(
                self.path, check_same_thread=False
            )
            self._sqlite_connection.execute("PRAGMA foreign_keys=ON")
            if self.path != ":memory:":
                self._sqlite_connection.execute("PRAGMA journal_mode=WAL")
            self._sqlite_connection.execute("PRAGMA busy_timeout=5000")
        else:
            self.path = ""

        self._initialize_schema()

    @property
    def backend(self) -> str:
        return "postgresql" if self.postgres else "sqlite"

    @contextmanager
    def session(self) -> Iterator[DatabaseSession]:
        """Run one committed transaction, serialized inside this process."""

        with self._lock:
            if self.postgres:
                try:
                    import psycopg
                except ImportError as exc:  # pragma: no cover - deployment guard
                    raise RuntimeError(
                        "PostgreSQL persistence requires the psycopg dependency"
                    ) from exc
                assert self.database_url is not None
                with psycopg.connect(self.database_url) as connection:
                    yield DatabaseSession(connection, postgres=True)
                return

            assert self._sqlite_connection is not None
            try:
                yield DatabaseSession(self._sqlite_connection, postgres=False)
                self._sqlite_connection.commit()
            except Exception:
                self._sqlite_connection.rollback()
                raise

    def _initialize_schema(self) -> None:
        real_type = "DOUBLE PRECISION" if self.postgres else "REAL"
        with self.session() as session:
            session.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS analysis_runs_created_at "
                "ON analysis_runs (created_at DESC)"
            )
            session.execute(
                f"""
                CREATE TABLE IF NOT EXISTS run_jobs (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    error TEXT,
                    created_at_epoch {real_type} NOT NULL,
                    updated_at_epoch {real_type} NOT NULL
                )
                """
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS run_jobs_created_at "
                "ON run_jobs (created_at_epoch DESC)"
            )
            session.execute(
                """
                CREATE TABLE IF NOT EXISTS run_job_events (
                    run_id TEXT NOT NULL REFERENCES run_jobs(run_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            session.execute(
                f"""
                CREATE TABLE IF NOT EXISTS telemetry_uploads (
                    upload_id TEXT PRIMARY KEY,
                    created_at_epoch {real_type} NOT NULL,
                    accessed_at_epoch {real_type} NOT NULL,
                    metadata_json TEXT NOT NULL,
                    frame_json TEXT NOT NULL
                )
                """
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS telemetry_uploads_accessed_at "
                "ON telemetry_uploads (accessed_at_epoch DESC)"
            )
