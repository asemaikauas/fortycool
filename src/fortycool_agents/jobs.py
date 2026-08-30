from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from uuid import uuid4

from .models import AnalysisRequest, JobState, RunJobStatus, TraceEvent
from .orchestrator import FortyCoolOrchestrator
from .storage import RunRepository

logger = logging.getLogger(__name__)


MAX_RETAINED_JOBS = int(os.getenv("FORTYCOOL_MAX_JOBS", 256))
JOB_TTL_SECONDS = float(os.getenv("FORTYCOOL_JOB_TTL_SECONDS", 3600))
STREAM_MAX_SECONDS = float(os.getenv("FORTYCOOL_STREAM_MAX_SECONDS", 900))
STREAM_QUEUED_TIMEOUT_SECONDS = float(
    os.getenv("FORTYCOOL_STREAM_QUEUED_TIMEOUT_SECONDS", 60)
)


class RunJobManager:
    """Durable job status and trace storage shared through the main database."""

    def __init__(
        self,
        orchestrator: FortyCoolOrchestrator,
        repository: RunRepository,
        *,
        max_jobs: int = MAX_RETAINED_JOBS,
        ttl_seconds: float = JOB_TTL_SECONDS,
    ) -> None:
        self.orchestrator = orchestrator
        self.repository = repository
        self.database = repository.database
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds
        # A Heroku restart cannot resume a FastAPI BackgroundTask. Expose that
        # explicitly instead of leaving a job in "running" forever.
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        now = time.time()
        with self.database.session() as session:
            session.execute(
                """
                UPDATE run_jobs
                SET state = ?, error = ?, updated_at_epoch = ?
                WHERE state IN (?, ?)
                """,
                (
                    JobState.FAILED.value,
                    "the analysis was interrupted by a service restart; submit a new run",
                    now,
                    JobState.QUEUED.value,
                    JobState.RUNNING.value,
                ),
            )

    def _evict(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        terminal = (JobState.COMPLETED.value, JobState.FAILED.value)
        with self.database.session() as session:
            session.execute(
                """
                DELETE FROM run_jobs
                WHERE state IN (?, ?) AND created_at_epoch < ?
                """,
                (*terminal, cutoff),
            )
            count_row = session.execute("SELECT COUNT(*) FROM run_jobs").fetchone()
            excess = max(0, int(count_row[0]) - self.max_jobs)
            if excess:
                session.execute(
                    """
                    DELETE FROM run_jobs
                    WHERE run_id IN (
                        SELECT run_id FROM run_jobs
                        WHERE state IN (?, ?)
                        ORDER BY created_at_epoch ASC, run_id ASC
                        LIMIT ?
                    )
                    """,
                    (*terminal, excess),
                )

    def create(self) -> RunJobStatus:
        self._evict()
        run_id = uuid4().hex
        now = time.time()
        with self.database.session() as session:
            session.execute(
                """
                INSERT INTO run_jobs (
                    run_id, state, error, created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, JobState.QUEUED.value, None, now, now),
            )
        return self.snapshot(run_id)

    def _set_state(
        self, run_id: str, state: JobState, *, error: str | None = None
    ) -> bool:
        with self.database.session() as session:
            cursor = session.execute(
                """
                UPDATE run_jobs
                SET state = ?, error = ?, updated_at_epoch = ?
                WHERE run_id = ?
                """,
                (state.value, error, time.time(), run_id),
            )
            return cursor.rowcount > 0

    def _append_event(self, run_id: str, event: TraceEvent) -> None:
        payload = event.model_dump_json()
        with self.database.session() as session:
            row = session.execute(
                """
                SELECT COALESCE(MAX(sequence), -1) + 1
                FROM run_job_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            session.execute(
                """
                INSERT INTO run_job_events (run_id, sequence, event_json)
                VALUES (?, ?, ?)
                """,
                (run_id, int(row[0]), payload),
            )

    async def execute(self, run_id: str, request: AnalysisRequest) -> None:
        if not self._set_state(run_id, JobState.RUNNING):
            return
        try:
            response = await self.orchestrator.run(
                request,
                run_id=run_id,
                event_sink=lambda event: self._append_event(run_id, event),
            )
            self.repository.save(response)
            self._set_state(run_id, JobState.COMPLETED)
        except Exception as exc:  # job boundary must preserve failure state for clients
            self._set_state(
                run_id,
                JobState.FAILED,
                error="the analysis could not be completed",
            )
            logger.exception("run job %s failed", run_id, exc_info=exc)

    def snapshot(self, run_id: str) -> RunJobStatus:
        with self.database.session() as session:
            row = session.execute(
                """
                SELECT state, error,
                    (SELECT COUNT(*) FROM run_job_events WHERE run_id = ?)
                FROM run_jobs
                WHERE run_id = ?
                """,
                (run_id, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        state = JobState(row[0])
        response = (
            self.repository.get(run_id) if state == JobState.COMPLETED else None
        )
        return RunJobStatus(
            run_id=run_id,
            state=state,
            event_count=int(row[2]),
            error=row[1],
            response=response,
        )

    def _events_since(self, run_id: str, cursor: int) -> list[TraceEvent]:
        with self.database.session() as session:
            rows = session.execute(
                """
                SELECT event_json FROM run_job_events
                WHERE run_id = ? AND sequence >= ?
                ORDER BY sequence ASC
                """,
                (run_id, cursor),
            ).fetchall()
        return [TraceEvent.model_validate_json(row[0]) for row in rows]

    async def stream(self, run_id: str):
        self.snapshot(run_id)
        cursor = 0
        started = time.monotonic()
        while True:
            try:
                record = self.snapshot(run_id)
            except KeyError:
                yield (
                    'event: terminal\ndata: {"run_id":"%s","state":"failed",'
                    '"error":"the run record expired"}\n\n' % run_id
                )
                return
            events = self._events_since(run_id, cursor)
            for event in events:
                cursor += 1
                payload = json.dumps(
                    event.model_dump(mode="json"), separators=(",", ":")
                )
                yield f"event: trace\ndata: {payload}\n\n"
            if record.state in {JobState.COMPLETED, JobState.FAILED}:
                payload = json.dumps(
                    record.model_dump(mode="json", exclude={"response"}),
                    separators=(",", ":"),
                )
                yield f"event: terminal\ndata: {payload}\n\n"
                return
            elapsed = time.monotonic() - started
            stalled = (
                record.state == JobState.QUEUED
                and elapsed > STREAM_QUEUED_TIMEOUT_SECONDS
            )
            if stalled or elapsed > STREAM_MAX_SECONDS:
                reason = (
                    "the run never started"
                    if stalled
                    else "the stream exceeded its time budget"
                )
                yield (
                    'event: terminal\ndata: {"run_id":"%s","state":"failed",'
                    '"error":"%s"}\n\n' % (run_id, reason)
                )
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)
