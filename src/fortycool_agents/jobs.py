from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

import logging

from .models import AnalysisRequest, AnalysisResponse, JobState, RunJobStatus, TraceEvent
from .orchestrator import FortyCoolOrchestrator
from .storage import RunRepository

logger = logging.getLogger(__name__)


# Each finished job holds a full analysis response, including charts, a GeoJSON
# heatmap, and the whole trace. Nothing ever evicted them, so the dictionary was
# an unbounded, attacker-fillable allocation reachable from an endpoint that
# returns in about a millisecond.
MAX_RETAINED_JOBS = int(os.getenv("FORTYCOOL_MAX_JOBS", 256))
JOB_TTL_SECONDS = float(os.getenv("FORTYCOOL_JOB_TTL_SECONDS", 3600))
# Ceiling on how long one SSE stream may stay open, and on how long a job may
# sit queued before a stream gives up on it. The loop's only exit was a terminal
# state, so a job that never started streamed keep-alives forever.
STREAM_MAX_SECONDS = float(os.getenv("FORTYCOOL_STREAM_MAX_SECONDS", 900))
STREAM_QUEUED_TIMEOUT_SECONDS = float(
    os.getenv("FORTYCOOL_STREAM_QUEUED_TIMEOUT_SECONDS", 60)
)


@dataclass
class JobRecord:
    run_id: str
    state: JobState = JobState.QUEUED
    events: list[TraceEvent] = field(default_factory=list)
    response: AnalysisResponse | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class RunJobManager:
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
        self._jobs: "OrderedDict[str, JobRecord]" = OrderedDict()
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds

    def _evict(self) -> None:
        now = time.monotonic()
        for run_id, record in list(self._jobs.items()):
            finished = record.state in {JobState.COMPLETED, JobState.FAILED}
            if finished and now - record.created_at > self.ttl_seconds:
                self._jobs.pop(run_id, None)
        while len(self._jobs) > self.max_jobs:
            for run_id, record in list(self._jobs.items()):
                if record.state in {JobState.COMPLETED, JobState.FAILED}:
                    self._jobs.pop(run_id, None)
                    break
            else:
                # Everything still in flight; do not drop a running job.
                break

    def create(self) -> RunJobStatus:
        run_id = uuid4().hex
        self._evict()
        self._jobs[run_id] = JobRecord(run_id=run_id)
        return self.snapshot(run_id)

    async def execute(self, run_id: str, request: AnalysisRequest) -> None:
        record = self._jobs.get(run_id)
        if record is None:
            # The job was evicted before the background task started. Nothing to
            # report to, and re-creating it would resurrect a dead run id.
            return
        record.state = JobState.RUNNING
        try:
            response = await self.orchestrator.run(
                request,
                run_id=run_id,
                event_sink=record.events.append,
            )
            record.response = response
            record.state = JobState.COMPLETED
            self.repository.save(response)
        except Exception as exc:  # job boundary must preserve failure state for clients
            record.state = JobState.FAILED
            # The client gets a stable, non-revealing message; the detail stays
            # in the server log. The old text handed back pydantic field paths,
            # offending values, and a documentation URL.
            record.error = "the analysis could not be completed"
            logger.exception("run job %s failed", run_id, exc_info=exc)

    def snapshot(self, run_id: str) -> RunJobStatus:
        if run_id not in self._jobs:
            raise KeyError(run_id)
        record = self._jobs[run_id]
        return RunJobStatus(
            run_id=record.run_id,
            state=record.state,
            event_count=len(record.events),
            error=record.error,
            response=record.response,
        )

    async def stream(self, run_id: str):
        if run_id not in self._jobs:
            raise KeyError(run_id)
        cursor = 0
        started = time.monotonic()
        while True:
            record = self._jobs.get(run_id)
            if record is None:
                yield (
                    'event: terminal\ndata: {"run_id":"%s","state":"failed",'
                    '"error":"the run record expired"}\n\n' % run_id
                )
                return
            while cursor < len(record.events):
                event = record.events[cursor]
                cursor += 1
                payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                yield f"event: trace\ndata: {payload}\n\n"
            if record.state in {JobState.COMPLETED, JobState.FAILED}:
                payload = json.dumps(
                    self.snapshot(run_id).model_dump(mode="json", exclude={"response"}),
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
