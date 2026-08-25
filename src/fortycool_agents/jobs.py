from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import uuid4

from .models import AnalysisRequest, AnalysisResponse, JobState, RunJobStatus, TraceEvent
from .orchestrator import FortyCoolOrchestrator
from .storage import RunRepository


@dataclass
class JobRecord:
    run_id: str
    state: JobState = JobState.QUEUED
    events: list[TraceEvent] = field(default_factory=list)
    response: AnalysisResponse | None = None
    error: str | None = None


class RunJobManager:
    def __init__(
        self, orchestrator: FortyCoolOrchestrator, repository: RunRepository
    ) -> None:
        self.orchestrator = orchestrator
        self.repository = repository
        self._jobs: dict[str, JobRecord] = {}

    def create(self) -> RunJobStatus:
        run_id = uuid4().hex
        self._jobs[run_id] = JobRecord(run_id=run_id)
        return self.snapshot(run_id)

    async def execute(self, run_id: str, request: AnalysisRequest) -> None:
        record = self._jobs[run_id]
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
            record.error = f"{type(exc).__name__}: {exc}"

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
        while True:
            record = self._jobs[run_id]
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
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)
