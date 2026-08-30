from __future__ import annotations

import asyncio

from fortycool_agents.database import Database
from fortycool_agents.jobs import RunJobManager
from fortycool_agents.models import AnalysisRequest
from fortycool_agents.orchestrator import FortyCoolOrchestrator
from fortycool_agents.storage import RunRepository
from fortycool_agents.telemetry import TelemetryStore


def test_sqlite_repository_round_trips_analysis(tmp_path) -> None:
    request = AnalysisRequest.model_validate(
        {
            "site": {
                "name": "Persistence Demo",
                "latitude": 39.01,
                "longitude": -77.46,
            },
            "analysis_modes": ["thermal_drift"],
        }
    )
    response = asyncio.run(FortyCoolOrchestrator().run(request))
    repository = RunRepository(tmp_path / "runs.sqlite3")

    repository.save(response)
    restored = repository.get(response.run_id)
    recent = repository.list_recent(limit=10)

    assert restored is not None
    assert restored.model_dump(mode="json") == response.model_dump(mode="json")
    assert len(recent) == 1
    assert recent[0][1].run_id == response.run_id


def test_completed_job_and_trace_survive_manager_restart(tmp_path) -> None:
    path = tmp_path / "durable.sqlite3"
    database = Database(path)
    telemetry = TelemetryStore(database=database)
    repository = RunRepository(database=database)
    manager = RunJobManager(
        FortyCoolOrchestrator(telemetry_store=telemetry), repository
    )
    request = AnalysisRequest.model_validate(
        {
            "site": {
                "name": "Durable Job Demo",
                "latitude": 39.01,
                "longitude": -77.46,
            },
            "analysis_modes": ["thermal_drift"],
            "simulation": {"enabled": True, "seed": 13},
        }
    )

    created = manager.create()
    asyncio.run(manager.execute(created.run_id, request))

    reopened_database = Database(path)
    reopened_repository = RunRepository(database=reopened_database)
    reopened = RunJobManager(
        FortyCoolOrchestrator(
            telemetry_store=TelemetryStore(database=reopened_database)
        ),
        reopened_repository,
    )
    restored = reopened.snapshot(created.run_id)

    assert restored.state.value == "completed"
    assert restored.event_count > 0
    assert restored.response is not None
    assert restored.response.run_id == created.run_id
