from __future__ import annotations

import asyncio

from fortycool_agents.models import AnalysisRequest
from fortycool_agents.orchestrator import FortyCoolOrchestrator
from fortycool_agents.storage import RunRepository


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

    assert restored is not None
    assert restored.model_dump(mode="json") == response.model_dump(mode="json")
