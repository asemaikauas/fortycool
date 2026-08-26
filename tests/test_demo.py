from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fortycool_agents.demo import _verified_manifest
from fortycool_agents.models import AnalysisRequest, DataClass
from fortycool_agents.orchestrator import FortyCoolOrchestrator


def _candidate():
    request = AnalysisRequest.model_validate(
        {
            "site": {
                "name": "Verified Demo Candidate",
                "latitude": 39.01,
                "longitude": -77.46,
            },
            "analysis_modes": ["thermal_drift", "investment"],
            "simulation": {"enabled": True, "seed": 47},
        }
    )
    run = asyncio.run(FortyCoolOrchestrator().run(request))
    thermal = next(item for item in run.evidence if item.id.startswith("thermal-"))
    thermal.source = "https://api.fortyguard.com/v1/heatmap"
    thermal.data_class = DataClass.INFERRED
    thermal.metadata.update(
        {
            "observed_years": [2022, 2023, 2024, 2025, 2026],
            "backcast_years": [],
            "control_method": "satellite_land_cover_matched_regional",
            "control_sites": [{"name": "Control A"}, {"name": "Control B"}],
        }
    )
    satellite = next(
        item for item in run.evidence if item.id.startswith("satellite-land-cover-")
    )
    satellite.source = "https://earthengine.googleapis.com"
    satellite.data_class = DataClass.OBSERVED
    satellite.metadata.update(
        {
            "dataset_id": "GOOGLE/DYNAMICWORLD/V1",
            "historical_control_stability_verified": True,
        }
    )
    return run


def test_verified_demo_requires_real_history_and_stable_controls() -> None:
    manifest = _verified_manifest(
        _candidate(), saved_at=datetime(2026, 8, 26, tzinfo=timezone.utc)
    )

    assert manifest is not None
    assert manifest.thermal_years == [2022, 2023, 2024, 2025, 2026]
    assert manifest.observed_heatmap is False
    assert "saved completed analysis" in " ".join(manifest.verification_notes)


def test_verified_demo_rejects_calibrated_backcast_years() -> None:
    run = _candidate()
    thermal = next(
        item
        for item in run.evidence
        if item.source.startswith("https://api.fortyguard.com/v1/heatmap")
    )
    thermal.metadata["backcast_years"] = [2021]

    assert (
        _verified_manifest(
            run, saved_at=datetime(2026, 8, 26, tzinfo=timezone.utc)
        )
        is None
    )


def test_verified_demo_rejects_a_failed_regional_control_gate() -> None:
    run = _candidate()
    control_status = next(
        metric for metric in run.metrics if metric.id == "regional_control_match_status"
    )
    control_status.value = "rejected"

    assert (
        _verified_manifest(
            run, saved_at=datetime(2026, 8, 26, tzinfo=timezone.utc)
        )
        is None
    )
