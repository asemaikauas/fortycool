from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from fortycool_agents.discovery import SiteDiscoveryAgent
from fortycool_agents.discovery_catalog import public_catalog
from fortycool_agents.models import DataClass, DiscoveryRequest, DiscoveryStatus
from fortycool_agents.providers.fixture import FixtureThermalProvider
from fortycool_agents.providers.live import FortyGuardThermalProvider
from fortycool_agents.providers.urban import FixtureUrbanProvider


def polygon_feature(
    longitude: float, latitude: float, properties: dict[str, float]
) -> dict[str, Any]:
    delta = 0.00015
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [longitude - delta, latitude - delta],
                    [longitude + delta, latitude - delta],
                    [longitude + delta, latitude + delta],
                    [longitude - delta, latitude + delta],
                    [longitude - delta, latitude - delta],
                ]
            ],
        },
    }


def test_fixture_discovery_requires_and_passes_deep_validation() -> None:
    agent = SiteDiscoveryAgent(FixtureThermalProvider(), FixtureUrbanProvider())

    response = asyncio.run(agent.run(DiscoveryRequest()))

    assert response.status == DiscoveryStatus.QUALIFIED_CANDIDATE_FOUND
    assert response.winner is not None
    assert response.winner.candidate.id == "digital-realty-acc5"
    assert response.winner.screening_data_class == DataClass.SIMULATED
    assert response.winner.passed_drift_screen is True
    assert response.winner.control_match_status == "accepted"
    assert response.winner.validated_local_drift_c is not None
    assert response.winner.qualified is True
    assert response.winner.rejection_reasons == []
    assert {chart.id for chart in response.charts} == {
        "site_discovery_ranking",
        "site_discovery_map",
    }
    evidence_ids = {item.id for item in response.evidence}
    assert set(response.winner.evidence_ids) <= evidence_ids
    assert any("simulated" in warning.lower() for warning in response.warnings)


def test_control_quality_failure_cannot_produce_a_winner() -> None:
    class RejectedUrbanProvider:
        async def analyze(self, site, baseline_year, end_year, *, seed):
            urban = await FixtureUrbanProvider().analyze(
                site, baseline_year, end_year, seed=seed
            )
            return replace(
                urban,
                metadata={
                    **urban.metadata,
                    "minimum_selected_similarity": 0.2,
                    "control_quality_passed": False,
                },
            )

    candidate = public_catalog()[0]
    request = DiscoveryRequest(candidates=[candidate])
    agent = SiteDiscoveryAgent(FixtureThermalProvider(), RejectedUrbanProvider())

    response = asyncio.run(agent.run(request))

    assert response.status == DiscoveryStatus.NO_QUALIFIED_CANDIDATE
    assert response.winner is None
    result = response.candidates[0]
    assert result.passed_drift_screen is True
    assert result.control_match_status == "rejected"
    assert result.qualified is False
    assert "control_quality_gate_failed" in result.rejection_reasons


def test_missing_historical_land_cover_cannot_produce_a_winner() -> None:
    class CurrentImageryOnlyProvider:
        async def analyze(self, site, baseline_year, end_year, *, seed):
            urban = await FixtureUrbanProvider().analyze(
                site, baseline_year, end_year, seed=seed
            )
            return replace(
                urban,
                metadata={
                    **urban.metadata,
                    "historical_change_available": False,
                    "complete_history_years": [],
                },
            )

    candidate = public_catalog()[0]
    agent = SiteDiscoveryAgent(FixtureThermalProvider(), CurrentImageryOnlyProvider())

    response = asyncio.run(agent.run(DiscoveryRequest(candidates=[candidate])))

    assert response.status == DiscoveryStatus.NO_QUALIFIED_CANDIDATE
    assert response.winner is None
    result = response.candidates[0]
    assert result.control_match_status == "accepted"
    assert result.historical_land_cover_status == "not_available"
    assert "historical_land_cover_unavailable" in result.rejection_reasons


def test_live_flat_screen_skips_satellite_and_returns_no_qualified_site() -> None:
    candidate = public_catalog()[0]

    class FlatFortyGuardClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create_heatmap(
            self, payload: dict[str, Any], *, use_cache: bool = True
        ) -> dict[str, Any]:
            del use_cache
            self.calls.append(payload)
            year = int(payload["date_time"]["start_date"][:4])
            regional_change = 0.1 * (year - 2022)
            features = [
                polygon_feature(
                    candidate.site.longitude,
                    candidate.site.latitude,
                    {"average_temperature": 26.0 + regional_change},
                ),
                polygon_feature(
                    candidate.site.longitude + 0.006,
                    candidate.site.latitude,
                    {"average_temperature": 25.0 + regional_change},
                ),
            ]
            return {
                "error": False,
                "data": {
                    "activity_id": f"flat-{year}",
                    "status": "Completed",
                    "result": {
                        "map_data": {
                            "type": "FeatureCollection",
                            "features": features,
                        },
                        "stats_data": {},
                    },
                },
            }

    class UrbanMustNotRun:
        async def analyze(self, site, baseline_year, end_year, *, seed):
            raise AssertionError(
                "satellite validation should not run for a flat screen"
            )

    client = FlatFortyGuardClient()
    thermal = FortyGuardThermalProvider(client)  # type: ignore[arg-type]
    agent = SiteDiscoveryAgent(thermal, UrbanMustNotRun())
    request = DiscoveryRequest(candidates=[candidate])

    response = asyncio.run(agent.run(request))

    assert response.status == DiscoveryStatus.NO_QUALIFIED_CANDIDATE
    assert response.winner is None
    assert len(client.calls) == 2
    result = response.candidates[0]
    assert result.screening_data_class == DataClass.INFERRED
    assert result.screening_local_drift_c == 0.0
    assert result.control_match_status == "not_evaluated"
    assert result.rejection_reasons == ["below_drift_threshold"]
    screen_evidence = next(
        item for item in response.evidence if item.id.startswith("screen-")
    )
    assert screen_evidence.endpoint == "/v1/heatmap"
    assert screen_evidence.metadata["activity_ids"] == ["flat-2022", "flat-2026"]
