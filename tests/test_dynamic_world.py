from __future__ import annotations

import asyncio
from typing import Any

from fortycool_agents.models import AnalysisRequest, DataClass, SiteInput, WarningCode
from fortycool_agents.orchestrator import FortyCoolOrchestrator
from fortycool_agents.providers.dynamic_world import (
    DYNAMIC_WORLD_DATASET,
    DynamicWorldRecord,
    DynamicWorldUrbanProvider,
)
from fortycool_agents.providers.fixture import FixtureThermalProvider


class FakeDynamicWorldGateway:
    project = "fortycool-test"
    dataset_id = DYNAMIC_WORLD_DATASET

    def __init__(self, *, incomplete_locations: set[str] | None = None) -> None:
        self.incomplete_locations = incomplete_locations or set()
        self.calls: list[dict[str, Any]] = []

    async def summarize(
        self,
        locations: dict[str, SiteInput],
        years: list[int],
        *,
        radius_m: float,
        workload_tag: str,
    ) -> list[DynamicWorldRecord]:
        self.calls.append(
            {
                "locations": locations,
                "years": years,
                "radius_m": radius_m,
                "workload_tag": workload_tag,
            }
        )
        records: list[DynamicWorldRecord] = []
        offsets = {
            "site": 0.0,
            "control_n": -2.0,
            "control_e": -1.0,
            "control_s": 1.0,
            "control_w": 2.0,
        }
        for location_id in locations:
            for year in years:
                if location_id in self.incomplete_locations and year == years[1]:
                    continue
                elapsed = year - years[0]
                if location_id == "site":
                    built = 30.0 + 2.0 * elapsed
                    trees = 38.0 - 1.5 * elapsed
                else:
                    built = 30.0 + offsets[location_id] + 0.5 * elapsed
                    trees = 38.0 - offsets[location_id] - 0.25 * elapsed
                bare = 18.0
                water = 2.0
                grass = 100.0 - built - trees - bare - water
                records.append(
                    DynamicWorldRecord(
                        location_id=location_id,
                        year=year,
                        segments={
                            "built": built,
                            "trees": trees,
                            "grass": grass,
                            "bare": bare,
                            "water": water,
                        },
                        scene_count=12,
                        valid_pixel_count=2_100,
                        mean_valid_observations=9.5,
                        metadata={
                            "first_acquisition": f"{year}-06-03T00:00:00+00:00",
                            "last_acquisition": f"{year}-08-27T00:00:00+00:00",
                        },
                    )
                )
        return records


def site() -> SiteInput:
    return SiteInput(
        name="Northern Virginia Demo Campus",
        latitude=39.01,
        longitude=-77.46,
    )


def test_dynamic_world_builds_complete_annual_site_control_history() -> None:
    gateway = FakeDynamicWorldGateway()
    provider = DynamicWorldUrbanProvider(gateway)

    result = asyncio.run(provider.analyze(site(), 2022, 2026, seed=7))

    assert result.data_class == DataClass.OBSERVED
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["years"] == [2022, 2023, 2024, 2025, 2026]
    assert len(gateway.calls[0]["locations"]) == 5
    assert result.metadata["dataset_id"] == DYNAMIC_WORLD_DATASET
    assert result.metadata["control_quality_passed"] is True
    assert result.metadata["historical_control_stability_verified"] is True
    assert result.metadata["complete_history_years"] == [
        2022,
        2023,
        2024,
        2025,
        2026,
    ]
    series = result.metadata["annual_land_cover_series"]
    assert len(series) == 5
    assert series[-1]["site_built_surface_percent"] == 38.0
    assert series[-1]["site_tree_canopy_percent"] == 32.0
    assert (
        series[-1]["local_excess_built_surface_percent"]
        > series[0]["local_excess_built_surface_percent"]
    )
    assert result.activity_ids[0].startswith("fortycool-")


def test_incomplete_control_history_fails_the_quality_gate() -> None:
    gateway = FakeDynamicWorldGateway(incomplete_locations={"control_n", "control_e"})
    provider = DynamicWorldUrbanProvider(gateway)

    result = asyncio.run(provider.analyze(site(), 2022, 2026, seed=7))

    assert len(result.matched_controls) == 2
    assert result.metadata["control_quality_passed"] is False
    assert result.metadata["annual_land_cover_series"] == []
    assert any("coverage gates" in warning for warning in result.warnings)


def test_orchestrator_links_dynamic_world_history_to_thermal_attribution() -> None:
    urban = DynamicWorldUrbanProvider(FakeDynamicWorldGateway())
    orchestrator = FortyCoolOrchestrator(
        provider=FixtureThermalProvider(), urban_provider=urban
    )
    request = AnalysisRequest(
        site=site(),
        analysis_modes=["thermal_drift"],
        baseline_year=2019,
    )

    response = asyncio.run(orchestrator.run(request))

    metric_ids = {metric.id for metric in response.metrics}
    assert metric_ids >= {
        "local_excess_built_surface_change_percentage_points",
        "land_cover_history_coverage_years",
        "thermal_land_cover_association_correlation",
    }
    assert {chart.id for chart in response.charts} >= {
        "historical_land_cover_timeseries",
        "thermal_land_cover_attribution",
    }
    satellite_evidence = next(
        item for item in response.evidence if item.id.startswith("satellite-land-cover")
    )
    assert satellite_evidence.metadata["dataset_id"] == DYNAMIC_WORLD_DATASET
    assert satellite_evidence.endpoint.endswith("/table:computeFeatures")
    attribution = next(
        metric
        for metric in response.metrics
        if metric.id == "thermal_land_cover_association_correlation"
    )
    evidence_ids = {item.id for item in response.evidence}
    assert set(attribution.evidence_ids) <= evidence_ids
    # Every correlation now ships with its sample size and interval; a bare r
    # over a handful of years reads as far more certain than it is.
    assert attribution.interval_low is not None
    assert attribution.interval_high is not None
    assert attribution.interval_low < attribution.value < attribution.interval_high


def test_attribution_is_withheld_when_too_few_paired_years_exist() -> None:
    """Below the minimum sample the correlation is refused, not published.

    At n=3 the 5% critical value for |r| is 0.997 and two unrelated series clear
    0.8 four times in ten, so a short-series correlation is not evidence of
    anything and must not reach a dashboard as a number.
    """

    urban = DynamicWorldUrbanProvider(FakeDynamicWorldGateway())
    orchestrator = FortyCoolOrchestrator(
        provider=FixtureThermalProvider(), urban_provider=urban
    )
    request = AnalysisRequest(
        site=site(),
        analysis_modes=["thermal_drift"],
        baseline_year=2022,
    )

    response = asyncio.run(orchestrator.run(request))

    metric_ids = {metric.id for metric in response.metrics}
    assert "thermal_land_cover_association_correlation" not in metric_ids
    status = next(
        metric
        for metric in response.metrics
        if metric.id == "thermal_land_cover_association_status"
    )
    assert status.value == "withheld"
    assert WarningCode.ATTRIBUTION_WITHHELD.value in response.warning_codes
