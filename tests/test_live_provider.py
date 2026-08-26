from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fortycool_agents.models import AnalysisRequest, DataClass, SiteInput
from fortycool_agents.orchestrator import FortyCoolOrchestrator
from fortycool_agents.providers import (
    FixtureThermalProvider,
    FortyGuardThermalProvider,
    build_thermal_provider,
)


def polygon_feature(
    longitude: float, latitude: float, properties: dict[str, float | int]
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


class FakeFortyGuardClient:
    def __init__(self, *, unavailable_years: set[int] | None = None) -> None:
        self.unavailable_years = unavailable_years or set()
        self.calls: list[dict[str, Any]] = []

    async def create_heatmap(
        self, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        self.calls.append(payload)
        date_time = payload["date_time"]
        year = int(date_time["start_date"][:4])
        analytic_type = payload.get("analytic_type", "tcm")
        activity_id = f"activity-{year}-{analytic_type}-{len(self.calls)}"
        if year in self.unavailable_years:
            features: list[dict[str, Any]] = []
        elif analytic_type == "exceedance":
            features = [
                polygon_feature(-77.46, 39.01, {"tile_id": 1, "value": 500}),
                polygon_feature(-77.454, 39.01, {"tile_id": 2, "value": 550}),
            ]
        else:
            year_offset = year - 2021
            features = [
                polygon_feature(
                    -77.46,
                    39.01,
                    {"tile_id": 1, "average_temperature": 26.0 + 0.2 * year_offset},
                ),
                polygon_feature(
                    -77.454,
                    39.01,
                    {"tile_id": 2, "average_temperature": 25.0 + 0.1 * year_offset},
                ),
            ]
        return {
            "data": {
                "activity_id": activity_id,
                "status": "Completed",
                "result": {
                    "map_data": {"type": "FeatureCollection", "features": features},
                    "stats_data": {},
                },
            },
            "error": False,
            "status_code": 200,
        }


def site() -> SiteInput:
    return SiteInput(
        name="Northern Virginia Demo Campus",
        latitude=39.01,
        longitude=-77.46,
    )


def test_live_forecast_normalizes_spatial_tiles_and_keeps_field_lineage() -> None:
    client = FakeFortyGuardClient()
    provider = FortyGuardThermalProvider(
        client, clock=lambda: datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc)
    )

    result = asyncio.run(provider.forecast(site(), 3, seed=9))

    assert result.data_class == DataClass.OBSERVED
    assert len(result.samples) == 3
    assert result.samples[0].timestamp == datetime(2026, 8, 25, 13, tzinfo=timezone.utc)
    assert result.samples[0].site_temperature_c == 27.0
    assert result.samples[0].control_temperature_c == 25.5
    assert result.metadata["observed_hours"] == 3
    assert result.metadata["environmental_fields_data_class"] == "simulated"
    assert result.heatmap is not None
    assert len(result.activity_ids) == 3


def test_historical_coverage_gap_becomes_calibrated_backcast() -> None:
    client = FakeFortyGuardClient(unavailable_years={2021})
    provider = FortyGuardThermalProvider(client)

    result = asyncio.run(provider.annual_history(site(), 2021, 2022, 18.0, seed=11))

    assert result.data_class == DataClass.INFERRED
    assert result.metadata["observed_years"] == [2022]
    assert result.metadata["backcast_years"] == [2021]
    assert [item.year for item in result.summaries] == [2021, 2022]
    assert result.summaries[1].site_mean_temperature_c == 26.2
    assert any("calibrated simulated backcasts" in warning for warning in result.warnings)


def test_historical_heatmap_uses_three_equal_weight_regional_control_zones() -> None:
    controls = [
        SiteInput(
            name=f"Control {index}",
            latitude=latitude,
            longitude=longitude,
        )
        for index, (latitude, longitude) in enumerate(
            [(39.037, -77.46), (39.01, -77.425), (38.983, -77.46)], start=1
        )
    ]

    class RegionalFakeClient(FakeFortyGuardClient):
        async def create_heatmap(
            self, payload: dict[str, Any], *, use_cache: bool = True
        ) -> dict[str, Any]:
            del use_cache
            self.calls.append(payload)
            year = int(payload["date_time"]["start_date"][:4])
            analytic_type = payload.get("analytic_type", "tcm")
            if analytic_type == "exceedance":
                site_value = 500
                control_values = [540, 550, 560]
                value_key = "value"
            else:
                site_value = 26.0 + 0.2 * (year - 2021)
                control_values = [
                    25.0 + 0.1 * (year - 2021),
                    25.2 + 0.1 * (year - 2021),
                    24.8 + 0.1 * (year - 2021),
                ]
                value_key = "average_temperature"
            features = [
                polygon_feature(-77.46, 39.01, {value_key: site_value}),
                *[
                    polygon_feature(
                        control.longitude,
                        control.latitude,
                        {value_key: value},
                    )
                    for control, value in zip(controls, control_values)
                ],
            ]
            return {
                "error": False,
                "data": {
                    "activity_id": f"regional-{year}-{analytic_type}",
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

    provider = FortyGuardThermalProvider(RegionalFakeClient())

    result = asyncio.run(
        provider.annual_history(site(), 2021, 2022, 18.0, seed=11, controls=controls)
    )

    assert result.metadata["control_method"] == (
        "satellite_land_cover_matched_regional"
    )
    assert result.metadata["tile_counts"]["2022"]["control_zones"] == 3
    assert result.metadata["tile_counts"]["2022"]["control_tiles_by_zone"] == [
        1,
        1,
        1,
    ]
    assert result.summaries[1].control_mean_temperature_c == 25.1
    assert len(result.metadata["control_sites"]) == 3


def test_orchestrator_emits_live_map_timeline_and_split_provenance() -> None:
    provider = FortyGuardThermalProvider(
        FakeFortyGuardClient(),
        clock=lambda: datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc),
    )
    request = AnalysisRequest.model_validate(
        {
            "site": site().model_dump(),
            "analysis_modes": ["operations_12h"],
            "facility": {"it_capacity_mw": 24},
            "simulation": {
                "enabled": True,
                "seed": 5,
                "history_days": 14,
                "forecast_hours": 3,
            },
        }
    )

    response = asyncio.run(FortyCoolOrchestrator(provider=provider).run(request))

    charts = {chart.id: chart for chart in response.charts}
    assert charts["fortyguard_heatmap"].data_class == DataClass.OBSERVED
    assert charts["temperature_forecast_12h"].data_class == DataClass.OBSERVED
    evidence = {item.id: item for item in response.evidence}
    history = next(item for key, item in evidence.items() if key.startswith("historical-weather"))
    forecast = next(item for key, item in evidence.items() if key.startswith("forecast-weather"))
    assert history.data_class == DataClass.SIMULATED
    assert forecast.data_class == DataClass.OBSERVED
    assert forecast.activity_id is not None


def test_provider_factory_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("FORTYGUARD_API_KEY", raising=False)
    assert isinstance(build_thermal_provider("fixture"), FixtureThermalProvider)
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test-only-key")
    assert isinstance(build_thermal_provider("live"), FortyGuardThermalProvider)
