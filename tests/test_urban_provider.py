from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fortycool_agents.models import DataClass, SiteInput
from fortycool_agents.providers.urban import (
    FortyGuardUrbanProvider,
    canonical_land_cover,
)


class FakeSatelliteClient:
    def __init__(
        self, *, historical_images: bool = False, poor_controls: bool = False
    ) -> None:
        self.historical_images = historical_images
        self.poor_controls = poor_controls
        self.calls: list[dict[str, Any]] = []

    async def satellite_segmentation(
        self, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        del use_cache
        self.calls.append(payload)
        latitude = float(payload["sat"]["latitude"])
        longitude = float(payload["sat"]["longitude"])
        requested_year = int(payload["date_time"]["start_date"][:4])
        image_year = requested_year if self.historical_images else 2026
        offset = abs(latitude - 39.01) + abs(longitude + 77.46)
        building = 20.0 + min(8.0, offset * 100)
        tree = 35.0 - min(8.0, offset * 100)
        if self.poor_controls and offset > 0:
            building = 0.0
            tree = 100.0
        return {
            "error": False,
            "data": {
                "activity_id": f"satellite-{len(self.calls)}",
                "status": "Completed",
                "result": {
                    "coordinates": {
                        "latitude": str(latitude),
                        "longitude": str(longitude),
                    },
                    "image_year": image_year,
                    "segmentation": {
                        "request_id": f"request-{len(self.calls)}",
                        "processing_time_seconds": 0.25,
                        "image_dimensions": {"width": 225, "height": 225},
                        "segments": {
                            "building": building,
                            "tree": tree,
                            "road, route": 15.0,
                            "earth, ground": 28.0,
                            "others": 2.0,
                        },
                        "image_content": "base64-is-never-normalized",
                    },
                    "original_image": ["base64-is-never-normalized"],
                },
            },
        }


def site() -> SiteInput:
    return SiteInput(
        name="Northern Virginia Demo Campus",
        latitude=39.01,
        longitude=-77.46,
    )


def test_live_satellite_context_selects_controls_without_claiming_false_history() -> (
    None
):
    client = FakeSatelliteClient(historical_images=False)
    provider = FortyGuardUrbanProvider(
        client,  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 26, tzinfo=timezone.utc),
    )

    result = asyncio.run(provider.analyze(site(), 2022, 2026, seed=7))

    assert result.data_class == DataClass.OBSERVED
    assert len(client.calls) == 6
    assert len(result.candidate_snapshots) == 4
    assert len(result.matched_controls) == 3
    assert result.metadata["historical_change_available"] is False
    assert result.metadata["baseline_image_year"] == 2026
    assert result.metadata["latest_image_year"] == 2026
    assert any("same satellite image year" in warning for warning in result.warnings)
    assert all(
        "image_content" not in snapshot.metadata
        for snapshot in [result.latest_snapshot, *result.candidate_snapshots]
    )


def test_distinct_image_years_enable_historical_land_cover_comparison() -> None:
    provider = FortyGuardUrbanProvider(
        FakeSatelliteClient(historical_images=True),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 26, tzinfo=timezone.utc),
    )

    result = asyncio.run(provider.analyze(site(), 2022, 2026, seed=7))

    assert result.metadata["historical_change_available"] is True
    assert result.baseline_snapshot is not None
    assert result.baseline_snapshot.image_year == 2022
    assert result.latest_snapshot.image_year == 2026


def test_low_similarity_controls_fail_the_quality_gate() -> None:
    provider = FortyGuardUrbanProvider(
        FakeSatelliteClient(poor_controls=True),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 26, tzinfo=timezone.utc),
    )

    result = asyncio.run(provider.analyze(site(), 2022, 2026, seed=7))

    assert result.metadata["control_quality_passed"] is False
    assert result.metadata["minimum_selected_similarity"] < 0.65
    assert any("falls back" in warning for warning in result.warnings)


def test_land_cover_labels_are_canonicalized_for_matching() -> None:
    groups = canonical_land_cover(
        {
            "building": 20,
            "road, route": 15,
            "tree": 35,
            "earth, ground": 28,
            "others": 2,
        }
    )

    assert groups == {
        "building": 20.0,
        "transport_surface": 15.0,
        "vegetation": 35.0,
        "bare_ground": 28.0,
        "water": 0.0,
        "other": 2.0,
    }
