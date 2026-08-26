from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import fmean
from typing import Any, Callable, Protocol

from ..models import DataClass, SiteInput
from .fortyguard import FortyGuardClient, FortyGuardError


@dataclass(frozen=True)
class SatelliteSnapshot:
    label: str
    latitude: float
    longitude: float
    requested_date: str
    image_year: int | None
    segments: dict[str, float]
    activity_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchedControl:
    site: SiteInput
    snapshot: SatelliteSnapshot
    similarity_score: float
    distance_km: float


@dataclass(frozen=True)
class UrbanContextDataset:
    baseline_snapshot: SatelliteSnapshot | None
    latest_snapshot: SatelliteSnapshot
    candidate_snapshots: list[SatelliteSnapshot]
    matched_controls: list[MatchedControl]
    data_class: DataClass
    source: str
    activity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class UrbanContextProvider(Protocol):
    async def analyze(
        self, site: SiteInput, baseline_year: int, end_year: int, *, seed: int
    ) -> UrbanContextDataset: ...


def canonical_land_cover(segments: dict[str, float]) -> dict[str, float]:
    """Map model-specific labels into stable comparison groups."""

    groups = {
        "building": 0.0,
        "transport_surface": 0.0,
        "vegetation": 0.0,
        "bare_ground": 0.0,
        "water": 0.0,
        "other": 0.0,
    }
    for raw_label, raw_value in segments.items():
        label = raw_label.strip().lower()
        value = float(raw_value)
        if any(token in label for token in ("building", "structure", "built")):
            groups["building"] += value
        elif any(
            token in label
            for token in ("road", "route", "pavement", "parking", "sidewalk")
        ):
            groups["transport_surface"] += value
        elif any(
            token in label
            for token in (
                "tree",
                "vegetation",
                "grass",
                "plant",
                "forest",
                "crop",
                "shrub",
                "flooded_vegetation",
            )
        ):
            groups["vegetation"] += value
        elif any(
            token in label for token in ("earth", "ground", "soil", "sand", "bare")
        ):
            groups["bare_ground"] += value
        elif "water" in label:
            groups["water"] += value
        else:
            groups["other"] += value
    return {key: round(value, 4) for key, value in groups.items()}


def tree_canopy_share(segments: dict[str, float]) -> float:
    return round(
        sum(
            float(value)
            for raw_label, value in segments.items()
            if any(token in raw_label.strip().lower() for token in ("tree", "forest"))
        ),
        4,
    )


def land_cover_similarity(
    reference: SatelliteSnapshot, candidate: SatelliteSnapshot
) -> float:
    reference_groups = canonical_land_cover(reference.segments)
    candidate_groups = canonical_land_cover(candidate.segments)
    l1_distance = sum(
        abs(reference_groups[key] - candidate_groups[key]) for key in reference_groups
    )
    return round(max(0.0, 1.0 - l1_distance / 200.0), 4)


class FixtureUrbanProvider:
    """Deterministic fixture for developing the complete urban-context workflow."""

    source = "fixture://fortyguard-shaped-satellite-segmentation/v1"

    @staticmethod
    def _snapshot(
        label: str,
        site: SiteInput,
        requested_date: str,
        image_year: int,
        segments: dict[str, float],
        activity_id: str,
    ) -> SatelliteSnapshot:
        return SatelliteSnapshot(
            label=label,
            latitude=site.latitude,
            longitude=site.longitude,
            requested_date=requested_date,
            image_year=image_year,
            segments=segments,
            activity_id=activity_id,
        )

    async def analyze(
        self, site: SiteInput, baseline_year: int, end_year: int, *, seed: int
    ) -> UrbanContextDataset:
        baseline_date = f"{baseline_year}-07-15"
        latest_date = f"{end_year}-07-15"
        baseline = self._snapshot(
            "site_baseline",
            site,
            baseline_date,
            baseline_year,
            {
                "building": 18.0,
                "road, route": 12.0,
                "tree": 36.0,
                "earth, ground": 32.0,
                "others": 2.0,
            },
            f"fixture-satellite-site-{baseline_year}-{seed}",
        )
        latest = self._snapshot(
            "site_latest",
            site,
            latest_date,
            end_year,
            {
                "building": 25.0,
                "road, route": 15.0,
                "tree": 28.0,
                "earth, ground": 30.0,
                "others": 2.0,
            },
            f"fixture-satellite-site-{end_year}-{seed}",
        )
        candidates: list[SatelliteSnapshot] = []
        controls: list[MatchedControl] = []
        candidate_segments = [
            {
                "building": 23.0,
                "road, route": 14.0,
                "tree": 31.0,
                "earth, ground": 30.0,
                "others": 2.0,
            },
            {
                "building": 26.0,
                "road, route": 14.0,
                "tree": 27.0,
                "earth, ground": 31.0,
                "others": 2.0,
            },
            {
                "building": 20.0,
                "road, route": 12.0,
                "tree": 38.0,
                "earth, ground": 28.0,
                "others": 2.0,
            },
            {
                "building": 11.0,
                "road, route": 8.0,
                "tree": 55.0,
                "earth, ground": 24.0,
                "others": 2.0,
            },
        ]
        for index, (direction, latitude, longitude) in enumerate(
            _candidate_coordinates(site), start=1
        ):
            control_site = SiteInput(
                name=f"{site.name} regional control {direction}",
                latitude=latitude,
                longitude=longitude,
                timezone=site.timezone,
            )
            snapshot = self._snapshot(
                f"candidate_{direction.lower()}",
                control_site,
                latest_date,
                end_year,
                candidate_segments[index - 1],
                f"fixture-satellite-control-{index}-{seed}",
            )
            candidates.append(snapshot)
            controls.append(
                MatchedControl(
                    site=control_site,
                    snapshot=snapshot,
                    similarity_score=land_cover_similarity(latest, snapshot),
                    distance_km=3.0,
                )
            )
        controls.sort(key=lambda item: item.similarity_score, reverse=True)
        selected = controls[:3]
        match_threshold = 0.65
        control_quality_passed = bool(
            selected
            and min(control.similarity_score for control in selected) >= match_threshold
        )
        baseline_groups = canonical_land_cover(baseline.segments)
        latest_groups = canonical_land_cover(latest.segments)
        latest_control_groups = [
            canonical_land_cover(control.snapshot.segments) for control in selected
        ]
        annual_land_cover_series: list[dict[str, float | int]] = []
        year_span = max(1, end_year - baseline_year)
        for year in range(baseline_year, end_year + 1):
            progress = (year - baseline_year) / year_span
            site_built = (
                baseline_groups["building"]
                + baseline_groups["transport_surface"]
                + progress
                * (
                    latest_groups["building"]
                    + latest_groups["transport_surface"]
                    - baseline_groups["building"]
                    - baseline_groups["transport_surface"]
                )
            )
            baseline_trees = tree_canopy_share(baseline.segments)
            latest_trees = tree_canopy_share(latest.segments)
            site_trees = baseline_trees + progress * (
                latest_trees - baseline_trees
            )
            control_latest_built = fmean(
                group["building"] + group["transport_surface"]
                for group in latest_control_groups
            )
            control_latest_trees = fmean(
                tree_canopy_share(control.snapshot.segments) for control in selected
            )
            control_built = (control_latest_built - 4.0) + progress * 4.0
            control_trees = (control_latest_trees + 3.0) - progress * 3.0
            annual_land_cover_series.append(
                {
                    "year": year,
                    "site_built_surface_percent": round(site_built, 4),
                    "control_built_surface_percent": round(control_built, 4),
                    "local_excess_built_surface_percent": round(
                        site_built - control_built, 4
                    ),
                    "site_tree_canopy_percent": round(site_trees, 4),
                    "control_tree_canopy_percent": round(control_trees, 4),
                    "local_excess_tree_canopy_percent": round(
                        site_trees - control_trees, 4
                    ),
                }
            )
        return UrbanContextDataset(
            baseline_snapshot=baseline,
            latest_snapshot=latest,
            candidate_snapshots=candidates,
            matched_controls=selected,
            data_class=DataClass.SIMULATED,
            source=self.source,
            activity_ids=[
                baseline.activity_id or "",
                latest.activity_id or "",
                *(item.snapshot.activity_id or "" for item in selected),
            ],
            metadata={
                "historical_change_available": True,
                "candidate_count": len(candidates),
                "selected_control_count": len(selected),
                "control_radius_km": 3.0,
                "control_match_threshold": match_threshold,
                "minimum_selected_similarity": min(
                    control.similarity_score for control in selected
                ),
                "control_quality_passed": control_quality_passed,
                "historical_control_stability_verified": True,
                "annual_land_cover_series": annual_land_cover_series,
                "complete_history_years": list(range(baseline_year, end_year + 1)),
                "land_cover_provider": "fixture_dynamic_world_shaped",
            },
            warnings=[
                "Urban land-cover history and matched controls are simulated in fixture mode."
            ],
        )


def _candidate_coordinates(
    site: SiteInput, radius_km: float = 3.0
) -> list[tuple[str, float, float]]:
    latitude_delta = radius_km * 1000 / 111_320.0
    longitude_scale = max(0.2, math.cos(math.radians(site.latitude)))
    longitude_delta = radius_km * 1000 / (111_320.0 * longitude_scale)
    return [
        ("N", site.latitude + latitude_delta, site.longitude),
        ("E", site.latitude, site.longitude + longitude_delta),
        ("S", site.latitude - latitude_delta, site.longitude),
        ("W", site.latitude, site.longitude - longitude_delta),
    ]


class FortyGuardUrbanProvider:
    """Retrieve real satellite segmentation and select comparable control zones."""

    source = "https://api.fortyguard.com/v1/satellite"

    def __init__(
        self,
        client: FortyGuardClient | None = None,
        *,
        granularity_m: int = 60,
        control_radius_km: float = 3.0,
        selected_control_count: int = 3,
        control_match_threshold: float = 0.65,
        max_concurrency: int = 3,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if granularity_m not in {60, 80, 100}:
            raise ValueError("FortyGuard granularity must be 60, 80, or 100 meters")
        self.client = client or FortyGuardClient()
        self.granularity_m = granularity_m
        self.control_radius_km = control_radius_km
        self.selected_control_count = selected_control_count
        self.control_match_threshold = control_match_threshold
        self.max_concurrency = max_concurrency
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _payload(
        self, latitude: float, longitude: float, requested_date: date
    ) -> dict[str, Any]:
        return {
            "sat": {"latitude": latitude, "longitude": longitude},
            "date_time": {
                "start_date": requested_date.isoformat(),
                "start_time": "14:00",
                "filter_type": 1,
            },
            "granularity": self.granularity_m,
        }

    @staticmethod
    def _normalize(
        response: dict[str, Any], *, label: str, requested_date: date
    ) -> SatelliteSnapshot:
        data = response.get("data", {})
        result = data.get("result", {})
        coordinates = result.get("coordinates", {})
        segmentation = result.get("segmentation", {})
        raw_segments = segmentation.get("segments", {})
        if not isinstance(raw_segments, dict) or not raw_segments:
            raise FortyGuardError("FortyGuard satellite result contained no segments")
        try:
            segments = {str(key): float(value) for key, value in raw_segments.items()}
            latitude = float(coordinates["latitude"])
            longitude = float(coordinates["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FortyGuardError(
                "FortyGuard satellite result schema was invalid"
            ) from exc
        image_year_raw = result.get("image_year")
        image_year = int(image_year_raw) if image_year_raw is not None else None
        activity_id = data.get("activity_id")
        return SatelliteSnapshot(
            label=label,
            latitude=latitude,
            longitude=longitude,
            requested_date=requested_date.isoformat(),
            image_year=image_year,
            segments=segments,
            activity_id=str(activity_id) if activity_id else None,
            metadata={
                "request_id": segmentation.get("request_id"),
                "processing_time_seconds": segmentation.get("processing_time_seconds"),
                "image_dimensions": segmentation.get("image_dimensions"),
            },
        )

    async def analyze(
        self, site: SiteInput, baseline_year: int, end_year: int, *, seed: int
    ) -> UrbanContextDataset:
        del seed
        baseline_date = date(baseline_year, 7, 15)
        latest_allowed_year = min(end_year, self.clock().astimezone(timezone.utc).year)
        latest_date = date(latest_allowed_year, 7, 15)
        coordinates = _candidate_coordinates(site, self.control_radius_km)
        requests: list[tuple[str, float, float, date]] = [
            ("site_baseline", site.latitude, site.longitude, baseline_date),
            ("site_latest", site.latitude, site.longitude, latest_date),
            *[
                (f"candidate_{direction.lower()}", latitude, longitude, latest_date)
                for direction, latitude, longitude in coordinates
            ],
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def retrieve(
            label: str, latitude: float, longitude: float, requested_date: date
        ) -> SatelliteSnapshot:
            async with semaphore:
                response = await self.client.satellite_segmentation(
                    self._payload(latitude, longitude, requested_date)
                )
            return self._normalize(response, label=label, requested_date=requested_date)

        results = await asyncio.gather(
            *(
                retrieve(label, latitude, longitude, requested_date)
                for label, latitude, longitude, requested_date in requests
            ),
            return_exceptions=True,
        )
        normalized = {
            request[0]: result
            for request, result in zip(requests, results)
            if isinstance(result, SatelliteSnapshot)
        }
        failures = [
            f"{request[0]}: {type(result).__name__}"
            for request, result in zip(requests, results)
            if isinstance(result, Exception)
        ]
        latest = normalized.get("site_latest")
        if latest is None:
            detail = ", ".join(failures) if failures else "site_latest unavailable"
            raise FortyGuardError(f"FortyGuard satellite context unavailable: {detail}")
        candidates = [
            snapshot
            for label, snapshot in normalized.items()
            if label.startswith("candidate_")
        ]
        if not candidates:
            raise FortyGuardError(
                "FortyGuard returned no usable satellite control candidates"
            )

        candidate_by_label = {snapshot.label: snapshot for snapshot in candidates}
        matched: list[MatchedControl] = []
        for direction, latitude, longitude in coordinates:
            snapshot = candidate_by_label.get(f"candidate_{direction.lower()}")
            if snapshot is None:
                continue
            control_site = SiteInput(
                name=f"{site.name} regional control {direction}",
                latitude=latitude,
                longitude=longitude,
                timezone=site.timezone,
            )
            matched.append(
                MatchedControl(
                    site=control_site,
                    snapshot=snapshot,
                    similarity_score=land_cover_similarity(latest, snapshot),
                    distance_km=self.control_radius_km,
                )
            )
        matched.sort(key=lambda item: item.similarity_score, reverse=True)
        selected = matched[: self.selected_control_count]
        minimum_selected_similarity = min(
            (control.similarity_score for control in selected), default=0.0
        )
        control_quality_passed = bool(
            len(selected) == self.selected_control_count
            and minimum_selected_similarity >= self.control_match_threshold
        )
        baseline = normalized.get("site_baseline")
        historical_change_available = bool(
            baseline
            and baseline.image_year is not None
            and latest.image_year is not None
            and baseline.image_year != latest.image_year
        )
        warnings: list[str] = []
        if failures:
            warnings.append(
                "Some satellite context requests failed: " + ", ".join(failures)
            )
        if not historical_change_available:
            warnings.append(
                "FortyGuard returned the same satellite image year for baseline and latest "
                "requests; historical land-cover change is unavailable and is not inferred."
            )
        warnings.append(
            "Regional control candidates are scored on current satellite land cover; "
            "historical control stability is not yet verified by the Satellite endpoint."
        )
        if not control_quality_passed:
            warnings.append(
                "Regional satellite-control candidates did not meet the configured similarity "
                "threshold; thermal drift falls back to the disclosed local outer ring."
            )
        activity_ids = [
            snapshot.activity_id
            for snapshot in [baseline, latest, *candidates]
            if snapshot is not None and snapshot.activity_id
        ]
        return UrbanContextDataset(
            baseline_snapshot=baseline,
            latest_snapshot=latest,
            candidate_snapshots=candidates,
            matched_controls=selected,
            data_class=DataClass.OBSERVED,
            source=self.source,
            activity_ids=activity_ids,
            metadata={
                "historical_change_available": historical_change_available,
                "candidate_count": len(candidates),
                "selected_control_count": len(selected),
                "control_radius_km": self.control_radius_km,
                "control_match_threshold": self.control_match_threshold,
                "minimum_selected_similarity": minimum_selected_similarity,
                "control_quality_passed": control_quality_passed,
                "baseline_requested_year": baseline_year,
                "latest_requested_year": latest_allowed_year,
                "baseline_image_year": baseline.image_year if baseline else None,
                "latest_image_year": latest.image_year,
                "failed_requests": failures,
            },
            warnings=warnings,
        )
