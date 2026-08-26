from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import fmean
from typing import Any, Protocol
from uuid import uuid4

from ..models import DataClass, SiteInput
from .urban import (
    MatchedControl,
    SatelliteSnapshot,
    UrbanContextDataset,
    _candidate_coordinates,
    canonical_land_cover,
    land_cover_similarity,
    tree_canopy_share,
)


DYNAMIC_WORLD_DATASET = "GOOGLE/DYNAMICWORLD/V1"
DYNAMIC_WORLD_BANDS = (
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
)


class DynamicWorldError(RuntimeError):
    pass


@dataclass(frozen=True)
class DynamicWorldRecord:
    location_id: str
    year: int
    segments: dict[str, float]
    scene_count: int
    valid_pixel_count: int
    mean_valid_observations: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DynamicWorldGateway(Protocol):
    project: str
    dataset_id: str

    async def summarize(
        self,
        locations: dict[str, SiteInput],
        years: list[int],
        *,
        radius_m: float,
        workload_tag: str,
    ) -> list[DynamicWorldRecord]: ...


class EarthEngineDynamicWorldClient:
    """Small Earth Engine adapter that batches all AOI/year statistics in one call."""

    def __init__(
        self,
        project: str | None = None,
        *,
        dataset_id: str = DYNAMIC_WORLD_DATASET,
        season_start_month: int = 6,
        season_end_month: int = 8,
        scale_m: int = 10,
    ) -> None:
        self.project = (
            project
            or os.getenv("EARTH_ENGINE_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip()
        if not self.project:
            raise DynamicWorldError(
                "Dynamic World requires EARTH_ENGINE_PROJECT or GOOGLE_CLOUD_PROJECT"
            )
        if not 1 <= season_start_month <= season_end_month <= 12:
            raise ValueError(
                "Dynamic World seasonal months must be ordered within 1..12"
            )
        self.dataset_id = dataset_id
        self.season_start_month = season_start_month
        self.season_end_month = season_end_month
        self.scale_m = scale_m
        try:
            import ee  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DynamicWorldError(
                "Dynamic World requires the earthengine-api package"
            ) from exc
        self.ee = ee
        try:
            ee.Initialize(project=self.project)
        except Exception as exc:
            raise DynamicWorldError(
                "Earth Engine initialization failed; configure Application Default "
                "Credentials or a service account"
            ) from exc

    def _compute(
        self,
        locations: dict[str, SiteInput],
        years: list[int],
        *,
        radius_m: float,
        workload_tag: str,
    ) -> list[DynamicWorldRecord]:
        ee = self.ee
        features: list[Any] = []
        combined_reducer = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.count(), sharedInputs=True
        )
        for location_id, site in locations.items():
            geometry = ee.Geometry.Point([site.longitude, site.latitude]).buffer(
                radius_m
            )
            for year in years:
                start = f"{year}-{self.season_start_month:02d}-01"
                next_month = self.season_end_month + 1
                if next_month == 13:
                    end = f"{year + 1}-01-01"
                else:
                    end = f"{year}-{next_month:02d}-01"
                collection = (
                    ee.ImageCollection(self.dataset_id)
                    .filterBounds(geometry)
                    .filterDate(start, end)
                    .select(list(DYNAMIC_WORLD_BANDS))
                )
                composite = collection.mean()
                statistics = composite.reduceRegion(
                    reducer=combined_reducer,
                    geometry=geometry,
                    scale=self.scale_m,
                    maxPixels=10_000_000,
                    tileScale=2,
                )
                valid_observations = (
                    collection.select(["built"])
                    .count()
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=self.scale_m,
                        maxPixels=10_000_000,
                        tileScale=2,
                    )
                )
                properties = ee.Dictionary(statistics).combine(
                    {
                        "location_id": location_id,
                        "year": year,
                        "scene_count": collection.size(),
                        "mean_valid_observations": valid_observations.get("built"),
                        "first_acquisition_ms": collection.aggregate_min(
                            "system:time_start"
                        ),
                        "last_acquisition_ms": collection.aggregate_max(
                            "system:time_start"
                        ),
                    }
                )
                features.append(ee.Feature(None, properties))

        response = ee.data.computeFeatures(
            {
                "expression": ee.FeatureCollection(features),
                "pageSize": 1000,
                "workloadTag": workload_tag,
            }
        )
        raw_features = response.get("features", [])
        records: list[DynamicWorldRecord] = []
        for feature in raw_features:
            properties = feature.get("properties", {})
            segments: dict[str, float] = {}
            for band in DYNAMIC_WORLD_BANDS:
                value = properties.get(f"{band}_mean")
                if value is not None:
                    segments[band] = round(float(value) * 100.0, 6)
            if len(segments) != len(DYNAMIC_WORLD_BANDS):
                continue
            probability_total = sum(segments.values())
            if not 95.0 <= probability_total <= 105.0:
                continue
            first_ms = properties.get("first_acquisition_ms")
            last_ms = properties.get("last_acquisition_ms")
            records.append(
                DynamicWorldRecord(
                    location_id=str(properties["location_id"]),
                    year=int(properties["year"]),
                    segments=segments,
                    scene_count=int(properties.get("scene_count") or 0),
                    valid_pixel_count=int(properties.get("built_count") or 0),
                    mean_valid_observations=round(
                        float(properties.get("mean_valid_observations") or 0.0), 3
                    ),
                    metadata={
                        "first_acquisition": _timestamp_to_iso(first_ms),
                        "last_acquisition": _timestamp_to_iso(last_ms),
                    },
                )
            )
        return records

    async def summarize(
        self,
        locations: dict[str, SiteInput],
        years: list[int],
        *,
        radius_m: float,
        workload_tag: str,
    ) -> list[DynamicWorldRecord]:
        try:
            return await asyncio.to_thread(
                self._compute,
                locations,
                years,
                radius_m=radius_m,
                workload_tag=workload_tag,
            )
        except DynamicWorldError:
            raise
        except Exception as exc:
            raise DynamicWorldError(
                f"Earth Engine Dynamic World computation failed: {type(exc).__name__}"
            ) from exc


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).isoformat()


class DynamicWorldUrbanProvider:
    """Historical 10 m land-cover context with complete site/control coverage gates."""

    def __init__(
        self,
        client: DynamicWorldGateway | None = None,
        *,
        project: str | None = None,
        control_radius_km: float = 3.0,
        aoi_radius_m: float = 500.0,
        selected_control_count: int = 3,
        control_match_threshold: float = 0.65,
        minimum_scene_count: int = 3,
        minimum_valid_pixels: int = 100,
    ) -> None:
        self.client = client or EarthEngineDynamicWorldClient(project)
        self.control_radius_km = control_radius_km
        self.aoi_radius_m = aoi_radius_m
        self.selected_control_count = selected_control_count
        self.control_match_threshold = control_match_threshold
        self.minimum_scene_count = minimum_scene_count
        self.minimum_valid_pixels = minimum_valid_pixels
        self.source = (
            "https://earthengine.googleapis.com/v1/projects/"
            f"{self.client.project}/table:computeFeatures"
        )

    def _usable(self, record: DynamicWorldRecord | None) -> bool:
        return bool(
            record
            and record.scene_count >= self.minimum_scene_count
            and record.valid_pixel_count >= self.minimum_valid_pixels
        )

    @staticmethod
    def _snapshot(
        label: str,
        site: SiteInput,
        record: DynamicWorldRecord,
        workload_tag: str,
    ) -> SatelliteSnapshot:
        return SatelliteSnapshot(
            label=label,
            latitude=site.latitude,
            longitude=site.longitude,
            requested_date=f"{record.year}-06-01/{record.year}-09-01",
            image_year=record.year,
            segments=record.segments,
            activity_id=workload_tag,
            metadata={
                "scene_count": record.scene_count,
                "valid_pixel_count": record.valid_pixel_count,
                "mean_valid_observations": record.mean_valid_observations,
                **record.metadata,
            },
        )

    async def analyze(
        self, site: SiteInput, baseline_year: int, end_year: int, *, seed: int
    ) -> UrbanContextDataset:
        del seed
        latest_year = min(end_year, datetime.now(timezone.utc).year)
        years = list(range(baseline_year, latest_year + 1))
        coordinates = _candidate_coordinates(site, self.control_radius_km)
        locations: dict[str, SiteInput] = {"site": site}
        for direction, latitude, longitude in coordinates:
            locations[f"control_{direction.lower()}"] = SiteInput(
                name=f"{site.name} regional control {direction}",
                latitude=latitude,
                longitude=longitude,
                timezone=site.timezone,
            )
        workload_tag = f"fortycool-{uuid4().hex[:20]}"
        records = await self.client.summarize(
            locations,
            years,
            radius_m=self.aoi_radius_m,
            workload_tag=workload_tag,
        )
        by_location_year = {
            (record.location_id, record.year): record for record in records
        }
        site_baseline = by_location_year.get(("site", baseline_year))
        site_latest = by_location_year.get(("site", latest_year))
        if not self._usable(site_baseline) or not self._usable(site_latest):
            raise DynamicWorldError(
                "Dynamic World did not return usable baseline and latest site coverage"
            )
        assert site_baseline is not None and site_latest is not None
        baseline_snapshot = self._snapshot(
            "site_baseline", site, site_baseline, workload_tag
        )
        latest_snapshot = self._snapshot("site_latest", site, site_latest, workload_tag)

        candidates: list[SatelliteSnapshot] = []
        matched: list[MatchedControl] = []
        missing_records: list[str] = []
        for direction, _, _ in coordinates:
            location_id = f"control_{direction.lower()}"
            control_site = locations[location_id]
            latest_record = by_location_year.get((location_id, latest_year))
            if not self._usable(latest_record):
                missing_records.append(f"{location_id}:{latest_year}")
                continue
            assert latest_record is not None
            snapshot = self._snapshot(
                f"candidate_{direction.lower()}",
                control_site,
                latest_record,
                workload_tag,
            )
            candidates.append(snapshot)
            complete = True
            for year in years:
                if not self._usable(by_location_year.get((location_id, year))):
                    missing_records.append(f"{location_id}:{year}")
                    complete = False
            if complete:
                matched.append(
                    MatchedControl(
                        site=control_site,
                        snapshot=snapshot,
                        similarity_score=land_cover_similarity(
                            latest_snapshot, snapshot
                        ),
                        distance_km=self.control_radius_km,
                    )
                )
        if not matched:
            raise DynamicWorldError(
                "Dynamic World returned no control with complete historical coverage"
            )
        matched.sort(key=lambda item: item.similarity_score, reverse=True)
        selected = matched[: self.selected_control_count]
        minimum_similarity = min(
            (control.similarity_score for control in selected), default=0.0
        )
        complete_site_years = [
            year for year in years if self._usable(by_location_year.get(("site", year)))
        ]
        full_history_available = len(complete_site_years) == len(years)
        control_quality_passed = bool(
            len(selected) == self.selected_control_count
            and minimum_similarity >= self.control_match_threshold
            and full_history_available
        )

        annual_series: list[dict[str, float | int]] = []
        if len(selected) == self.selected_control_count:
            selected_location_ids = [
                _location_id_for_control(control.site, locations)
                for control in selected
            ]
            for year in complete_site_years:
                site_record = by_location_year[("site", year)]
                control_records = [
                    by_location_year[(location_id, year)]
                    for location_id in selected_location_ids
                ]
                site_groups = canonical_land_cover(site_record.segments)
                control_groups = [
                    canonical_land_cover(record.segments) for record in control_records
                ]
                site_built = site_groups["building"] + site_groups["transport_surface"]
                control_built = fmean(
                    group["building"] + group["transport_surface"]
                    for group in control_groups
                )
                control_trees = fmean(
                    tree_canopy_share(record.segments) for record in control_records
                )
                control_bare = fmean(group["bare_ground"] for group in control_groups)
                annual_series.append(
                    {
                        "year": year,
                        "site_built_surface_percent": round(site_built, 4),
                        "control_built_surface_percent": round(control_built, 4),
                        "local_excess_built_surface_percent": round(
                            site_built - control_built, 4
                        ),
                        "site_tree_canopy_percent": tree_canopy_share(
                            site_record.segments
                        ),
                        "control_tree_canopy_percent": round(control_trees, 4),
                        "local_excess_tree_canopy_percent": round(
                            tree_canopy_share(site_record.segments) - control_trees, 4
                        ),
                        "site_bare_ground_percent": round(
                            site_groups["bare_ground"], 4
                        ),
                        "control_bare_ground_percent": round(control_bare, 4),
                    }
                )

        warnings = [
            "Dynamic World values are modeled land-cover probabilities derived from "
            "Sentinel-2, not a surveyed inventory.",
            "Land-cover co-movement can support an attribution hypothesis but does not "
            "by itself establish causality for thermal drift.",
        ]
        if missing_records or not full_history_available:
            warnings.append(
                "Some Dynamic World AOI/year records failed scene or valid-pixel coverage gates."
            )
        if not control_quality_passed:
            warnings.append(
                "Dynamic World controls did not meet complete-history and similarity gates; "
                "thermal drift falls back to the disclosed local outer ring."
            )
        return UrbanContextDataset(
            baseline_snapshot=baseline_snapshot,
            latest_snapshot=latest_snapshot,
            candidate_snapshots=candidates,
            matched_controls=selected,
            data_class=DataClass.OBSERVED,
            source=self.source,
            activity_ids=[workload_tag],
            metadata={
                "endpoint": f"/v1/projects/{self.client.project}/table:computeFeatures",
                "dataset_id": self.client.dataset_id,
                "land_cover_provider": "google_dynamic_world_v1",
                "workload_tag": workload_tag,
                "historical_change_available": True,
                "historical_control_stability_verified": full_history_available,
                "candidate_count": len(candidates),
                "selected_control_count": len(selected),
                "control_radius_km": self.control_radius_km,
                "aoi_radius_m": self.aoi_radius_m,
                "resolution_m": 10,
                "seasonal_window": "June-August",
                "control_match_threshold": self.control_match_threshold,
                "minimum_selected_similarity": minimum_similarity,
                "control_quality_passed": control_quality_passed,
                "baseline_image_year": baseline_year,
                "latest_image_year": latest_year,
                "complete_history_years": complete_site_years,
                "annual_land_cover_series": annual_series,
                "missing_records": sorted(set(missing_records)),
                "minimum_scene_count": self.minimum_scene_count,
                "minimum_valid_pixels": self.minimum_valid_pixels,
            },
            warnings=warnings,
        )


def _location_id_for_control(
    control: SiteInput, locations: dict[str, SiteInput]
) -> str:
    for location_id, candidate in locations.items():
        if (
            candidate.latitude == control.latitude
            and candidate.longitude == control.longitude
        ):
            return location_id
    raise DynamicWorldError("Selected Dynamic World control was not found in its batch")
