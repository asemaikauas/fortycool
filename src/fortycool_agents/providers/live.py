from __future__ import annotations

import asyncio
import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Any, Callable

from ..models import DataClass, SiteInput
from .fixture import (
    AnnualThermalDataset,
    AnnualThermalSummary,
    FixtureThermalProvider,
    ThermalDataset,
    ThermalSample,
)
from .fortyguard import FortyGuardClient, FortyGuardError


@dataclass(frozen=True)
class SpatialAggregate:
    site_value: float
    control_value: float
    site_tile_count: int
    control_tile_count: int
    control_zone_count: int = 1
    control_tile_counts: tuple[int, ...] = ()


class FortyGuardThermalProvider:
    """Normalize live FortyGuard heatmaps for the FortyCool agent workflows.

    The provider intentionally remains hybrid. FortyGuard supplies the spatial
    temperatures used for the live forecast and historical screening windows;
    dense BMS-training weather and environmental fields that are not requested
    from the API remain explicit fixture data.
    """

    source = "https://api.fortyguard.com/v1/heatmap"

    def __init__(
        self,
        client: FortyGuardClient | None = None,
        *,
        fallback: FixtureThermalProvider | None = None,
        granularity_m: int = 60,
        core_radius_m: float = 250.0,
        control_min_radius_m: float = 450.0,
        max_concurrency: int = 3,
        annual_reference_month: int = 7,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if granularity_m not in {60, 80, 100}:
            raise ValueError("FortyGuard granularity must be 60, 80, or 100 meters")
        if not 1 <= annual_reference_month <= 12:
            raise ValueError("annual_reference_month must be between 1 and 12")
        self.client = client or FortyGuardClient()
        self.fallback = fallback or FixtureThermalProvider()
        self.granularity_m = granularity_m
        self.core_radius_m = core_radius_m
        self.control_min_radius_m = control_min_radius_m
        self.max_concurrency = max_concurrency
        self.annual_reference_month = annual_reference_month
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _generated_aoi(site: SiteInput) -> dict[str, Any]:
        half_height_m = 450.0
        half_width_m = 550.0
        latitude_delta = half_height_m / 111_320.0
        longitude_scale = max(0.2, math.cos(math.radians(site.latitude)))
        longitude_delta = half_width_m / (111_320.0 * longitude_scale)
        west = site.longitude - longitude_delta
        east = site.longitude + longitude_delta
        south = site.latitude - latitude_delta
        north = site.latitude + latitude_delta
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [west, south],
                                [east, south],
                                [east, north],
                                [west, north],
                                [west, south],
                            ]
                        ],
                    },
                }
            ],
        }

    @classmethod
    def _aoi(cls, site: SiteInput) -> dict[str, Any]:
        if site.aoi is None:
            return cls._generated_aoi(site)
        aoi_type = site.aoi.get("type")
        if aoi_type == "FeatureCollection":
            return site.aoi
        if aoi_type == "Feature":
            return {"type": "FeatureCollection", "features": [site.aoi]}
        if aoi_type == "Polygon":
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": site.aoi,
                    }
                ],
            }
        raise FortyGuardError(
            "site.aoi must be a GeoJSON Polygon, Feature, or FeatureCollection"
        )

    @staticmethod
    def _result(
        response: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
        data = response.get("data", {})
        result = data.get("result", {})
        map_data = result.get("map_data", {})
        features = map_data.get("features", [])
        if not isinstance(features, list):
            raise FortyGuardError("FortyGuard map_data.features was not a list")
        activity_id = data.get("activity_id")
        return str(activity_id) if activity_id else None, map_data, features

    @staticmethod
    def _centroid(feature: dict[str, Any]) -> tuple[float, float]:
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            raise FortyGuardError("FortyGuard heatmap feature was not a Polygon")
        coordinates = geometry.get("coordinates", [])
        if not coordinates or not coordinates[0]:
            raise FortyGuardError("FortyGuard heatmap polygon had no coordinates")
        ring = coordinates[0]
        points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
        longitude = fmean(float(point[0]) for point in points)
        latitude = fmean(float(point[1]) for point in points)
        return latitude, longitude

    @staticmethod
    def _distance_m(
        latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
    ) -> float:
        radius_m = 6_371_000.0
        phi_a = math.radians(latitude_a)
        phi_b = math.radians(latitude_b)
        delta_phi = math.radians(latitude_b - latitude_a)
        delta_lambda = math.radians(longitude_b - longitude_a)
        value = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
        )
        return (
            radius_m * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value)))
        )

    def _spatial_aggregate(
        self, features: list[dict[str, Any]], site: SiteInput, value_key: str
    ) -> SpatialAggregate:
        values: list[tuple[float, float]] = []
        for feature in features:
            raw_value = feature.get("properties", {}).get(value_key)
            if raw_value is None:
                continue
            latitude, longitude = self._centroid(feature)
            distance = self._distance_m(
                site.latitude, site.longitude, latitude, longitude
            )
            values.append((distance, float(raw_value)))
        if len(values) < 2:
            raise FortyGuardError(
                f"FortyGuard heatmap did not contain enough {value_key} tiles"
            )

        site_values = [
            value for distance, value in values if distance <= self.core_radius_m
        ]
        control_values = [
            value for distance, value in values if distance >= self.control_min_radius_m
        ]
        ordered = sorted(values)
        fallback_count = max(1, len(ordered) // 4)
        if not site_values:
            site_values = [value for _, value in ordered[:fallback_count]]
        if not control_values:
            control_values = [value for _, value in ordered[-fallback_count:]]
        return SpatialAggregate(
            site_value=fmean(site_values),
            control_value=fmean(control_values),
            site_tile_count=len(site_values),
            control_tile_count=len(control_values),
            control_tile_counts=(len(control_values),),
        )

    def _matched_spatial_aggregate(
        self,
        features: list[dict[str, Any]],
        site: SiteInput,
        controls: list[SiteInput],
        value_key: str,
    ) -> SpatialAggregate:
        values: list[tuple[float, float, float]] = []
        for feature in features:
            raw_value = feature.get("properties", {}).get(value_key)
            if raw_value is None:
                continue
            latitude, longitude = self._centroid(feature)
            values.append((latitude, longitude, float(raw_value)))
        site_values = [
            value
            for latitude, longitude, value in values
            if self._distance_m(site.latitude, site.longitude, latitude, longitude)
            <= self.core_radius_m
        ]
        if not site_values:
            raise FortyGuardError(
                f"FortyGuard heatmap contained no site {value_key} tiles"
            )
        control_zones: list[list[float]] = []
        for control in controls:
            zone = [
                value
                for latitude, longitude, value in values
                if self._distance_m(
                    control.latitude, control.longitude, latitude, longitude
                )
                <= self.core_radius_m
            ]
            if not zone:
                raise FortyGuardError(
                    f"FortyGuard heatmap contained no {value_key} tiles for {control.name}"
                )
            control_zones.append(zone)
        zone_means = [fmean(zone) for zone in control_zones]
        return SpatialAggregate(
            site_value=fmean(site_values),
            control_value=fmean(zone_means),
            site_tile_count=len(site_values),
            control_tile_count=sum(len(zone) for zone in control_zones),
            control_zone_count=len(control_zones),
            control_tile_counts=tuple(len(zone) for zone in control_zones),
        )

    @staticmethod
    def _combined_aoi(site: SiteInput, controls: list[SiteInput]) -> dict[str, Any]:
        margin_m = 450.0
        latitudes = [site.latitude, *(control.latitude for control in controls)]
        longitudes = [site.longitude, *(control.longitude for control in controls)]
        latitude_delta = margin_m / 111_320.0
        center_latitude = fmean(latitudes)
        longitude_scale = max(0.2, math.cos(math.radians(center_latitude)))
        longitude_delta = margin_m / (111_320.0 * longitude_scale)
        west = min(longitudes) - longitude_delta
        east = max(longitudes) + longitude_delta
        south = min(latitudes) - latitude_delta
        north = max(latitudes) + latitude_delta
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [west, south],
                                [east, south],
                                [east, north],
                                [west, north],
                                [west, south],
                            ]
                        ],
                    },
                }
            ],
        }

    def _heatmap_payload(
        self,
        site: SiteInput,
        *,
        start: datetime,
        filter_type: int,
        analytic_type: str = "tcm",
        end: datetime | None = None,
        threshold_c: float | None = None,
        polygon_aoi: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_utc = start.astimezone(timezone.utc)
        date_time: dict[str, Any] = {
            "start_date": start_utc.strftime("%Y-%m-%d"),
            "filter_type": filter_type,
        }
        if filter_type in {1, 2}:
            date_time["start_time"] = start_utc.strftime("%H:%M")
        if end is not None:
            end_utc = end.astimezone(timezone.utc)
            if filter_type == 2:
                date_time["end_time"] = end_utc.strftime("%H:%M")
            elif filter_type == 4:
                date_time["end_date"] = end_utc.strftime("%Y-%m-%d")
        payload: dict[str, Any] = {
            "polygon_aoi": polygon_aoi or self._aoi(site),
            "date_time": date_time,
            "granularity": self.granularity_m,
            "analytic_type": analytic_type,
        }
        if analytic_type in {"exceedance", "persistence"}:
            payload.update({"threshold": threshold_c, "direction": "below"})
        return payload

    async def history(
        self, site: SiteInput, start: datetime, end: datetime, *, seed: int
    ) -> ThermalDataset:
        fallback = await self.fallback.history(site, start, end, seed=seed)
        return ThermalDataset(
            samples=fallback.samples,
            data_class=DataClass.SIMULATED,
            source=fallback.source,
            activity_ids=fallback.activity_ids,
            metadata={
                "provider_mode": "hybrid",
                "purpose": "dense_bms_training_weather",
                "temperature_fields_data_class": DataClass.SIMULATED.value,
            },
            warnings=[
                "Dense historical weather used to train the BMS digital twin is simulated; "
                "live FortyGuard data is used separately for forecast and drift screening."
            ],
        )

    async def forecast(
        self, site: SiteInput, hours: int, *, seed: int
    ) -> ThermalDataset:
        start = (
            self.clock()
            .astimezone(timezone.utc)
            .replace(minute=0, second=0, microsecond=0)
        )
        start += timedelta(hours=1)
        timestamps = [start + timedelta(hours=index) for index in range(hours)]
        fallback = await self.fallback.history(
            site, start, start + timedelta(hours=hours), seed=seed + 1000
        )
        fallback_by_timestamp = {
            sample.timestamp: sample for sample in fallback.samples
        }
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def retrieve(timestamp: datetime) -> dict[str, Any]:
            async with semaphore:
                return await self.client.create_heatmap(
                    self._heatmap_payload(site, start=timestamp, filter_type=1)
                )

        responses = await asyncio.gather(
            *(retrieve(timestamp) for timestamp in timestamps), return_exceptions=True
        )
        samples: list[ThermalSample] = []
        activity_ids: list[str] = []
        attempted_activity_ids: list[str] = []
        heatmap: dict[str, Any] | None = None
        observed_hours = 0
        failures: list[str] = []
        tile_counts: list[dict[str, int]] = []
        for timestamp, response in zip(timestamps, responses):
            fixture_sample = fallback_by_timestamp[timestamp]
            if isinstance(response, Exception):
                failures.append(f"{timestamp.isoformat()}: {type(response).__name__}")
                samples.append(fixture_sample)
                continue
            activity_id, map_data, features = self._result(response)
            if activity_id:
                attempted_activity_ids.append(activity_id)
            try:
                aggregate = self._spatial_aggregate(
                    features, site, "average_temperature"
                )
            except FortyGuardError:
                failures.append(f"{timestamp.isoformat()}: no temperature tiles")
                samples.append(fixture_sample)
                continue
            if activity_id:
                activity_ids.append(activity_id)
            observed_hours += 1
            if heatmap is None:
                heatmap = map_data
            tile_counts.append(
                {
                    "site_tiles": aggregate.site_tile_count,
                    "control_tiles": aggregate.control_tile_count,
                }
            )
            samples.append(
                ThermalSample(
                    timestamp=timestamp,
                    site_temperature_c=round(aggregate.site_value, 3),
                    control_temperature_c=round(aggregate.control_value, 3),
                    wet_bulb_temperature_c=fixture_sample.wet_bulb_temperature_c,
                    relative_humidity_percent=fixture_sample.relative_humidity_percent,
                    solar_irradiance_w_m2=fixture_sample.solar_irradiance_w_m2,
                )
            )

        if observed_hours == hours:
            data_class = DataClass.OBSERVED
            source = self.source
        elif observed_hours:
            data_class = DataClass.INFERRED
            source = f"{self.source} + {fallback.source}"
        else:
            data_class = DataClass.SIMULATED
            source = fallback.source
        warnings = (
            [
                "Forecast humidity, wet-bulb temperature, and solar irradiance are simulated; "
                "site and control air temperatures are supplied by FortyGuard."
            ]
            if observed_hours
            else []
        )
        if failures:
            warnings.append(
                f"FortyGuard forecast coverage was available for {observed_hours}/{hours} hours; "
                "unavailable hours use simulated temperatures."
            )
        return ThermalDataset(
            samples=samples,
            data_class=data_class,
            source=source,
            activity_ids=activity_ids or fallback.activity_ids,
            metadata={
                "provider_mode": "hybrid",
                "temperature_fields_data_class": (
                    DataClass.OBSERVED.value
                    if observed_hours == hours
                    else data_class.value
                ),
                "environmental_fields_data_class": DataClass.SIMULATED.value,
                "observed_hours": observed_hours,
                "requested_hours": hours,
                "attempted_activity_ids": attempted_activity_ids,
                "tile_counts": tile_counts,
                "failed_hours": failures,
            },
            warnings=warnings,
            heatmap=heatmap,
        )

    async def annual_history(
        self,
        site: SiteInput,
        baseline_year: int,
        end_year: int,
        threshold_c: float,
        *,
        seed: int,
        controls: list[SiteInput] | None = None,
    ) -> AnnualThermalDataset:
        controls = controls or []
        years = list(range(baseline_year, end_year + 1))
        fallback = await self.fallback.annual_history(
            site, baseline_year, end_year, threshold_c, seed=seed, controls=controls
        )
        fallback_by_year = {summary.year: summary for summary in fallback.summaries}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def retrieve(payload: dict[str, Any]) -> Any:
            async with semaphore:
                return await self.client.create_heatmap(payload)

        async def retrieve_year(year: int) -> tuple[int, Any, Any, int]:
            days = calendar.monthrange(year, self.annual_reference_month)[1]
            start = datetime(year, self.annual_reference_month, 1, tzinfo=timezone.utc)
            end = datetime(year, self.annual_reference_month, days, tzinfo=timezone.utc)
            annual_aoi = (
                self._combined_aoi(site, controls) if controls else self._aoi(site)
            )
            temperature_payload = self._heatmap_payload(
                site,
                start=start,
                end=end,
                filter_type=4,
                analytic_type="tcm",
                polygon_aoi=annual_aoi,
            )
            eligibility_payload = self._heatmap_payload(
                site,
                start=start,
                end=end,
                filter_type=4,
                analytic_type="exceedance",
                threshold_c=threshold_c,
                polygon_aoi=annual_aoi,
            )
            temperature, eligibility = await asyncio.gather(
                retrieve(temperature_payload),
                retrieve(eligibility_payload),
                return_exceptions=True,
            )
            return year, temperature, eligibility, days

        results = await asyncio.gather(*(retrieve_year(year) for year in years))
        observed: dict[int, AnnualThermalSummary] = {}
        activity_ids: list[str] = []
        attempted_activity_ids: list[str] = []
        missing_years: list[int] = []
        failure_types: set[str] = set()
        tile_counts: dict[str, dict[str, Any]] = {}
        for year, temperature_response, eligibility_response, days in results:
            if isinstance(temperature_response, Exception) or isinstance(
                eligibility_response, Exception
            ):
                missing_years.append(year)
                if isinstance(temperature_response, Exception):
                    failure_types.add(type(temperature_response).__name__)
                if isinstance(eligibility_response, Exception):
                    failure_types.add(type(eligibility_response).__name__)
                continue
            temperature_id, _, temperature_features = self._result(temperature_response)
            eligibility_id, _, eligibility_features = self._result(eligibility_response)
            response_activity_ids = [
                activity_id
                for activity_id in (temperature_id, eligibility_id)
                if activity_id is not None
            ]
            attempted_activity_ids.extend(response_activity_ids)
            try:
                if controls:
                    temperatures = self._matched_spatial_aggregate(
                        temperature_features, site, controls, "average_temperature"
                    )
                    eligible = self._matched_spatial_aggregate(
                        eligibility_features, site, controls, "value"
                    )
                else:
                    temperatures = self._spatial_aggregate(
                        temperature_features, site, "average_temperature"
                    )
                    eligible = self._spatial_aggregate(
                        eligibility_features, site, "value"
                    )
            except FortyGuardError:
                missing_years.append(year)
                continue
            activity_ids.extend(response_activity_ids)
            window_hours = days * 24
            annualization = 8760 / window_hours
            observed[year] = AnnualThermalSummary(
                year=year,
                site_mean_temperature_c=round(temperatures.site_value, 3),
                control_mean_temperature_c=round(temperatures.control_value, 3),
                site_eligible_hours=max(
                    0, min(8760, round(eligible.site_value * annualization))
                ),
                control_eligible_hours=max(
                    0, min(8760, round(eligible.control_value * annualization))
                ),
            )
            tile_counts[str(year)] = {
                "site_tiles": temperatures.site_tile_count,
                "control_tiles": temperatures.control_tile_count,
                "control_zones": temperatures.control_zone_count,
                "control_tiles_by_zone": list(temperatures.control_tile_counts),
            }

        if not observed:
            warnings = [
                "FortyGuard returned no usable historical tiles for the requested years; "
                "thermal drift uses the simulated fixture series."
            ]
            if failure_types:
                warnings.append(
                    "Historical request failures: " + ", ".join(sorted(failure_types))
                )
            return AnnualThermalDataset(
                summaries=fallback.summaries,
                data_class=DataClass.SIMULATED,
                source=fallback.source,
                activity_ids=fallback.activity_ids,
                metadata={
                    "provider_mode": "hybrid_fallback",
                    "observed_years": [],
                    "backcast_years": years,
                    "attempted_activity_ids": attempted_activity_ids,
                    "control_method": (
                        "satellite_land_cover_matched_regional"
                        if controls
                        else "local_outer_ring"
                    ),
                    "control_sites": [control.model_dump() for control in controls],
                },
                warnings=warnings,
            )

        first_observed_year = min(observed)
        observed_anchor = observed[first_observed_year]
        fixture_anchor = fallback_by_year[first_observed_year]
        site_offset = (
            observed_anchor.site_mean_temperature_c
            - fixture_anchor.site_mean_temperature_c
        )
        control_offset = (
            observed_anchor.control_mean_temperature_c
            - fixture_anchor.control_mean_temperature_c
        )
        site_hours_offset = (
            observed_anchor.site_eligible_hours - fixture_anchor.site_eligible_hours
        )
        control_hours_offset = (
            observed_anchor.control_eligible_hours
            - fixture_anchor.control_eligible_hours
        )
        summaries: list[AnnualThermalSummary] = []
        for year in years:
            if year in observed:
                summaries.append(observed[year])
                continue
            fixture = fallback_by_year[year]
            summaries.append(
                AnnualThermalSummary(
                    year=year,
                    site_mean_temperature_c=round(
                        fixture.site_mean_temperature_c + site_offset, 3
                    ),
                    control_mean_temperature_c=round(
                        fixture.control_mean_temperature_c + control_offset, 3
                    ),
                    site_eligible_hours=max(
                        0, min(8760, fixture.site_eligible_hours + site_hours_offset)
                    ),
                    control_eligible_hours=max(
                        0,
                        min(
                            8760, fixture.control_eligible_hours + control_hours_offset
                        ),
                    ),
                )
            )
        observed_years = sorted(observed)
        missing_years = sorted(set(years) - set(observed_years))
        month_name = calendar.month_name[self.annual_reference_month]
        warnings = [
            f"Historical screening uses {month_name} heatmaps and annualized below-threshold "
            "hours; it is not a full-year utility-grade weather reconstruction."
        ]
        if controls:
            warnings.append(
                "Historical regional controls were selected using current satellite land-cover "
                "similarity; their historical land-cover stability is not yet verified."
            )
        if missing_years:
            warnings.append(
                "FortyGuard returned no usable tiles for "
                + ", ".join(str(year) for year in missing_years)
                + "; those years are calibrated simulated backcasts."
            )
        if failure_types:
            warnings.append(
                "Historical request failures: " + ", ".join(sorted(failure_types))
            )
        return AnnualThermalDataset(
            summaries=summaries,
            data_class=DataClass.INFERRED,
            source=f"{self.source} + {fallback.source}"
            if missing_years
            else self.source,
            activity_ids=activity_ids,
            metadata={
                "provider_mode": "hybrid",
                "analysis_window": month_name,
                "observed_years": observed_years,
                "backcast_years": missing_years,
                "attempted_activity_ids": attempted_activity_ids,
                "eligibility_annualized": True,
                "tile_counts": tile_counts,
                "control_method": (
                    "satellite_land_cover_matched_regional"
                    if controls
                    else "local_outer_ring"
                ),
                "control_sites": [control.model_dump() for control in controls],
            },
            warnings=warnings,
        )
