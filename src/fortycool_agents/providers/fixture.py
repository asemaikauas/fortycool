from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from ..models import DataClass, SiteInput


@dataclass(frozen=True)
class ThermalSample:
    timestamp: datetime
    site_temperature_c: float
    control_temperature_c: float
    wet_bulb_temperature_c: float
    relative_humidity_percent: float
    solar_irradiance_w_m2: float


@dataclass(frozen=True)
class AnnualThermalSummary:
    year: int
    site_mean_temperature_c: float
    control_mean_temperature_c: float
    site_eligible_hours: int
    control_eligible_hours: int


@dataclass(frozen=True)
class ThermalDataset:
    samples: list[ThermalSample]
    data_class: DataClass
    source: str
    activity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    heatmap: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnnualThermalDataset:
    summaries: list[AnnualThermalSummary]
    data_class: DataClass
    source: str
    activity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ThermalDataProvider(Protocol):
    async def history(
        self, site: SiteInput, start: datetime, end: datetime, *, seed: int
    ) -> ThermalDataset: ...

    async def forecast(
        self, site: SiteInput, hours: int, *, seed: int
    ) -> ThermalDataset: ...

    async def annual_history(
        self,
        site: SiteInput,
        baseline_year: int,
        end_year: int,
        threshold_c: float,
        *,
        seed: int,
        controls: list[SiteInput] | None = None,
    ) -> AnnualThermalDataset: ...


class FixtureThermalProvider:
    """Deterministic synthetic thermal data with FortyGuard-shaped provenance.

    This provider never labels its output as observed. It exists so the complete
    agent workflow can be developed and demonstrated without spending API credits.
    """

    source = "fixture://fortyguard-shaped-thermal-series/v1"

    @staticmethod
    def _temperature(
        site: SiteInput, timestamp: datetime, rng: random.Random
    ) -> ThermalSample:
        day = timestamp.timetuple().tm_yday
        hour = timestamp.hour + timestamp.minute / 60
        latitude_adjustment = -0.38 * (site.latitude - 32.0)
        seasonal = 11.5 * math.sin(2 * math.pi * (day - 172) / 365.25)
        daily = 5.2 * math.sin(2 * math.pi * (hour - 14) / 24)
        regional = 17.5 + latitude_adjustment + seasonal + daily

        years_since_2021 = max(0.0, (timestamp.year - 2021) + (day - 1) / 365.25)
        control_drift = 0.035 * years_since_2021
        local_excess_drift = 0.105 * years_since_2021
        weather_noise = rng.gauss(0, 0.55)
        control = regional + control_drift + weather_noise
        site_temp = control + 0.65 + local_excess_drift + rng.gauss(0, 0.18)

        humidity = min(
            96.0, max(22.0, 63.0 - 1.25 * (site_temp - 20) + rng.gauss(0, 4))
        )
        wet_bulb = site_temp - max(1.0, (100.0 - humidity) / 7.2)
        daylight = max(0.0, math.sin(math.pi * (hour - 6) / 14))
        solar = daylight * 820.0 * max(0.35, 1 - humidity / 180)
        return ThermalSample(
            timestamp=timestamp,
            site_temperature_c=round(site_temp, 3),
            control_temperature_c=round(control, 3),
            wet_bulb_temperature_c=round(wet_bulb, 3),
            relative_humidity_percent=round(humidity, 2),
            solar_irradiance_w_m2=round(solar, 2),
        )

    async def history(
        self, site: SiteInput, start: datetime, end: datetime, *, seed: int
    ) -> ThermalDataset:
        rng = random.Random(seed)
        cursor = start.astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        stop = end.astimezone(timezone.utc)
        samples: list[ThermalSample] = []
        while cursor < stop:
            samples.append(self._temperature(site, cursor, rng))
            cursor += timedelta(hours=1)
        return ThermalDataset(
            samples=samples,
            data_class=DataClass.SIMULATED,
            source=self.source,
            activity_ids=[f"fixture-history-{seed}"],
        )

    async def forecast(
        self, site: SiteInput, hours: int, *, seed: int
    ) -> ThermalDataset:
        # A fixed reference instant keeps fixture-mode snapshots reproducible.
        start = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
        dataset = await self.history(
            site, start, start + timedelta(hours=hours), seed=seed + 1000
        )
        return ThermalDataset(
            samples=dataset.samples,
            data_class=dataset.data_class,
            source=self.source,
            activity_ids=[f"fixture-forecast-{seed}"],
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
        rng = random.Random(seed + 2000)
        summaries: list[AnnualThermalSummary] = []
        for year in range(baseline_year, end_year + 1):
            years = year - baseline_year
            regional_trend = 0.035 * years
            local_excess = 0.105 * years
            control_mean = 17.8 + regional_trend + rng.gauss(0, 0.05)
            site_mean = control_mean + 0.65 + local_excess + rng.gauss(0, 0.025)

            # Eligibility here means temperature-only hours below the threshold.
            threshold_offset = (threshold_c - 18.0) * 105
            control_hours = round(
                4520 + threshold_offset - regional_trend * 230 + rng.gauss(0, 18)
            )
            site_hours = round(
                control_hours - 155 - local_excess * 245 + rng.gauss(0, 10)
            )
            summaries.append(
                AnnualThermalSummary(
                    year=year,
                    site_mean_temperature_c=round(site_mean, 3),
                    control_mean_temperature_c=round(control_mean, 3),
                    site_eligible_hours=max(0, min(8760, site_hours)),
                    control_eligible_hours=max(0, min(8760, control_hours)),
                )
            )
        return AnnualThermalDataset(
            summaries=summaries,
            data_class=DataClass.SIMULATED,
            source=self.source,
            activity_ids=[f"fixture-annual-{seed}"],
            metadata={
                "control_method": (
                    "satellite_land_cover_matched_regional"
                    if controls
                    else "fixture_regional_control"
                ),
                "control_sites": [control.model_dump() for control in controls],
            },
        )
