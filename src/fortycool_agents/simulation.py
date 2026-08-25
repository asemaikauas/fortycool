from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import FacilityProfile
from .providers.fixture import ThermalDataset


@dataclass(frozen=True)
class FacilityParameters:
    it_capacity_mw: float
    typical_it_load_mw: float
    pue: float


def resolve_facility_parameters(facility: FacilityProfile) -> FacilityParameters:
    capacity = facility.it_capacity_mw or 24.0
    typical_load = facility.typical_it_load_mw or capacity * 0.72
    pue = facility.reported_pue or 1.38
    return FacilityParameters(
        it_capacity_mw=capacity,
        typical_it_load_mw=min(typical_load, capacity),
        pue=pue,
    )


def simulate_bms(
    thermal: ThermalDataset,
    facility: FacilityProfile,
    *,
    seed: int,
) -> pd.DataFrame:
    """Create a coherent synthetic BMS stream driven by a thermal dataset."""
    if not thermal.samples:
        raise ValueError("thermal dataset is empty")

    params = resolve_facility_parameters(facility)
    rng = np.random.default_rng(seed)
    rows = [
        {
            "timestamp": sample.timestamp,
            "outdoor_temperature_c": sample.site_temperature_c,
            "control_temperature_c": sample.control_temperature_c,
            "wet_bulb_temperature_c": sample.wet_bulb_temperature_c,
            "outdoor_humidity_percent": sample.relative_humidity_percent,
            "solar_irradiance_w_m2": sample.solar_irradiance_w_m2,
        }
        for sample in thermal.samples
    ]
    frame = pd.DataFrame(rows)
    count = len(frame)
    hours = np.arange(count, dtype=float)
    hour_of_day = frame["timestamp"].map(lambda value: value.hour).to_numpy(dtype=float)

    capacity_kw = params.it_capacity_mw * 1000
    typical_kw = params.typical_it_load_mw * 1000
    daily = 0.045 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    weekly = 0.025 * np.sin(2 * np.pi * hours / (24 * 7))
    burst = rng.normal(0, 0.018, count)
    it_load = np.clip(typical_kw * (1 + daily + weekly + burst), capacity_kw * 0.35, capacity_kw)

    outdoor = frame["outdoor_temperature_c"].to_numpy(dtype=float)
    wet_bulb = frame["wet_bulb_temperature_c"].to_numpy(dtype=float)
    humidity = frame["outdoor_humidity_percent"].to_numpy(dtype=float)

    supply_setpoint = 19.0 + 0.35 * np.sin(2 * np.pi * hours / (24 * 5))
    supply_setpoint += rng.choice([-0.25, 0.0, 0.25], size=count, p=[0.15, 0.7, 0.15])
    chilled_water = 7.0 + rng.choice([-0.25, 0.0, 0.25], size=count, p=[0.1, 0.8, 0.1])
    economizer = ((outdoor < 18.0) & (humidity < 75.0)).astype(int)
    fan_speed = np.clip(
        63
        + 19 * (it_load / capacity_kw)
        + 0.55 * np.maximum(outdoor - 25, 0)
        - economizer * 3
        + rng.normal(0, 1.1, count),
        45,
        100,
    )

    base_cooling = it_load * max(params.pue - 1.0, 0.12) * 0.72
    dry_bulb_penalty = it_load * 0.0038 * np.maximum(outdoor - 18, 0)
    wet_bulb_penalty = it_load * 0.0018 * np.maximum(wet_bulb - 12, 0)
    solar_penalty = frame["solar_irradiance_w_m2"].to_numpy(dtype=float) * 0.18
    setpoint_effect = -it_load * 0.011 * (supply_setpoint - 18.5)
    chilled_water_effect = -it_load * 0.006 * (chilled_water - 6.5)
    economizer_saving = -it_load * 0.035 * economizer
    fan_power = 280 * np.power(fan_speed / 100, 3)
    noise = rng.normal(0, np.maximum(35, base_cooling * 0.018), count)
    cooling_power = np.maximum(
        100,
        base_cooling
        + dry_bulb_penalty
        + wet_bulb_penalty
        + solar_penalty
        + setpoint_effect
        + chilled_water_effect
        + economizer_saving
        + fan_power
        + noise,
    )

    load_ratio = it_load / capacity_kw
    server_inlet = (
        supply_setpoint
        + 3.15
        + 1.45 * load_ratio
        + 0.035 * np.maximum(outdoor - 25, 0)
        + 0.055 * np.maximum(70 - fan_speed, 0)
        + rng.normal(0, 0.16, count)
    )
    return_air = server_inlet + 8.2 + 2.0 * load_ratio + rng.normal(0, 0.25, count)
    auxiliary_power = it_load * max(params.pue - 1.0, 0.12) * 0.28
    total_power = it_load + cooling_power + auxiliary_power

    frame["it_load_kw"] = np.round(it_load, 3)
    frame["cooling_power_kw"] = np.round(cooling_power, 3)
    frame["total_facility_power_kw"] = np.round(total_power, 3)
    frame["supply_air_setpoint_c"] = np.round(supply_setpoint, 3)
    frame["chilled_water_supply_c"] = np.round(chilled_water, 3)
    frame["fan_speed_percent"] = np.round(fan_speed, 3)
    frame["economizer_state"] = economizer
    frame["server_inlet_temperature_c"] = np.round(server_inlet, 3)
    frame["return_air_temperature_c"] = np.round(return_air, 3)
    frame["indoor_humidity_percent"] = np.round(np.clip(humidity * 0.45 + 20, 30, 70), 3)
    return frame


def make_forecast_operating_frame(
    thermal: ThermalDataset,
    facility: FacilityProfile,
    history: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Create the no-action operating plan used as the 12-hour baseline."""
    simulated = simulate_bms(thermal, facility, seed=seed)
    tail = history.tail(min(24, len(history)))
    for column in (
        "supply_air_setpoint_c",
        "chilled_water_supply_c",
        "fan_speed_percent",
    ):
        simulated[column] = float(tail[column].median())
    # Forecast load is a projection, not hidden future BMS truth.
    return simulated


def enrich_uploaded_bms(
    uploaded: pd.DataFrame,
    thermal: ThermalDataset,
    facility: FacilityProfile,
) -> tuple[pd.DataFrame, list[str]]:
    """Align uploaded BMS telemetry with thermal data and fill optional controls.

    Uploaded series are normalized to hourly cadence so model validation and the
    forecast horizon retain consistent semantics.
    """
    warnings: list[str] = []
    frame = uploaded.copy(deep=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = (
        frame.set_index("timestamp")
        .resample("1h")
        .mean(numeric_only=True)
        .interpolate(limit=2)
        .dropna(subset=["it_load_kw", "cooling_power_kw", "server_inlet_temperature_c"])
        .reset_index()
    )

    weather = pd.DataFrame(
        [
            {
                "timestamp": sample.timestamp,
                "outdoor_temperature_c": sample.site_temperature_c,
                "control_temperature_c": sample.control_temperature_c,
                "wet_bulb_temperature_c": sample.wet_bulb_temperature_c,
                "outdoor_humidity_percent": sample.relative_humidity_percent,
                "solar_irradiance_w_m2": sample.solar_irradiance_w_m2,
            }
            for sample in thermal.samples
        ]
    )
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    frame = frame.merge(weather, on="timestamp", how="left", validate="one_to_one")
    weather_columns = [
        "outdoor_temperature_c",
        "control_temperature_c",
        "wet_bulb_temperature_c",
        "outdoor_humidity_percent",
        "solar_irradiance_w_m2",
    ]
    if frame[weather_columns].isna().any().any():
        raise ValueError("thermal provider did not cover the uploaded telemetry interval")

    params = resolve_facility_parameters(facility)
    defaults: dict[str, float] = {
        "supply_air_setpoint_c": 19.0,
        "chilled_water_supply_c": 7.0,
        "fan_speed_percent": 72.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
            warnings.append(f"{column} was missing and filled with the digital-twin default {value}")
    if "economizer_state" not in frame.columns:
        frame["economizer_state"] = (
            (frame["outdoor_temperature_c"] < 18)
            & (frame["outdoor_humidity_percent"] < 75)
        ).astype(float)
        warnings.append("economizer_state was inferred from outdoor conditions")
    if "total_facility_power_kw" not in frame.columns:
        auxiliary = frame["it_load_kw"] * max(params.pue - 1.0, 0.12) * 0.28
        frame["total_facility_power_kw"] = (
            frame["it_load_kw"] + frame["cooling_power_kw"] + auxiliary
        )
    return frame, warnings
