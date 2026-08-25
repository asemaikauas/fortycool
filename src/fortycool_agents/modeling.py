from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "it_load_kw",
    "outdoor_temperature_c",
    "wet_bulb_temperature_c",
    "outdoor_humidity_percent",
    "solar_irradiance_w_m2",
    "supply_air_setpoint_c",
    "chilled_water_supply_c",
    "fan_speed_percent",
    "economizer_state",
]


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: str) -> "RidgeRegressor":
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        standardized = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coefficients_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise RuntimeError("model must be fitted before prediction")
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        standardized = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        return design @ self.coefficients_


@dataclass(frozen=True)
class ModelBundle:
    cooling_model: RidgeRegressor
    inlet_model: RidgeRegressor
    cooling_mae_kw: float
    inlet_mae_c: float
    confidence: float
    action_response_identifiable: bool
    backtest: pd.DataFrame


def train_and_backtest(history: pd.DataFrame) -> ModelBundle:
    if len(history) < 14 * 24:
        raise ValueError("at least 14 days of hourly data are required")
    holdout_size = min(24, max(1, len(history) // 10))
    training = history.iloc[:-holdout_size].copy()
    holdout = history.iloc[-holdout_size:].copy()

    cooling_model = RidgeRegressor(alpha=3.0).fit(training, "cooling_power_kw")
    inlet_model = RidgeRegressor(alpha=3.0).fit(training, "server_inlet_temperature_c")
    cooling_prediction = cooling_model.predict(holdout)
    inlet_prediction = inlet_model.predict(holdout)
    cooling_mae = float(np.mean(np.abs(cooling_prediction - holdout["cooling_power_kw"])))
    inlet_mae = float(
        np.mean(np.abs(inlet_prediction - holdout["server_inlet_temperature_c"]))
    )
    normalized_cooling_error = cooling_mae / max(float(holdout["cooling_power_kw"].mean()), 1.0)
    confidence = float(np.clip(1 - normalized_cooling_error * 4 - inlet_mae / 8, 0.45, 0.96))
    action_response_identifiable = bool(
        training["supply_air_setpoint_c"].std() >= 0.1
        or training["chilled_water_supply_c"].std() >= 0.1
        or training["fan_speed_percent"].std() >= 1.0
    )

    backtest = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].tolist(),
            "actual_cooling_kw": holdout["cooling_power_kw"].round(3).tolist(),
            "predicted_cooling_kw": np.round(cooling_prediction, 3).tolist(),
            "actual_inlet_c": holdout["server_inlet_temperature_c"].round(3).tolist(),
            "predicted_inlet_c": np.round(inlet_prediction, 3).tolist(),
        }
    )
    return ModelBundle(
        cooling_model=cooling_model,
        inlet_model=inlet_model,
        cooling_mae_kw=round(cooling_mae, 3),
        inlet_mae_c=round(inlet_mae, 3),
        confidence=round(confidence, 4),
        action_response_identifiable=action_response_identifiable,
        backtest=backtest,
    )
