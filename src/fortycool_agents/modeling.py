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


# A feature whose training standard deviation falls below this carries no
# information about how the plant responds, so any coefficient fitted to it is
# noise and must never authorise a control change.
ZERO_VARIANCE_TOLERANCE = 1e-6


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None
    zero_variance_features_: tuple[str, ...] = ()
    training_minimum_: np.ndarray | None = None
    training_maximum_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: str) -> "RidgeRegressor":
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        # Record which features had no variation before neutralising their
        # scale. Neutralising alone silently turns an out-of-distribution input
        # into a plausible-looking standardized value multiplied by a
        # coefficient that was fitted on nothing.
        self.zero_variance_features_ = tuple(
            column
            for column, deviation in zip(FEATURE_COLUMNS, self.scale_)
            if deviation < ZERO_VARIANCE_TOLERANCE
        )
        self.scale_ = np.where(
            self.scale_ < ZERO_VARIANCE_TOLERANCE, 1.0, self.scale_
        )
        self.training_minimum_ = x.min(axis=0)
        self.training_maximum_ = x.max(axis=0)
        standardized = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coefficients_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def outside_training_envelope(self, frame: pd.DataFrame) -> dict[str, float]:
        """Report how far each feature falls outside the fitted range.

        A linear model extrapolated far past the data it was fitted on is not a
        prediction. The recommended chilled-water setpoint sat six standard
        deviations beyond the training maximum, and nothing in the pipeline
        could see it.
        """

        if self.training_minimum_ is None or self.training_maximum_ is None:
            raise RuntimeError("model must be fitted before envelope checks")
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        breaches: dict[str, float] = {}
        for index, column in enumerate(FEATURE_COLUMNS):
            column_values = values[:, index]
            low = float(self.training_minimum_[index])
            high = float(self.training_maximum_[index])
            below = float(np.min(column_values)) - low
            above = float(np.max(column_values)) - high
            if above > 0:
                breaches[column] = round(above, 6)
            elif below < 0:
                breaches[column] = round(below, 6)
        return breaches

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
    # One-sided upper bound on inlet prediction error, taken across every
    # cross-validation fold. A safety limit needs an upper quantile; a mean
    # absolute error is roughly 2.4 times too small for the job and understated
    # the true inlet temperature at the exact setpoint being recommended.
    inlet_safety_buffer_c: float = 0.0
    cross_validation_folds: int = 0
    inlet_fold_mae_c: tuple[float, ...] = ()
    cooling_fold_mae_kw: tuple[float, ...] = ()
    zero_variance_features: tuple[str, ...] = ()


MINIMUM_TRAINING_HOURS = 14 * 24
CROSS_VALIDATION_FOLDS = 5
SAFETY_QUANTILE = 0.95


def _rolling_origin_errors(
    history: pd.DataFrame, block_size: int, folds: int
) -> tuple[list[float], list[float], np.ndarray]:
    """Blocked rolling-origin cross-validation over the tail of the series.

    A single contiguous final day is 1.7% of a 60-day history, one season and
    one weather regime, with no repeats. Every fold below is a genuine
    out-of-sample block, and the spread across folds is what the safety buffer
    is built from.
    """

    cooling_fold_mae: list[float] = []
    inlet_fold_mae: list[float] = []
    inlet_absolute_errors: list[np.ndarray] = []
    for fold in range(folds, 0, -1):
        split = len(history) - fold * block_size
        if split < MINIMUM_TRAINING_HOURS:
            continue
        training = history.iloc[:split]
        validation = history.iloc[split : split + block_size]
        if validation.empty:
            continue
        cooling = RidgeRegressor(alpha=3.0).fit(training, "cooling_power_kw")
        inlet = RidgeRegressor(alpha=3.0).fit(training, "server_inlet_temperature_c")
        cooling_error = np.abs(
            cooling.predict(validation) - validation["cooling_power_kw"].to_numpy()
        )
        inlet_error = np.abs(
            inlet.predict(validation)
            - validation["server_inlet_temperature_c"].to_numpy()
        )
        cooling_fold_mae.append(float(np.mean(cooling_error)))
        inlet_fold_mae.append(float(np.mean(inlet_error)))
        inlet_absolute_errors.append(inlet_error)
    pooled = (
        np.concatenate(inlet_absolute_errors)
        if inlet_absolute_errors
        else np.array([], dtype=float)
    )
    return cooling_fold_mae, inlet_fold_mae, pooled


def train_and_backtest(history: pd.DataFrame) -> ModelBundle:
    if len(history) < MINIMUM_TRAINING_HOURS:
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

    cooling_fold_mae, inlet_fold_mae, pooled_inlet_errors = _rolling_origin_errors(
        history, holdout_size, CROSS_VALIDATION_FOLDS
    )
    holdout_inlet_errors = np.abs(
        inlet_prediction - holdout["server_inlet_temperature_c"].to_numpy()
    )
    error_population = (
        np.concatenate([pooled_inlet_errors, holdout_inlet_errors])
        if pooled_inlet_errors.size
        else holdout_inlet_errors
    )
    # One-sided upper quantile of the absolute error, not its mean, and never
    # smaller than the mean it replaces.
    safety_buffer = float(
        max(np.quantile(error_population, SAFETY_QUANTILE), inlet_mae)
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
        inlet_safety_buffer_c=round(safety_buffer, 4),
        cross_validation_folds=len(inlet_fold_mae),
        inlet_fold_mae_c=tuple(round(value, 4) for value in inlet_fold_mae),
        cooling_fold_mae_kw=tuple(round(value, 3) for value in cooling_fold_mae),
        zero_variance_features=tuple(
            sorted(
                set(cooling_model.zero_variance_features_)
                | set(inlet_model.zero_variance_features_)
            )
        ),
    )
