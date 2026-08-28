"""Small-sample uncertainty helpers for the difference-in-differences estimate.

The annual thermal series is short (typically five July observations), so every
headline drift number carries real sampling error. This module keeps that error
explicit: it fits the site-minus-control gap against time by ordinary least
squares, derives a residual standard error on ``n - 2`` degrees of freedom, and
returns a two-sided confidence interval for the projected drift.

The interval is a *between-year* interval. It answers "how well is the trend
pinned down by these annual points", not "how much spatial variation is there
inside one year" - the providers return per-year spatial means, so within-year
dispersion is not recoverable downstream. ``DriftEstimate.basis`` records that
limitation so it can be surfaced next to the number instead of implied away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
# A tiny table avoids adding scipy for one lookup; beyond the table the normal
# approximation is accurate to better than 1%.
_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_NORMAL_CRITICAL_95 = 1.96


def t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% critical value for ``degrees_of_freedom``."""

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be at least 1")
    return _T_CRITICAL_95.get(degrees_of_freedom, _NORMAL_CRITICAL_95)


def fisher_interval_95(correlation: float, observations: int) -> tuple[float, float]:
    """Two-sided 95% interval for a Pearson correlation via Fisher's z.

    Reported alongside every correlation so the reader sees how little a short
    series pins down: at n=5 the interval covers roughly half the unit range.
    """

    if observations < 4:
        return (-1.0, 1.0)
    bounded = float(np.clip(correlation, -0.999999, 0.999999))
    z = np.arctanh(bounded)
    standard_error = 1.0 / np.sqrt(observations - 3)
    half_width = _NORMAL_CRITICAL_95 * standard_error
    return (
        float(np.tanh(z - half_width)),
        float(np.tanh(z + half_width)),
    )


@dataclass(frozen=True)
class DriftEstimate:
    """A projected drift with the uncertainty that comes with it."""

    slope_c_per_year: float
    drift_c: float
    span_years: float
    observations: int
    standard_error_c: float | None
    ci_low_c: float | None
    ci_high_c: float | None
    basis: str

    @property
    def identified(self) -> bool:
        """True when the series supports a variance estimate at all."""

        return self.standard_error_c is not None

    @property
    def distinguishable_from_zero(self) -> bool:
        """True when the 95% interval excludes zero."""

        if self.ci_low_c is None or self.ci_high_c is None:
            return False
        return self.ci_low_c > 0 or self.ci_high_c < 0

    def scale(self, factor: float) -> tuple[float, float | None, float | None]:
        """Propagate the point estimate and interval through a linear factor."""

        point = self.drift_c * factor
        if self.ci_low_c is None or self.ci_high_c is None:
            return point, None, None
        bounds = sorted((self.ci_low_c * factor, self.ci_high_c * factor))
        return point, bounds[0], bounds[1]


def estimate_drift(years: np.ndarray, gaps: np.ndarray) -> DriftEstimate:
    """Fit the site-minus-control gap against time and bound the projection.

    ``years`` and ``gaps`` must be the same length and ordered by year. The
    projected drift is ``slope * (last_year - first_year)`` - the ordinary
    least-squares projection rather than the two-point endpoint difference,
    which discards the interior observations and carries a wider variance.
    """

    years = np.asarray(years, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    if years.shape != gaps.shape:
        raise ValueError("years and gaps must have the same shape")
    observations = int(years.size)
    if observations < 2:
        raise ValueError("at least two annual observations are required")

    centered = years - years[0]
    span = float(centered[-1])
    design = np.column_stack([np.ones(observations), centered])
    coefficients, *_ = np.linalg.lstsq(design, gaps, rcond=None)
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    drift = slope * span

    degrees_of_freedom = observations - 2
    sum_squares_x = float(np.sum((centered - centered.mean()) ** 2))
    if degrees_of_freedom < 1 or sum_squares_x <= 0.0 or span == 0.0:
        return DriftEstimate(
            slope_c_per_year=slope,
            drift_c=drift,
            span_years=span,
            observations=observations,
            standard_error_c=None,
            ci_low_c=None,
            ci_high_c=None,
            basis=(
                "Too few annual observations to estimate sampling error; "
                "the drift is reported without an interval"
            ),
        )

    residuals = gaps - (intercept + slope * centered)
    residual_variance = float(np.sum(residuals**2)) / degrees_of_freedom
    slope_standard_error = float(np.sqrt(residual_variance / sum_squares_x))
    drift_standard_error = slope_standard_error * abs(span)
    critical = t_critical_95(degrees_of_freedom)
    half_width = critical * drift_standard_error
    return DriftEstimate(
        slope_c_per_year=slope,
        drift_c=drift,
        span_years=span,
        observations=observations,
        standard_error_c=drift_standard_error,
        ci_low_c=drift - half_width,
        ci_high_c=drift + half_width,
        basis=(
            f"Ordinary least squares on {observations} annual site-minus-control gaps; "
            f"95% interval from the residual standard error on {degrees_of_freedom} "
            "degrees of freedom. Between-year sampling error only: the providers return "
            "one spatial mean per year, so within-year spatial dispersion is not included."
        ),
    )
