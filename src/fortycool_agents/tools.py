from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean
from typing import Any

import numpy as np
import pandas as pd

from .context import RunContext
from .inference import DriftEstimate, estimate_drift, fisher_interval_95
from .modeling import SAFETY_QUANTILE, ModelBundle
from .models import (
    Assumption,
    Chart,
    DataClass,
    EconomicsInput,
    EvidenceGrade,
    EvidenceRef,
    Metric,
    Recommendation,
    SafetyConstraints,
    SafetyVerdict,
    WarningCode,
)
from .providers.fixture import AnnualThermalDataset
from .providers.urban import (
    UrbanContextDataset,
    canonical_land_cover,
    tree_canopy_share,
)
from .simulation import resolve_facility_parameters


# Below this many paired observed years a Pearson correlation carries no
# information: at n=3 the 5% critical value is 0.997 and two unrelated series
# clear |r| > 0.8 four times in ten.
MINIMUM_ATTRIBUTION_YEARS = 8

# Cooling-energy response to one degree of sustained local warming, as a
# fraction of cooling-plant load per degree. This is the fallback when the
# operator supplies none, and it is recorded as an assumption every time it is
# used rather than published as a measured facility property.
# Expressed per unit of COOLING-PLANT load, not per unit of IT load. The
# simulator encodes 0.0038 per unit of IT load; the cooling plant carries
# max(PUE-1, 0.12) * 0.72 of that load, so at the demo PUE of 1.38 this default
# reproduces the simulator's coefficient exactly while letting a real PUE move
# the answer. It is an assumption on every run, never a measured property.
DEFAULT_TEMPERATURE_SENSITIVITY_PER_C = 0.0038 / (0.38 * 0.72)

# Share of the (PUE - 1) auxiliary load attributed to the cooling plant, matching
# the split the digital-twin simulator uses.
COOLING_LOAD_FACTOR = 0.72
HOURS_PER_YEAR = 8760


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (datetime, pd.Timestamp)):
                record[key] = value.isoformat()
            elif isinstance(value, np.generic):
                record[key] = value.item()
    return records


def apply_demo_defaults(context: RunContext) -> SafetyConstraints:
    provided = context.request.constraints
    values = provided.model_dump()
    defaults: dict[str, float | int] = {
        "maximum_server_inlet_temperature_c": 27.0,
        "minimum_safety_margin_c": 2.0,
        "supply_air_setpoint_min_c": 18.0,
        "supply_air_setpoint_max_c": 22.0,
        "maximum_setpoint_change_c": 1.0,
        "minimum_model_confidence": 0.70,
        "forecast_max_age_minutes": 180,
    }
    if context.request.use_demo_defaults:
        for field, default in defaults.items():
            if values[field] is None:
                values[field] = default
                context.assumptions.append(
                    Assumption(
                        field=f"constraints.{field}",
                        value=default,
                        reason="Illustrative digital-twin constraint; not approved for a real facility",
                    )
                )
    return SafetyConstraints(**values)


def register_facility_assumptions(context: RunContext) -> None:
    facility = context.request.facility
    if facility.it_capacity_mw is None:
        context.assumptions.append(
            Assumption(
                field="facility.it_capacity_mw",
                value=24.0,
                reason="Representative colocation digital-twin capacity",
            )
        )
    if facility.typical_it_load_mw is None:
        capacity = facility.it_capacity_mw or 24.0
        context.assumptions.append(
            Assumption(
                field="facility.typical_it_load_mw",
                value=round(capacity * 0.72, 3),
                reason="Digital-twin base load at 72% of configured IT capacity",
            )
        )
    if facility.reported_pue is None:
        context.assumptions.append(
            Assumption(
                field="facility.reported_pue",
                value=1.38,
                reason="Illustrative digital-twin PUE; not a claim about the mapped facility",
            )
        )


def analyze_urban_context(context: RunContext, urban: UrbanContextDataset) -> None:
    latest = urban.latest_snapshot
    latest_groups = canonical_land_cover(latest.segments)
    baseline = urban.baseline_snapshot
    historical_change_available = bool(
        urban.metadata.get("historical_change_available") and baseline is not None
    )
    control_quality_passed = bool(urban.metadata.get("control_quality_passed"))
    historical_control_stability_verified = bool(
        urban.metadata.get("historical_control_stability_verified")
    )
    context.artifacts["historical_control_stability_verified"] = (
        historical_control_stability_verified
    )
    land_cover_provider = str(
        urban.metadata.get("land_cover_provider", "fortyguard_satellite_segmentation")
    )
    annual_land_cover_series = list(urban.metadata.get("annual_land_cover_series", []))
    satellite_evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"satellite-land-cover-{context.run_id}",
            source=urban.source,
            description="Satellite land-cover history for the site and control candidates",
            data_class=urban.data_class,
            activity_id=urban.activity_ids[0] if urban.activity_ids else None,
            endpoint=urban.metadata.get("endpoint")
            or ("/v1/satellite" if urban.data_class == DataClass.OBSERVED else None),
            metadata={
                "activity_ids": urban.activity_ids,
                "site_requested_dates": [
                    snapshot.requested_date
                    for snapshot in (baseline, latest)
                    if snapshot is not None
                ],
                "site_image_years": [
                    snapshot.image_year
                    for snapshot in (baseline, latest)
                    if snapshot is not None
                ],
                "latest_site_segments": latest.segments,
                **urban.metadata,
            },
        )
    )
    control_evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"control-matching-{context.run_id}",
            source="fortycool://analytics/land-cover-control-matching/v1",
            description="Regional control selection by satellite land-cover similarity",
            data_class=DataClass.INFERRED,
            metadata={
                "formula": "1 - L1(canonical_land_cover_site, candidate) / 200",
                "evaluated_controls": [
                    {
                        "name": control.site.name,
                        "latitude": control.site.latitude,
                        "longitude": control.site.longitude,
                        "similarity_score": control.similarity_score,
                        "distance_km": control.distance_km,
                        "activity_id": control.snapshot.activity_id,
                        "image_year": control.snapshot.image_year,
                    }
                    for control in urban.matched_controls
                ],
            },
        )
    )
    context.metrics.extend(
        [
            Metric(
                id="current_built_surface_share_percent",
                label="Current built-surface share",
                value=round(
                    latest_groups["building"] + latest_groups["transport_surface"], 3
                ),
                unit="%",
                confidence=0.9 if urban.data_class == DataClass.OBSERVED else 0.7,
                data_class=urban.data_class,
                evidence_grade=evidence_grade_for(urban.data_class),
                evidence_ids=[satellite_evidence_id],
                caveats=[
                    "Built share follows the source model's classification of developed surfaces"
                ],
            ),
            Metric(
                id="current_tree_canopy_share_percent",
                label="Current tree-canopy share",
                value=tree_canopy_share(latest.segments),
                unit="%",
                confidence=0.9 if urban.data_class == DataClass.OBSERVED else 0.7,
                data_class=urban.data_class,
                evidence_grade=evidence_grade_for(urban.data_class),
                evidence_ids=[satellite_evidence_id],
            ),
            Metric(
                id="regional_control_match_score",
                label="Mean regional-control land-cover match",
                value=round(
                    fmean(
                        control.similarity_score for control in urban.matched_controls
                    ),
                    3,
                ),
                unit="score",
                confidence=0.78 if urban.data_class == DataClass.OBSERVED else 0.65,
                data_class=DataClass.INFERRED,
                evidence_grade=evidence_grade_for(urban.data_class),
                evidence_ids=[satellite_evidence_id, control_evidence_id],
                caveats=(
                    [
                        "Historical coverage was verified for selected controls; classification uncertainty remains"
                    ]
                    if historical_control_stability_verified
                    else [
                        "Similarity is based on current imagery and does not establish historical stability"
                    ]
                ),
            ),
            Metric(
                id="regional_control_match_status",
                label="Regional-control quality gate",
                value="accepted" if control_quality_passed else "rejected",
                unit="status",
                confidence=1.0,
                data_class=DataClass.INFERRED,
                evidence_grade=EvidenceGrade.B,
                evidence_ids=[satellite_evidence_id, control_evidence_id],
                caveats=(
                    []
                    if control_quality_passed
                    else [
                        "Thermal drift uses the local outer ring because regional candidates failed the similarity threshold"
                    ]
                ),
            ),
        ]
    )
    if historical_change_available and baseline is not None:
        baseline_groups = canonical_land_cover(baseline.segments)
        context.metrics.extend(
            [
                Metric(
                    id="built_surface_change_percentage_points",
                    label="Built-surface change",
                    value=round(
                        (latest_groups["building"] + latest_groups["transport_surface"])
                        - (
                            baseline_groups["building"]
                            + baseline_groups["transport_surface"]
                        ),
                        3,
                    ),
                    unit="percentage points",
                    confidence=0.8 if urban.data_class == DataClass.OBSERVED else 0.65,
                    data_class=DataClass.INFERRED,
                    evidence_grade=evidence_grade_for(urban.data_class),
                    evidence_ids=[satellite_evidence_id],
                ),
                Metric(
                    id="tree_canopy_change_percentage_points",
                    label="Tree-canopy change",
                    value=round(
                        tree_canopy_share(latest.segments)
                        - tree_canopy_share(baseline.segments),
                        3,
                    ),
                    unit="percentage points",
                    confidence=0.8 if urban.data_class == DataClass.OBSERVED else 0.65,
                    data_class=DataClass.INFERRED,
                    evidence_grade=evidence_grade_for(urban.data_class),
                    evidence_ids=[satellite_evidence_id],
                ),
            ]
        )
    else:
        context.metrics.append(
            Metric(
                id="satellite_change_status",
                label="Historical satellite change status",
                value="not_available",
                unit="status",
                confidence=1.0,
                data_class=urban.data_class,
                evidence_grade=EvidenceGrade.C,
                evidence_ids=[satellite_evidence_id],
                caveats=[
                    "Baseline and latest requests did not resolve to distinct usable image years; no change was calculated"
                ],
            )
        )

    if len(annual_land_cover_series) >= 2:
        first_land_cover = annual_land_cover_series[0]
        latest_land_cover = annual_land_cover_series[-1]
        context.metrics.extend(
            [
                Metric(
                    id="local_excess_built_surface_change_percentage_points",
                    label="Local excess built-surface change versus controls",
                    value=round(
                        float(latest_land_cover["local_excess_built_surface_percent"])
                        - float(first_land_cover["local_excess_built_surface_percent"]),
                        3,
                    ),
                    unit="percentage points",
                    confidence=(
                        0.76 if urban.data_class == DataClass.OBSERVED else 0.62
                    ),
                    data_class=DataClass.INFERRED,
                    evidence_grade=evidence_grade_for(urban.data_class),
                    evidence_ids=[satellite_evidence_id, control_evidence_id],
                    caveats=[
                        "Dynamic World is a modeled classification; this comparison does not establish causality"
                    ],
                ),
                Metric(
                    id="land_cover_history_coverage_years",
                    label="Land-cover history coverage",
                    value=len(annual_land_cover_series),
                    unit="years",
                    confidence=1.0,
                    data_class=urban.data_class,
                    evidence_grade=evidence_grade_for(urban.data_class),
                    evidence_ids=[satellite_evidence_id],
                ),
            ]
        )
        context.charts.append(
            Chart(
                id="historical_land_cover_timeseries",
                title="Annual site-versus-control land-cover history",
                kind="line",
                data=annual_land_cover_series,
                data_class=urban.data_class,
                evidence_ids=[satellite_evidence_id, control_evidence_id],
            )
        )
    snapshots = [latest, *(control.snapshot for control in urban.matched_controls)]
    control_scores = {
        control.snapshot.label: control.similarity_score
        for control in urban.matched_controls
    }
    context.charts.append(
        Chart(
            id="satellite_land_cover_context",
            title=(
                f"{land_cover_provider}: site and matched controls"
                if control_quality_passed
                else f"{land_cover_provider}: site and rejected candidates"
            ),
            kind="stacked_bar",
            data=[
                {
                    "location": snapshot.label,
                    "role": (
                        "site"
                        if snapshot is latest
                        else (
                            "matched_control"
                            if control_quality_passed
                            else "rejected_control_candidate"
                        )
                    ),
                    "image_year": snapshot.image_year,
                    "similarity_score": control_scores.get(snapshot.label, 1.0),
                    **canonical_land_cover(snapshot.segments),
                }
                for snapshot in snapshots
            ],
            data_class=urban.data_class,
            evidence_ids=[satellite_evidence_id, control_evidence_id],
        )
    )
    context.charts.append(
        Chart(
            id="regional_control_map",
            title=(
                "Satellite-matched regional control locations"
                if control_quality_passed
                else "Rejected regional control candidates"
            ),
            kind="point_map",
            data=[
                {
                    "location": latest.label,
                    "role": "site",
                    "latitude": latest.latitude,
                    "longitude": latest.longitude,
                    "similarity_score": 1.0,
                },
                *[
                    {
                        "location": control.snapshot.label,
                        "role": (
                            "matched_control"
                            if control_quality_passed
                            else "rejected_control_candidate"
                        ),
                        "latitude": control.site.latitude,
                        "longitude": control.site.longitude,
                        "similarity_score": control.similarity_score,
                    }
                    for control in urban.matched_controls
                ],
            ],
            data_class=DataClass.INFERRED,
            evidence_ids=[satellite_evidence_id, control_evidence_id],
        )
    )
    context.artifacts["matched_control_sites"] = (
        [control.site for control in urban.matched_controls]
        if control_quality_passed
        else []
    )
    context.artifacts["historical_satellite_change_available"] = (
        historical_change_available
    )
    context.artifacts["annual_land_cover_series"] = annual_land_cover_series
    context.artifacts["annual_land_cover_series_basis"] = str(
        urban.metadata.get("annual_land_cover_series_basis", "unknown")
    )
    context.artifacts["satellite_land_cover_evidence_id"] = satellite_evidence_id
    context.event(
        "urban_change_agent",
        (
            (
                f"Selected {len(urban.matched_controls)} regional controls from "
                f"{len(urban.candidate_snapshots)} satellite candidates"
            )
            if control_quality_passed
            else (
                f"Rejected {len(urban.matched_controls)} regional control candidates "
                "that failed the land-cover quality gate"
            )
        ),
        evidence_ids=[satellite_evidence_id, control_evidence_id],
        details={
            "historical_change_available": historical_change_available,
            "site_image_year": latest.image_year,
            "control_quality_passed": control_quality_passed,
        },
    )


def evidence_grade_for(
    data_class: DataClass, *, identified: bool = True, backcast: bool = False
) -> EvidenceGrade:
    """Map a data class onto the ordinal grade shown to users.

    The numeric `confidence` on a metric is a hand-set constant. It was being
    rendered as "56% confidence" beside a dollar figure, which reads as a
    probability. The grade says what is actually known about the input.
    """

    if backcast or not identified:
        return EvidenceGrade.C
    if data_class == DataClass.OBSERVED:
        return EvidenceGrade.A
    if data_class in {DataClass.UPLOADED, DataClass.INFERRED}:
        return EvidenceGrade.B
    return EvidenceGrade.C


def _control_label(annual: AnnualThermalDataset) -> str:
    """Name the comparison honestly on every surface that shows it.

    A ring 450 m from the site core is a local outer ring, not a regional
    control, and calling it regional in one place and local in another is how
    the dashboard ended up asserting matched controls on runs where the
    candidates were rejected.
    """

    method = str(annual.metadata.get("control_method", "unspecified"))
    if method == "satellite_land_cover_matched_regional":
        return "satellite land-cover matched regional controls"
    if method == "local_outer_ring":
        return "the disclosed local outer ring"
    if method == "fixture_regional_control":
        return "a simulated fixture control"
    return "an unspecified control"


def analyze_thermal_drift(context: RunContext, annual: AnnualThermalDataset) -> None:
    rows = annual.summaries
    if len(rows) < 2:
        raise ValueError(
            "thermal drift analysis requires at least two annual observations"
        )
    years = np.array([item.year for item in rows], dtype=float)
    gaps = np.array(
        [
            item.site_mean_temperature_c - item.control_mean_temperature_c
            for item in rows
        ]
    )
    eligibility_gaps = np.array(
        [item.site_eligible_hours - item.control_eligible_hours for item in rows],
        dtype=float,
    )
    estimate = estimate_drift(years, gaps)
    # Signed on purpose. Clamping this at zero rectified symmetric measurement
    # noise into a strictly positive reported exposure, so the expected output
    # grew with the error in the input and a site that measurably cooled was
    # reported as unaffected rather than improved.
    eligible_hours_delta = int(round(float(eligibility_gaps[0] - eligibility_gaps[-1])))

    backcast_years = [int(year) for year in (annual.metadata.get("backcast_years") or [])]
    simulated_series = annual.data_class == DataClass.SIMULATED
    control_label = _control_label(annual)

    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"thermal-{context.run_id}",
            source=annual.source,
            description="Site and disclosed-control historical temperature summaries",
            data_class=annual.data_class,
            activity_id=annual.activity_ids[0] if annual.activity_ids else None,
            endpoint=("/v1/heatmap" if annual.metadata.get("observed_years") else None),
            metadata={
                "baseline_year": rows[0].year,
                "latest_year": rows[-1].year,
                "threshold_c": context.request.temperature_eligibility_threshold_c,
                "activity_ids": annual.activity_ids,
                **annual.metadata,
            },
        )
    )
    calc_id = context.add_evidence(
        EvidenceRef(
            id=f"did-{context.run_id}",
            source="fortycool://analytics/difference-in-differences/v1",
            description="Difference-in-differences calculation against disclosed controls",
            data_class=DataClass.INFERRED,
            metadata={
                "formula": (
                    "ordinary-least-squares slope of (site-control) on year, "
                    "projected across the analysis window"
                ),
                "estimator": "ols_slope_projection",
                "observations": estimate.observations,
                "span_years": estimate.span_years,
                "slope_c_per_year": round(estimate.slope_c_per_year, 6),
                "standard_error_c": (
                    round(estimate.standard_error_c, 6)
                    if estimate.standard_error_c is not None
                    else None
                ),
                "confidence_interval_95_c": (
                    [round(estimate.ci_low_c, 6), round(estimate.ci_high_c, 6)]
                    if estimate.ci_low_c is not None and estimate.ci_high_c is not None
                    else None
                ),
                "uncertainty_basis": estimate.basis,
                "control_method": annual.metadata.get("control_method", "unspecified"),
                "backcast_years": backcast_years,
            },
        )
    )

    # The chart is drawn either way: a reader may look at the series even when
    # the headline number is withheld.
    context.charts.append(
        Chart(
            id="thermal_drift_timeseries",
            title=f"Site temperature versus {control_label}",
            kind="line",
            data=[
                {
                    "year": item.year,
                    "site_mean_temperature_c": item.site_mean_temperature_c,
                    "control_mean_temperature_c": item.control_mean_temperature_c,
                    "site_control_gap_c": round(
                        item.site_mean_temperature_c - item.control_mean_temperature_c,
                        3,
                    ),
                    "site_eligible_hours": item.site_eligible_hours,
                    "control_eligible_hours": item.control_eligible_hours,
                    "is_backcast": item.year in backcast_years,
                }
                for item in rows
            ],
            data_class=annual.data_class,
            evidence_ids=[evidence_id, calc_id],
        )
    )

    # A calibrated backcast is fixture data wearing the offset of one observed
    # year. When the window ends on a backcast the observed term cancels out of
    # the difference entirely, so the "measured" drift is arithmetic on a
    # hard-coded constant. Refuse to publish a number in that case, on every
    # endpoint rather than only on the gated demo route.
    if backcast_years and not simulated_series:
        context.metrics.append(
            Metric(
                id="thermal_drift_status",
                label="Thermal drift availability",
                value="withheld_calibrated_backcast",
                unit="status",
                confidence=1.0,
                data_class=DataClass.INFERRED,
                evidence_grade=EvidenceGrade.C,
                evidence_ids=[evidence_id, calc_id],
                caveats=[
                    "The annual series contains calibrated simulated backcast years for "
                    + ", ".join(str(year) for year in backcast_years)
                    + "; the difference-in-differences estimate is not identified from "
                    "observed data, so no drift value is published."
                ],
            )
        )
        context.warn(
            "Local thermal drift and its financial translation were withheld: "
            f"{len(backcast_years)} of {len(rows)} annual observations are calibrated "
            "simulated backcasts.",
            WarningCode.THERMAL_SERIES_BACKCAST,
        )
        context.mark_stage_degraded("thermal_drift")
        context.event(
            "temperature_intelligence_agent",
            "Withheld thermal drift because the annual series contains backcast years",
            status="blocked",
            evidence_ids=[evidence_id, calc_id],
            details={"backcast_years": backcast_years},
        )
        return

    if annual.data_class == DataClass.OBSERVED:
        confidence = 0.84
    elif (
        annual.data_class != DataClass.SIMULATED
        and annual.metadata.get("control_method")
        == "satellite_land_cover_matched_regional"
    ):
        confidence = 0.77
    else:
        confidence = 0.72
    grade = evidence_grade_for(
        annual.data_class, identified=estimate.identified, backcast=bool(backcast_years)
    )

    interval_caveats: list[str] = [estimate.basis]
    if estimate.ci_low_c is not None and estimate.ci_high_c is not None:
        interval_caveats.append(
            "95% interval "
            f"[{estimate.ci_low_c:.3f}, {estimate.ci_high_c:.3f}] °C."
        )
        if not estimate.distinguishable_from_zero:
            interval_caveats.append(
                "The interval includes zero: this series does not distinguish the "
                "measured drift from no drift at all."
            )
            context.warn(
                "Local thermal drift is not distinguishable from zero at 95% confidence "
                f"({estimate.drift_c:.3f} °C, interval "
                f"[{estimate.ci_low_c:.3f}, {estimate.ci_high_c:.3f}]).",
                WarningCode.DRIFT_NOT_DISTINGUISHABLE,
            )
    if simulated_series:
        context.warn(
            "The annual thermal series is simulated fixture data, so the drift, the "
            "hours, and any financial translation describe the demo series and not "
            "the entered coordinates.",
            WarningCode.THERMAL_SERIES_SIMULATED,
        )
        if not annual.metadata.get("location_dependent", True):
            context.warn(
                "Fixture thermal values do not vary with latitude or longitude; the "
                "same figures are produced for any site.",
                WarningCode.FIXTURE_SERIES_LOCATION_INDEPENDENT,
            )
    if annual.metadata.get("spatial_fallback_years"):
        context.warn(
            "Some annual heatmaps contained no tiles inside the site radius or beyond "
            "the control radius; nearest and farthest tile quartiles were substituted "
            "for "
            + ", ".join(
                str(year) for year in annual.metadata["spatial_fallback_years"]
            )
            + ".",
            WarningCode.SPATIAL_FALLBACK_ZONES,
        )

    rate_interval = (
        (
            estimate.ci_low_c / estimate.span_years,
            estimate.ci_high_c / estimate.span_years,
        )
        if estimate.ci_low_c is not None
        and estimate.ci_high_c is not None
        and estimate.span_years
        else (None, None)
    )
    context.metrics.extend(
        [
            Metric(
                id="local_thermal_drift_c",
                label=f"Local thermal drift versus {control_label}",
                value=round(estimate.drift_c, 3),
                unit="°C",
                confidence=confidence,
                data_class=DataClass.INFERRED,
                evidence_grade=grade,
                interval_low=(
                    round(estimate.ci_low_c, 3)
                    if estimate.ci_low_c is not None
                    else None
                ),
                interval_high=(
                    round(estimate.ci_high_c, 3)
                    if estimate.ci_high_c is not None
                    else None
                ),
                evidence_ids=[evidence_id, calc_id],
                caveats=interval_caveats,
            ),
            Metric(
                id="local_drift_rate_c_per_year",
                label="Local excess warming rate",
                value=round(estimate.slope_c_per_year, 3),
                unit="°C/year",
                confidence=max(0, confidence - 0.03),
                data_class=DataClass.INFERRED,
                evidence_grade=grade,
                interval_low=(
                    round(rate_interval[0], 4) if rate_interval[0] is not None else None
                ),
                interval_high=(
                    round(rate_interval[1], 4) if rate_interval[1] is not None else None
                ),
                evidence_ids=[evidence_id, calc_id],
                caveats=[
                    "The drift and the rate are the same ordinary-least-squares fit: "
                    "drift equals rate multiplied by the "
                    f"{estimate.span_years:.0f}-year window."
                ],
            ),
            Metric(
                id="temperature_eligible_hours_lost",
                label="Temperature-eligible cooling hours lost",
                value=eligible_hours_delta,
                unit="hours/year",
                confidence=max(0, confidence - 0.05),
                data_class=DataClass.INFERRED,
                evidence_grade=grade,
                evidence_ids=[evidence_id, calc_id],
                caveats=[
                    "Temperature-only eligibility is not equivalent to verified free-cooling operation",
                    "Signed: a negative value means the site gained eligible hours "
                    "relative to the control.",
                ],
            ),
        ]
    )

    land_cover_series = list(context.artifacts.get("annual_land_cover_series", []))
    land_cover_basis = str(
        context.artifacts.get("annual_land_cover_series_basis", "unknown")
    )
    land_cover_by_year = {
        int(item["year"]): item
        for item in land_cover_series
        if "year" in item and "local_excess_built_surface_percent" in item
    }
    paired_attribution = [
        {
            "year": item.year,
            "site_control_temperature_gap_c": round(
                item.site_mean_temperature_c - item.control_mean_temperature_c, 4
            ),
            "local_excess_built_surface_percent": float(
                land_cover_by_year[item.year]["local_excess_built_surface_percent"]
            ),
        }
        for item in rows
        if item.year in land_cover_by_year
    ]
    _record_land_cover_attribution(
        context,
        paired_attribution,
        basis=land_cover_basis,
        confidence=confidence,
        thermal_evidence_id=evidence_id,
    )

    context.artifacts["local_thermal_drift_c"] = estimate.drift_c
    context.artifacts["local_thermal_drift_estimate"] = estimate
    context.artifacts["temperature_eligible_hours_lost"] = eligible_hours_delta
    # Hours the site spends below the eligibility threshold in the latest year.
    # The financial model needs its complement: the drift penalty only applies
    # while the plant is actually above the economizer threshold.
    context.artifacts["latest_site_eligible_hours"] = rows[-1].site_eligible_hours
    context.event(
        "temperature_intelligence_agent",
        f"Compared {len(rows)} annual site observations with {control_label}",
        evidence_ids=[evidence_id, calc_id],
        details={
            "control_method": annual.metadata.get("control_method", "unspecified"),
            "estimator": "ols_slope_projection",
            "observations": estimate.observations,
        },
    )


def _record_land_cover_attribution(
    context: RunContext,
    paired_attribution: list[dict[str, Any]],
    *,
    basis: str,
    confidence: float,
    thermal_evidence_id: str,
) -> None:
    """Publish the thermal/land-cover association only when it can mean anything.

    A Pearson r over three to five points is not evidence: under the null the
    expected |r| at n=3 is 0.64 and the 5% critical value is 0.997. Worse, when
    the land-cover series is linearly interpolated between two snapshots and the
    thermal series trends linearly, the correlation is 1.0 by construction and
    measures the interpolation, not any association.
    """

    observations = len(paired_attribution)
    observed_basis = basis == "observed_per_year"
    if observations >= MINIMUM_ATTRIBUTION_YEARS and observed_basis:
        thermal_values = np.array(
            [item["site_control_temperature_gap_c"] for item in paired_attribution],
            dtype=float,
        )
        built_values = np.array(
            [item["local_excess_built_surface_percent"] for item in paired_attribution],
            dtype=float,
        )
        if np.std(thermal_values) <= 1e-9 or np.std(built_values) <= 1e-9:
            return
        correlation = float(np.corrcoef(thermal_values, built_values)[0, 1])
        low, high = fisher_interval_95(correlation, observations)
        satellite_evidence_id = context.artifacts.get(
            "satellite_land_cover_evidence_id"
        )
        attribution_evidence_id = context.add_evidence(
            EvidenceRef(
                id=f"thermal-land-cover-attribution-{context.run_id}",
                source="fortycool://analytics/thermal-land-cover-association/v1",
                description=(
                    "Association between annual local thermal gaps and local excess "
                    "built-surface share"
                ),
                data_class=DataClass.INFERRED,
                metadata={
                    "formula": (
                        "Pearson correlation(site-control temperature gap, "
                        "site-control built-surface share)"
                    ),
                    "paired_years": [item["year"] for item in paired_attribution],
                    "observations": paired_attribution,
                    "sample_size": observations,
                    "series_basis": basis,
                    "fisher_interval_95": [round(low, 4), round(high, 4)],
                },
            )
        )
        attribution_evidence_ids = [
            thermal_evidence_id,
            attribution_evidence_id,
            *([satellite_evidence_id] if satellite_evidence_id else []),
        ]
        context.metrics.append(
            Metric(
                id="thermal_land_cover_association_correlation",
                label="Thermal-gap versus local built-surface association",
                value=round(correlation, 3),
                unit="correlation",
                confidence=max(0.0, confidence - 0.2),
                data_class=DataClass.INFERRED,
                evidence_grade=EvidenceGrade.B,
                interval_low=round(low, 3),
                interval_high=round(high, 3),
                evidence_ids=attribution_evidence_ids,
                caveats=[
                    f"n={observations} observed years; Fisher 95% interval "
                    f"[{low:.3f}, {high:.3f}].",
                    "A temporal association supports a hypothesis but does not "
                    "establish causality",
                ],
            )
        )
        context.charts.append(
            Chart(
                id="thermal_land_cover_attribution",
                title="Thermal drift and local land-cover change",
                kind="dual_axis_line",
                data=paired_attribution,
                data_class=DataClass.INFERRED,
                evidence_ids=attribution_evidence_ids,
            )
        )
        return

    if not paired_attribution:
        return
    if not observed_basis:
        reason = (
            "the annual land-cover series is interpolated between two snapshots, so a "
            "correlation against a trending thermal series is one by construction"
        )
    else:
        reason = (
            f"only {observations} paired observed years are available and at least "
            f"{MINIMUM_ATTRIBUTION_YEARS} are required for the correlation to carry "
            "information"
        )
    context.metrics.append(
        Metric(
            id="thermal_land_cover_association_status",
            label="Thermal/land-cover attribution availability",
            value="withheld",
            unit="status",
            confidence=1.0,
            data_class=DataClass.INFERRED,
            evidence_grade=EvidenceGrade.C,
            evidence_ids=[thermal_evidence_id],
            caveats=[f"Attribution correlation withheld because {reason}."],
        )
    )
    context.warn(
        f"Thermal/land-cover attribution was withheld because {reason}.",
        WarningCode.ATTRIBUTION_WITHHELD,
    )


def record_model_results(
    context: RunContext, bundle: ModelBundle, *, data_class: DataClass
) -> None:
    is_simulated = data_class == DataClass.SIMULATED
    caveat = (
        "Backtest uses held-out simulated BMS telemetry"
        if is_simulated
        else "Backtest uses held-out user-uploaded BMS telemetry"
    )
    if is_simulated:
        # The regression fits the same linear equations the simulator used to
        # build the data, so a low error measures round-off in an identity, not
        # predictive skill on a real plant.
        context.warn(
            "The backtest is run against simulated telemetry generated by the same "
            "linear relationships the model fits, so its error does not measure "
            "predictive skill on a real facility."
        )
    if bundle.zero_variance_features:
        context.warn(
            "These features had no variation in training, so no response to them can "
            "be estimated: "
            + ", ".join(bundle.zero_variance_features)
            + "."
        )
    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"cooling-model-{context.run_id}",
            source="fortycool://models/ridge-digital-twin/v1",
            description="Cooling-demand and inlet-temperature regression backtest",
            data_class=data_class,
            metadata={
                "cooling_mae_kw": bundle.cooling_mae_kw,
                "inlet_mae_c": bundle.inlet_mae_c,
                "holdout_hours": len(bundle.backtest),
                "action_response_identifiable": bundle.action_response_identifiable,
                "cross_validation_folds": bundle.cross_validation_folds,
                "inlet_fold_mae_c": list(bundle.inlet_fold_mae_c),
                "cooling_fold_mae_kw": list(bundle.cooling_fold_mae_kw),
                "inlet_safety_buffer_c": bundle.inlet_safety_buffer_c,
                "inlet_safety_buffer_quantile": SAFETY_QUANTILE,
                "zero_variance_features": list(bundle.zero_variance_features),
                "identification_basis": (
                    "control variation present in training; this is a variance check, "
                    "not a test that the variation was exogenous"
                ),
            },
        )
    )
    context.metrics.extend(
        [
            Metric(
                id="cooling_model_mae_kw",
                label="Held-out cooling forecast MAE",
                value=bundle.cooling_mae_kw,
                unit="kW",
                confidence=bundle.confidence,
                data_class=data_class,
                evidence_grade=evidence_grade_for(data_class),
                evidence_ids=[evidence_id],
                caveats=[caveat],
            ),
            Metric(
                id="inlet_model_mae_c",
                label="Held-out inlet-temperature MAE",
                value=bundle.inlet_mae_c,
                unit="°C",
                confidence=bundle.confidence,
                data_class=data_class,
                evidence_grade=evidence_grade_for(data_class),
                evidence_ids=[evidence_id],
                caveats=[caveat],
            ),
            Metric(
                id="inlet_safety_buffer_c",
                label="Inlet-temperature safety buffer",
                value=bundle.inlet_safety_buffer_c,
                unit="°C",
                confidence=bundle.confidence,
                data_class=data_class,
                evidence_grade=evidence_grade_for(data_class),
                evidence_ids=[evidence_id],
                caveats=[
                    f"One-sided {int(SAFETY_QUANTILE * 100)}th-percentile absolute "
                    f"inlet error across {bundle.cross_validation_folds} rolling-origin "
                    "folds plus the final holdout block.",
                    "A mean absolute error is not a safety bound; this replaces it.",
                ],
            ),
        ]
    )
    context.charts.append(
        Chart(
            id="historical_day_backtest",
            title=(
                "Held-out simulated BMS backtest"
                if is_simulated
                else "Held-out uploaded BMS backtest"
            ),
            kind="line",
            data=dataframe_records(bundle.backtest),
            data_class=data_class,
            evidence_ids=[evidence_id],
        )
    )
    context.artifacts["model_evidence_id"] = evidence_id
    context.event(
        "asset_modeling_agent",
        f"Backtested cooling model on {len(bundle.backtest)} held-out hours",
        evidence_ids=[evidence_id],
        details={"confidence": bundle.confidence},
    )


# The action grid. These are declared constraints on what the service is allowed
# to propose, not an optimizer's search space: both models are linear, so the
# minimum of a linear objective over a box always sits on a vertex and the
# result is always the edge of whatever range is written here.
SUPPLY_AIR_DELTA_OPTIONS_C = (0.0, 0.5, 1.0, 1.5)
CHILLED_WATER_DELTA_OPTIONS_C = (0.0, 0.5, 1.0)
FAN_SPEED_DELTA_OPTIONS_PERCENT = (0.0, -5.0, -10.0)
MINIMUM_FAN_SPEED_PERCENT = 40.0

CONSTRAINT_CHECKS = (
    "maximum_setpoint_change",
    "supply_air_setpoint_range",
    "server_inlet_safety_margin",
    "minimum_model_confidence",
    "training_envelope",
    "control_variation_present",
)


@dataclass(frozen=True)
class CandidateEvaluation:
    supply_delta_c: float
    chilled_water_delta_c: float
    fan_delta_percent: float
    energy_kwh: float
    peak_kw: float
    max_inlet_c: float
    safe: bool
    rejection_reasons: list[str]
    envelope_breaches: dict[str, float] = field(default_factory=dict)


def optimize_operations(
    context: RunContext,
    bundle: ModelBundle,
    forecast: pd.DataFrame,
    constraints: SafetyConstraints,
) -> None:
    if not bundle.action_response_identifiable:
        # The buffer belongs on every predicted inlet temperature this service
        # publishes, not only on the ones attached to an approved action.
        predicted_inlet = float(
            bundle.inlet_model.predict(forecast).max() + bundle.inlet_safety_buffer_c
        )
        context.recommendations.append(
            Recommendation(
                id="hold-unidentified-action-response",
                action="hold_current_settings",
                description=(
                    "Telemetry does not contain enough control variation to estimate the effect "
                    "of changing setpoints or fan speed."
                ),
                expected_savings_kwh=0,
                expected_peak_reduction_kw=0,
                predicted_max_inlet_temperature_c=round(predicted_inlet, 3),
                safety_margin_c=0,
                confidence=bundle.confidence,
                verdict=SafetyVerdict.INSUFFICIENT_DATA,
                evidence_ids=[context.artifacts["model_evidence_id"]],
            )
        )
        context.warn(
            "Operating recommendation withheld: historical action response is not identifiable"
        )
        context.mark_stage_degraded("operations_recommendation")
        context.event(
            "evidence_and_safety_agent",
            "Withheld action because telemetry lacks sufficient control variation",
            status="blocked",
            evidence_ids=[context.artifacts["model_evidence_id"]],
        )
        return
    required = [
        constraints.maximum_server_inlet_temperature_c,
        constraints.minimum_safety_margin_c,
        constraints.supply_air_setpoint_min_c,
        constraints.supply_air_setpoint_max_c,
        constraints.maximum_setpoint_change_c,
        constraints.minimum_model_confidence,
    ]
    if any(value is None for value in required):
        context.recommendations.append(
            Recommendation(
                id="hold-missing-constraints",
                action="hold_current_settings",
                description="Operator-approved safety constraints are incomplete.",
                expected_savings_kwh=0,
                expected_peak_reduction_kw=0,
                predicted_max_inlet_temperature_c=float(
                    bundle.inlet_model.predict(forecast).max()
                    + bundle.inlet_safety_buffer_c
                ),
                safety_margin_c=0,
                confidence=bundle.confidence,
                verdict=SafetyVerdict.INSUFFICIENT_DATA,
            )
        )
        context.warn(
            "Operational recommendation withheld: safety constraints are incomplete"
        )
        context.mark_stage_degraded("operations_recommendation")
        context.event(
            "evidence_and_safety_agent",
            "Withheld operating recommendation because constraints are incomplete",
            status="blocked",
        )
        return

    step_hours = context.request.simulation.interval_minutes / 60
    baseline_cooling = np.maximum(bundle.cooling_model.predict(forecast), 0)
    baseline_inlet = bundle.inlet_model.predict(forecast)
    baseline_energy = float(baseline_cooling.sum() * step_hours)
    baseline_peak = float(baseline_cooling.max())
    current_supply = float(forecast["supply_air_setpoint_c"].iloc[0])
    current_chilled = float(forecast["chilled_water_supply_c"].iloc[0])

    # Controls with no training variation cannot be moved: the coefficient that
    # would justify the move was fitted on a constant.
    frozen_controls = {
        column
        for column in (
            "supply_air_setpoint_c",
            "chilled_water_supply_c",
            "fan_speed_percent",
        )
        if column in bundle.zero_variance_features
    }
    candidates: list[CandidateEvaluation] = []
    for supply_delta in SUPPLY_AIR_DELTA_OPTIONS_C:
        for chilled_delta in CHILLED_WATER_DELTA_OPTIONS_C:
            for fan_delta in FAN_SPEED_DELTA_OPTIONS_PERCENT:
                candidate = forecast.copy()
                candidate["supply_air_setpoint_c"] = current_supply + supply_delta
                candidate["chilled_water_supply_c"] = current_chilled + chilled_delta
                candidate["fan_speed_percent"] = np.maximum(
                    MINIMUM_FAN_SPEED_PERCENT,
                    candidate["fan_speed_percent"] + fan_delta,
                )
                cooling = np.maximum(bundle.cooling_model.predict(candidate), 0)
                inlet = bundle.inlet_model.predict(candidate)
                # One-sided upper bound, not the mean absolute error. The mean
                # is roughly 2.4 times too small for a safety limit and
                # understated the true inlet temperature at the recommended
                # operating point.
                max_inlet = float(inlet.max() + bundle.inlet_safety_buffer_c)
                rejection_reasons: list[str] = []
                if max(supply_delta, chilled_delta) > float(
                    constraints.maximum_setpoint_change_c
                ):
                    rejection_reasons.append("maximum_setpoint_change")
                if not (
                    float(constraints.supply_air_setpoint_min_c)
                    <= current_supply + supply_delta
                    <= float(constraints.supply_air_setpoint_max_c)
                ):
                    rejection_reasons.append("supply_air_setpoint_range")
                available_limit = float(
                    constraints.maximum_server_inlet_temperature_c
                ) - float(constraints.minimum_safety_margin_c)
                if max_inlet > available_limit:
                    rejection_reasons.append("server_inlet_safety_margin")
                if bundle.confidence < float(constraints.minimum_model_confidence):
                    rejection_reasons.append("minimum_model_confidence")
                envelope_breaches = bundle.inlet_model.outside_training_envelope(
                    candidate
                )
                envelope_breaches.update(
                    bundle.cooling_model.outside_training_envelope(candidate)
                )
                # Only a breach on a control we are actively changing disqualifies
                # the action; weather that runs past the training range is a
                # caveat on the forecast, not on the recommendation.
                actionable_breaches = {
                    column: amount
                    for column, amount in envelope_breaches.items()
                    if column
                    in {
                        "supply_air_setpoint_c",
                        "chilled_water_supply_c",
                        "fan_speed_percent",
                    }
                }
                if actionable_breaches:
                    rejection_reasons.append("training_envelope")
                moved = {
                    "supply_air_setpoint_c": supply_delta,
                    "chilled_water_supply_c": chilled_delta,
                    "fan_speed_percent": fan_delta,
                }
                if any(
                    delta != 0.0 and column in frozen_controls
                    for column, delta in moved.items()
                ):
                    rejection_reasons.append("control_variation_present")
                candidates.append(
                    CandidateEvaluation(
                        supply_delta_c=supply_delta,
                        chilled_water_delta_c=chilled_delta,
                        fan_delta_percent=fan_delta,
                        energy_kwh=float(cooling.sum() * step_hours),
                        peak_kw=float(cooling.max()),
                        max_inlet_c=max_inlet,
                        safe=not rejection_reasons,
                        rejection_reasons=rejection_reasons,
                        envelope_breaches=actionable_breaches,
                    )
                )

    safe_candidates = [candidate for candidate in candidates if candidate.safe]
    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"optimizer-{context.run_id}",
            source="fortycool://analytics/constraint-envelope-evaluation/v1",
            description=(
                "Evaluation of the declared cooling-action envelope against the "
                "operator's safety constraints"
            ),
            data_class=DataClass.INFERRED,
            metadata={
                "candidate_count": len(candidates),
                "safe_candidate_count": len(safe_candidates),
                "constraints": constraints.model_dump(),
                # Declared, not discovered. Both models are linear, so the
                # selected action always lands on an edge of this grid.
                "action_grid": {
                    "supply_air_delta_c": list(SUPPLY_AIR_DELTA_OPTIONS_C),
                    "chilled_water_delta_c": list(CHILLED_WATER_DELTA_OPTIONS_C),
                    "fan_speed_delta_percent": list(FAN_SPEED_DELTA_OPTIONS_PERCENT),
                    "minimum_fan_speed_percent": MINIMUM_FAN_SPEED_PERCENT,
                },
                "selection_method": "linear_models_over_a_declared_box",
                "inlet_safety_buffer_c": bundle.inlet_safety_buffer_c,
                "inlet_safety_buffer_basis": (
                    f"{int(SAFETY_QUANTILE * 100)}th percentile of absolute inlet error "
                    f"across {bundle.cross_validation_folds} rolling-origin folds"
                ),
                "frozen_controls": sorted(frozen_controls),
                "rejection_counts": {
                    reason: sum(
                        1
                        for candidate in candidates
                        if reason in candidate.rejection_reasons
                    )
                    for reason in CONSTRAINT_CHECKS
                },
            },
        )
    )
    if not safe_candidates:
        context.recommendations.append(
            Recommendation(
                id="hold-no-safe-candidate",
                action="hold_current_settings",
                description="No candidate action passed all configured safety checks.",
                expected_savings_kwh=0,
                expected_peak_reduction_kw=0,
                predicted_max_inlet_temperature_c=round(
                    float(baseline_inlet.max()) + bundle.inlet_safety_buffer_c, 3
                ),
                safety_margin_c=round(
                    float(constraints.maximum_server_inlet_temperature_c)
                    - float(baseline_inlet.max())
                    - bundle.inlet_safety_buffer_c,
                    3,
                ),
                confidence=bundle.confidence,
                verdict=SafetyVerdict.HOLD,
                evidence_ids=[evidence_id],
            )
        )
        return

    best = min(
        safe_candidates, key=lambda candidate: (candidate.energy_kwh, candidate.peak_kw)
    )
    savings = max(0.0, baseline_energy - best.energy_kwh)
    peak_reduction = max(0.0, baseline_peak - best.peak_kw)
    margin = float(constraints.maximum_server_inlet_temperature_c) - best.max_inlet_c
    contains_simulated_evidence = any(
        item.data_class == DataClass.SIMULATED for item in context.evidence
    )
    verdict = (
        SafetyVerdict.APPROVED_WITH_WARNING
        if contains_simulated_evidence or context.assumptions
        else SafetyVerdict.APPROVED
    )
    description = (
        f"Raise supply air by {best.supply_delta_c:.1f}°C, raise chilled water by "
        f"{best.chilled_water_delta_c:.1f}°C, and adjust fan speed by "
        f"{best.fan_delta_percent:.0f} percentage points for the forecast window."
    )
    at_grid_edge = (
        best.supply_delta_c == max(SUPPLY_AIR_DELTA_OPTIONS_C)
        or best.chilled_water_delta_c == max(CHILLED_WATER_DELTA_OPTIONS_C)
        or best.fan_delta_percent == min(FAN_SPEED_DELTA_OPTIONS_PERCENT)
    )
    if at_grid_edge:
        context.warn(
            "The selected action sits on the edge of the declared action envelope. "
            "Both models are linear, so the evaluation returns a corner of that box "
            "rather than an interior optimum; widening the envelope moves the answer."
        )
    decision_evidence_ids = [context.artifacts["model_evidence_id"], evidence_id]
    if forecast_evidence_id := context.artifacts.get("forecast_evidence_id"):
        decision_evidence_ids.insert(1, forecast_evidence_id)
    recommendation = Recommendation(
        id="optimized-12h-plan",
        action="apply_simulated_12h_plan",
        description=description,
        expected_savings_kwh=round(savings, 3),
        expected_peak_reduction_kw=round(peak_reduction, 3),
        predicted_max_inlet_temperature_c=round(best.max_inlet_c, 3),
        safety_margin_c=round(margin, 3),
        confidence=bundle.confidence,
        verdict=verdict,
        constraints_checked=list(CONSTRAINT_CHECKS),
        evidence_ids=decision_evidence_ids,
    )
    context.recommendations.append(recommendation)

    baseline = forecast[["timestamp"]].copy()
    baseline["baseline_cooling_kw"] = np.round(baseline_cooling, 3)
    best_frame = forecast.copy()
    best_frame["supply_air_setpoint_c"] = current_supply + best.supply_delta_c
    best_frame["chilled_water_supply_c"] = current_chilled + best.chilled_water_delta_c
    best_frame["fan_speed_percent"] = np.maximum(
        40, best_frame["fan_speed_percent"] + best.fan_delta_percent
    )
    baseline["optimized_cooling_kw"] = np.round(
        np.maximum(bundle.cooling_model.predict(best_frame), 0), 3
    )
    baseline["forecast_temperature_c"] = forecast["outdoor_temperature_c"].round(3)
    context.charts.append(
        Chart(
            id="baseline_vs_optimized",
            title="12-hour baseline versus optimized cooling demand",
            kind="area",
            data=dataframe_records(baseline),
            data_class=DataClass.INFERRED,
            evidence_ids=decision_evidence_ids,
        )
    )
    context.metrics.extend(
        [
            Metric(
                id="forecast_savings_kwh",
                label="Estimated forecast-window savings",
                value=round(savings, 3),
                unit="kWh",
                confidence=bundle.confidence,
                data_class=DataClass.INFERRED,
                evidence_grade=EvidenceGrade.C if context.request.telemetry.source.value == "simulated" else EvidenceGrade.B,
                evidence_ids=decision_evidence_ids,
                caveats=[
                    "Derived from a simulated facility digital twin"
                    if context.request.telemetry.source.value == "simulated"
                    else "Derived from uploaded telemetry and modeled forecast conditions"
                ],
            ),
            Metric(
                id="forecast_safety_margin_c",
                label="Predicted inlet-temperature safety margin",
                value=round(margin, 3),
                unit="°C",
                confidence=bundle.confidence,
                data_class=DataClass.INFERRED,
                evidence_grade=EvidenceGrade.C if context.request.telemetry.source.value == "simulated" else EvidenceGrade.B,
                evidence_ids=decision_evidence_ids,
            ),
        ]
    )
    context.event(
        "decision_agent",
        f"Evaluated {len(candidates)} declared actions and retained "
        f"{len(safe_candidates)} that passed every safety constraint",
        evidence_ids=decision_evidence_ids,
    )
    context.event(
        "evidence_and_safety_agent",
        f"Approved the advisory plan with a {margin:.2f}°C modeled safety margin",
        status=verdict.value,
        evidence_ids=decision_evidence_ids,
    )


def _resolved_economics(context: RunContext) -> tuple[EconomicsInput, bool]:
    """Return economics with demo defaults filled, without mutating the request.

    The previous version wrote defaults back onto `context.request.economics`,
    which made the function non-idempotent: a second call recorded no
    assumptions because the fields were already populated, so the assumption
    trail depended on call order.
    """

    economics = context.request.economics.model_copy(deep=True)
    if not context.request.use_demo_defaults:
        return economics, False
    defaults: list[tuple[str, Any, str]] = [
        ("electricity_price_per_kwh", 0.09, "Illustrative electricity price"),
        ("horizon_years", 10, "Illustrative investment horizon"),
        ("discount_rate", 0.08, "Illustrative discount rate"),
        (
            "annual_tariff_inflation",
            0.0,
            "No tariff escalation assumed; set economics.annual_tariff_inflation to model one",
        ),
        (
            "temperature_sensitivity_per_c",
            DEFAULT_TEMPERATURE_SENSITIVITY_PER_C,
            "Illustrative cooling-plant sensitivity; not a measured property of this facility",
        ),
    ]
    used_default_sensitivity = False
    for field, default, reason in defaults:
        if getattr(economics, field) is None:
            setattr(economics, field, default)
            context.assumptions.append(
                Assumption(field=f"economics.{field}", value=default, reason=reason)
            )
            if field == "temperature_sensitivity_per_c":
                used_default_sensitivity = True
    return economics, used_default_sensitivity


def calculate_investment_impact(context: RunContext) -> None:
    if "local_thermal_drift_c" not in context.artifacts:
        return
    economics, used_default_sensitivity = _resolved_economics(context)
    price = economics.electricity_price_per_kwh
    horizon = economics.horizon_years
    discount = economics.discount_rate
    sensitivity = economics.temperature_sensitivity_per_c
    inflation = economics.annual_tariff_inflation or 0.0
    if price is None or horizon is None or discount is None:
        context.warn(
            "Financial impact omitted: tariff, horizon, or discount rate is missing"
        )
        return
    if sensitivity is None:
        sensitivity = DEFAULT_TEMPERATURE_SENSITIVITY_PER_C
        used_default_sensitivity = True
        context.assumptions.append(
            Assumption(
                field="economics.temperature_sensitivity_per_c",
                value=sensitivity,
                reason=(
                    "Illustrative cooling-plant sensitivity; not a measured property "
                    "of this facility"
                ),
            )
        )

    facility = resolve_facility_parameters(context.request.facility)
    load_kw = facility.typical_it_load_mw * 1000
    # The drift penalty falls on the cooling plant, not on the IT load, so the
    # base scales with PUE. `reported_pue` used to be accepted, validated, and
    # then ignored, which meant the dial on the setup screen moved the answer by
    # exactly zero dollars.
    cooling_reference_kw = load_kw * max(facility.pue - 1.0, 0.12) * COOLING_LOAD_FACTOR

    # The penalty only accrues while the plant is above the economizer
    # threshold. Applying it to all 8,760 hours overstated the exposure by the
    # inverse of the eligible-hours share, using the same repo's own definition
    # of eligibility as hours below that threshold.
    eligible_hours = context.artifacts.get("latest_site_eligible_hours")
    if eligible_hours is None:
        penalty_hours = float(HOURS_PER_YEAR)
        hours_basis = "all 8,760 hours (eligibility hours unavailable)"
        context.warn(
            "Financial exposure assumes the drift penalty applies for the whole year "
            "because temperature-eligible hours were not available."
        )
    else:
        penalty_hours = float(max(0, HOURS_PER_YEAR - int(eligible_hours)))
        hours_basis = (
            f"{penalty_hours:.0f} hours above the "
            f"{context.request.temperature_eligibility_threshold_c:g} °C eligibility "
            "threshold in the latest analysed year"
        )

    local_drift = float(context.artifacts["local_thermal_drift_c"])
    estimate: DriftEstimate | None = context.artifacts.get(
        "local_thermal_drift_estimate"
    )
    # Signed. A site that measurably cooled relative to its control shows a
    # benefit rather than a floor of zero.
    kwh_per_degree = cooling_reference_kw * sensitivity * penalty_hours
    annual_penalty_kwh = kwh_per_degree * local_drift

    def _npv(energy_kwh: float) -> float:
        total = 0.0
        for year in range(1, horizon + 1):
            escalated_price = price * ((1 + inflation) ** (year - 1))
            total += (energy_kwh * escalated_price) / ((1 + discount) ** year)
        return total

    npv = _npv(annual_penalty_kwh)
    demand_charge_annual = 0.0
    if economics.demand_charge_per_kw_month:
        # Sustained warming raises the cooling-plant peak by the same
        # sensitivity, and a demand tariff bills that peak every month.
        peak_penalty_kw = cooling_reference_kw * sensitivity * local_drift
        demand_charge_annual = (
            peak_penalty_kw * economics.demand_charge_per_kw_month * 12
        )
        npv += sum(
            demand_charge_annual / ((1 + discount) ** year)
            for year in range(1, horizon + 1)
        )

    interval_kwh_low = interval_kwh_high = None
    interval_npv_low = interval_npv_high = None
    if estimate is not None and estimate.ci_low_c is not None:
        interval_kwh_low = kwh_per_degree * estimate.ci_low_c
        interval_kwh_high = kwh_per_degree * estimate.ci_high_c
        interval_npv_low = _npv(interval_kwh_low)
        interval_npv_high = _npv(interval_kwh_high)
        if interval_npv_low > interval_npv_high:
            interval_npv_low, interval_npv_high = interval_npv_high, interval_npv_low
            interval_kwh_low, interval_kwh_high = interval_kwh_high, interval_kwh_low

    circular = (
        used_default_sensitivity
        and context.request.telemetry.source.value == "simulated"
    )
    if circular:
        context.warn(
            "The demo path is circular: the simulated telemetry is generated with the "
            "same temperature-sensitivity constant that this model uses to convert "
            "degrees into energy, so the backtest does not validate the money figure.",
            WarningCode.CIRCULAR_SENSITIVITY_CONSTANT,
        )

    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"investment-{context.run_id}",
            source="fortycool://finance/thermal-drift-npv/v1",
            description="Scenario translation of local thermal drift into cooling energy and NPV",
            data_class=DataClass.INFERRED,
            metadata={
                "temperature_sensitivity_per_c": sensitivity,
                "temperature_sensitivity_basis": (
                    "fraction of cooling-plant load per degree Celsius; supplied by the "
                    "operator or defaulted, never measured by this analysis"
                ),
                "temperature_sensitivity_is_default": used_default_sensitivity,
                "cooling_reference_kw": round(cooling_reference_kw, 3),
                "reported_pue": facility.pue,
                "penalty_hours_per_year": penalty_hours,
                "penalty_hours_basis": hours_basis,
                "electricity_price_per_kwh": price,
                "annual_tariff_inflation": inflation,
                "demand_charge_per_kw_month": economics.demand_charge_per_kw_month,
                "annual_demand_charge_usd": round(demand_charge_annual, 2),
                "horizon_years": horizon,
                "discount_rate": discount,
                "circular_with_simulator_constant": circular,
            },
        )
    )
    caveats = [
        "Indicative scenario based on a simulated facility profile and tariff assumptions",
        f"Energy penalty applied over {hours_basis}.",
        "Sensitivity is an operator input, not a measured property of this facility.",
    ]
    if interval_npv_low is not None:
        caveats.append(
            f"95% interval [{interval_npv_low:,.0f}, {interval_npv_high:,.0f}] USD, "
            "propagated from the thermal-drift interval."
        )
        if interval_npv_low <= 0 <= interval_npv_high:
            caveats.append(
                "The interval spans zero: this analysis does not establish a financial "
                "exposure distinguishable from none."
            )
    grade = EvidenceGrade.C if circular else EvidenceGrade.B
    context.metrics.extend(
        [
            Metric(
                id="annual_thermal_drift_energy_penalty_kwh",
                label="Annual cooling-energy penalty from local drift",
                value=round(annual_penalty_kwh, 1),
                unit="kWh/year",
                confidence=0.61,
                data_class=DataClass.INFERRED,
                evidence_grade=grade,
                interval_low=(
                    round(interval_kwh_low, 1) if interval_kwh_low is not None else None
                ),
                interval_high=(
                    round(interval_kwh_high, 1)
                    if interval_kwh_high is not None
                    else None
                ),
                evidence_ids=[f"did-{context.run_id}", evidence_id],
                caveats=caveats,
            ),
            Metric(
                id="thermal_drift_npv",
                label="Lease-life thermal-drift exposure",
                value=round(npv, 2),
                unit="USD NPV",
                confidence=0.56,
                data_class=DataClass.INFERRED,
                evidence_grade=grade,
                interval_low=(
                    round(interval_npv_low, 2)
                    if interval_npv_low is not None
                    else None
                ),
                interval_high=(
                    round(interval_npv_high, 2)
                    if interval_npv_high is not None
                    else None
                ),
                evidence_ids=[f"did-{context.run_id}", evidence_id],
                caveats=caveats,
            ),
        ]
    )
    context.event(
        "investment_analyst_agent",
        f"Calculated an indicative {horizon}-year thermal-drift NPV",
        evidence_ids=[evidence_id],
    )
