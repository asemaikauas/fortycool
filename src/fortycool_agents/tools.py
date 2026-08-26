from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Any

import numpy as np
import pandas as pd

from .context import RunContext
from .modeling import ModelBundle
from .models import (
    Assumption,
    Chart,
    DataClass,
    EvidenceRef,
    Metric,
    Recommendation,
    SafetyConstraints,
    SafetyVerdict,
)
from .providers.fixture import AnnualThermalDataset
from .providers.urban import (
    UrbanContextDataset,
    canonical_land_cover,
    tree_canopy_share,
)
from .simulation import resolve_facility_parameters


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
    slope = float(np.polyfit(years - years[0], gaps, 1)[0])
    latest_local_drift = float(gaps[-1] - gaps[0])
    lost_eligible_hours = max(0, round(eligibility_gaps[0] - eligibility_gaps[-1]))
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
                "formula": "(site_latest-site_baseline)-(control_latest-control_baseline)"
            },
        )
    )
    context.metrics.extend(
        [
            Metric(
                id="local_thermal_drift_c",
                label="Local thermal drift versus control",
                value=round(latest_local_drift, 3),
                unit="°C",
                confidence=confidence,
                data_class=DataClass.INFERRED,
                evidence_ids=[evidence_id, calc_id],
            ),
            Metric(
                id="local_drift_rate_c_per_year",
                label="Local excess warming rate",
                value=round(slope, 3),
                unit="°C/year",
                confidence=max(0, confidence - 0.03),
                data_class=DataClass.INFERRED,
                evidence_ids=[evidence_id, calc_id],
            ),
            Metric(
                id="temperature_eligible_hours_lost",
                label="Temperature-eligible cooling hours lost",
                value=lost_eligible_hours,
                unit="hours/year",
                confidence=max(0, confidence - 0.05),
                data_class=DataClass.INFERRED,
                evidence_ids=[evidence_id, calc_id],
                caveats=[
                    "Temperature-only eligibility is not equivalent to verified free-cooling operation"
                ],
            ),
        ]
    )
    context.charts.append(
        Chart(
            id="thermal_drift_timeseries",
            title="Site temperature versus disclosed control",
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
                }
                for item in rows
            ],
            data_class=annual.data_class,
            evidence_ids=[evidence_id, calc_id],
        )
    )
    land_cover_series = list(context.artifacts.get("annual_land_cover_series", []))
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
    if len(paired_attribution) >= 3 and context.artifacts.get("matched_control_sites"):
        thermal_values = np.array(
            [item["site_control_temperature_gap_c"] for item in paired_attribution],
            dtype=float,
        )
        built_values = np.array(
            [item["local_excess_built_surface_percent"] for item in paired_attribution],
            dtype=float,
        )
        if np.std(thermal_values) > 1e-9 and np.std(built_values) > 1e-9:
            correlation = float(np.corrcoef(thermal_values, built_values)[0, 1])
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
                    },
                )
            )
            attribution_evidence_ids = [
                evidence_id,
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
                    evidence_ids=attribution_evidence_ids,
                    caveats=[
                        "A small-sample temporal association supports a hypothesis but does not establish causality"
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
    context.artifacts["local_thermal_drift_c"] = latest_local_drift
    context.artifacts["temperature_eligible_hours_lost"] = lost_eligible_hours
    context.event(
        "temperature_intelligence_agent",
        f"Compared {len(rows)} annual site observations with disclosed controls",
        evidence_ids=[evidence_id, calc_id],
        details={
            "control_method": annual.metadata.get("control_method", "unspecified")
        },
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
                evidence_ids=[evidence_id],
                caveats=[caveat],
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


def optimize_operations(
    context: RunContext,
    bundle: ModelBundle,
    forecast: pd.DataFrame,
    constraints: SafetyConstraints,
) -> None:
    if not bundle.action_response_identifiable:
        predicted_inlet = float(bundle.inlet_model.predict(forecast).max())
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
        context.warnings.append(
            "Operating recommendation withheld: historical action response is not identifiable"
        )
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
                ),
                safety_margin_c=0,
                confidence=bundle.confidence,
                verdict=SafetyVerdict.INSUFFICIENT_DATA,
            )
        )
        context.warnings.append(
            "Operational recommendation withheld: safety constraints are incomplete"
        )
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

    candidates: list[CandidateEvaluation] = []
    for supply_delta in (0.0, 0.5, 1.0, 1.5):
        for chilled_delta in (0.0, 0.5, 1.0):
            for fan_delta in (0.0, -5.0, -10.0):
                candidate = forecast.copy()
                candidate["supply_air_setpoint_c"] = current_supply + supply_delta
                candidate["chilled_water_supply_c"] = current_chilled + chilled_delta
                candidate["fan_speed_percent"] = np.maximum(
                    40, candidate["fan_speed_percent"] + fan_delta
                )
                cooling = np.maximum(bundle.cooling_model.predict(candidate), 0)
                inlet = bundle.inlet_model.predict(candidate)
                max_inlet = float(inlet.max() + bundle.inlet_mae_c)
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
                    )
                )

    safe_candidates = [candidate for candidate in candidates if candidate.safe]
    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"optimizer-{context.run_id}",
            source="fortycool://optimizer/enumeration/v1",
            description="Constrained enumeration of digital-twin cooling actions",
            data_class=DataClass.INFERRED,
            metadata={
                "candidate_count": len(candidates),
                "safe_candidate_count": len(safe_candidates),
                "constraints": constraints.model_dump(),
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
                predicted_max_inlet_temperature_c=round(float(baseline_inlet.max()), 3),
                safety_margin_c=round(
                    float(constraints.maximum_server_inlet_temperature_c)
                    - float(baseline_inlet.max()),
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
        constraints_checked=[
            "maximum_setpoint_change",
            "supply_air_setpoint_range",
            "server_inlet_safety_margin",
            "minimum_model_confidence",
        ],
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
                evidence_ids=decision_evidence_ids,
            ),
        ]
    )
    context.event(
        "decision_agent",
        f"Evaluated {len(candidates)} actions and retained {len(safe_candidates)} safe candidates",
        evidence_ids=decision_evidence_ids,
    )
    context.event(
        "evidence_and_safety_agent",
        f"Approved the advisory plan with a {margin:.2f}°C modeled safety margin",
        status=verdict.value,
        evidence_ids=decision_evidence_ids,
    )


def calculate_investment_impact(context: RunContext) -> None:
    if "local_thermal_drift_c" not in context.artifacts:
        return
    economics = context.request.economics
    price = economics.electricity_price_per_kwh
    horizon = economics.horizon_years
    discount = economics.discount_rate
    if context.request.use_demo_defaults:
        defaults: list[tuple[str, Any, Any, str]] = [
            (
                "electricity_price_per_kwh",
                price,
                0.09,
                "Illustrative electricity price",
            ),
            ("horizon_years", horizon, 10, "Illustrative investment horizon"),
            ("discount_rate", discount, 0.08, "Illustrative discount rate"),
        ]
        for field, current, default, reason in defaults:
            if current is None:
                setattr(economics, field, default)
                context.assumptions.append(
                    Assumption(field=f"economics.{field}", value=default, reason=reason)
                )
        price = economics.electricity_price_per_kwh
        horizon = economics.horizon_years
        discount = economics.discount_rate
    if price is None or horizon is None or discount is None:
        context.warnings.append(
            "Financial impact omitted: tariff, horizon, or discount rate is missing"
        )
        return

    facility = resolve_facility_parameters(context.request.facility)
    load_kw = facility.typical_it_load_mw * 1000
    local_drift = float(context.artifacts["local_thermal_drift_c"])
    annual_penalty_kwh = max(0.0, load_kw * 0.0038 * local_drift * 8760)
    annual_cost = annual_penalty_kwh * price
    npv = sum(annual_cost / ((1 + discount) ** year) for year in range(1, horizon + 1))
    evidence_id = context.add_evidence(
        EvidenceRef(
            id=f"investment-{context.run_id}",
            source="fortycool://finance/thermal-drift-npv/v1",
            description="Scenario translation of local thermal drift into cooling energy and NPV",
            data_class=DataClass.INFERRED,
            metadata={
                "temperature_sensitivity_per_c": 0.0038,
                "electricity_price_per_kwh": price,
                "horizon_years": horizon,
                "discount_rate": discount,
            },
        )
    )
    caveat = "Indicative scenario based on a simulated facility profile and tariff assumptions"
    context.metrics.extend(
        [
            Metric(
                id="annual_thermal_drift_energy_penalty_kwh",
                label="Annual cooling-energy penalty from local drift",
                value=round(annual_penalty_kwh, 1),
                unit="kWh/year",
                confidence=0.61,
                data_class=DataClass.INFERRED,
                evidence_ids=[f"did-{context.run_id}", evidence_id],
                caveats=[caveat],
            ),
            Metric(
                id="thermal_drift_npv",
                label="Lease-life thermal-drift exposure",
                value=round(npv, 2),
                unit="USD NPV",
                confidence=0.56,
                data_class=DataClass.INFERRED,
                evidence_ids=[f"did-{context.run_id}", evidence_id],
                caveats=[caveat],
            ),
        ]
    )
    context.event(
        "investment_analyst_agent",
        f"Calculated an indicative {horizon}-year thermal-drift NPV",
        evidence_ids=[evidence_id],
    )
