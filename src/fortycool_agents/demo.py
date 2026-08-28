from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    AnalysisResponse,
    DataClass,
    RunStatus,
    VerifiedDemoResponse,
)
from .storage import RunRepository


def _saved_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _verified_manifest(
    run: AnalysisResponse, *, saved_at: datetime
) -> VerifiedDemoResponse | None:
    if run.status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}:
        return None
    metrics = {metric.id: metric for metric in run.metrics}
    if not {"local_thermal_drift_c", "thermal_drift_npv"} <= metrics.keys():
        return None

    thermal = next(
        (
            evidence
            for evidence in run.evidence
            if evidence.source.startswith("https://api.fortyguard.com/v1/heatmap")
            and evidence.id.startswith("thermal-")
        ),
        None,
    )
    satellite = next(
        (
            evidence
            for evidence in run.evidence
            if evidence.metadata.get("dataset_id") == "GOOGLE/DYNAMICWORLD/V1"
            and evidence.data_class == DataClass.OBSERVED
        ),
        None,
    )
    if thermal is None or satellite is None:
        return None

    thermal_years = sorted(int(year) for year in thermal.metadata.get("observed_years", []))
    consecutive_years = thermal_years == list(
        range(thermal_years[0], thermal_years[-1] + 1)
    ) if thermal_years else False
    if (
        len(thermal_years) < 3
        or not consecutive_years
        or thermal.metadata.get("backcast_years")
    ):
        return None
    if not satellite.metadata.get("historical_control_stability_verified"):
        return None
    control_status = metrics.get("regional_control_match_status")
    built_change = metrics.get("local_excess_built_surface_change_percentage_points")
    if (
        control_status is None
        or control_status.value != "accepted"
        or thermal.metadata.get("control_method")
        != "satellite_land_cover_matched_regional"
        or len(thermal.metadata.get("control_sites", [])) < 2
        or thermal.id not in metrics["local_thermal_drift_c"].evidence_ids
        or built_change is None
        or satellite.id not in built_change.evidence_ids
    ):
        return None

    chart_map = {chart.id: chart for chart in run.charts}
    heatmap = chart_map.get("fortyguard_heatmap")
    operations = chart_map.get("temperature_forecast_12h")
    observed_heatmap = bool(heatmap and heatmap.data_class == DataClass.OBSERVED)
    operations_class = operations.data_class if operations else None
    verified_at = max(thermal.retrieved_at, satellite.retrieved_at)
    notes = [
        (
            f"FortyGuard thermal history covers {thermal_years[0]}-"
            f"{thermal_years[-1]} with no calibrated backcast years."
        ),
        "Google Dynamic World history, control similarity, and historical stability passed all gates.",
        "This is a saved completed analysis, not a newly executed provider request.",
    ]
    if observed_heatmap:
        notes.append("The saved run includes an observed FortyGuard forecast heatmap.")
    else:
        notes.append(
            "Operational forecast and BMS panels may be simulated; their data classes remain visible."
        )
    return VerifiedDemoResponse(
        run=run,
        saved_at=saved_at,
        verified_at=verified_at,
        thermal_years=thermal_years,
        observed_heatmap=observed_heatmap,
        operations_data_class=operations_class,
        verification_notes=notes,
    )


def verified_manifest_for(run: AnalysisResponse) -> VerifiedDemoResponse | None:
    """Run the verification gates against one specific run.

    Exposed so a client can ask "is THIS run verified" instead of asserting it
    from a query parameter it supplied itself.
    """

    return _verified_manifest(run, saved_at=datetime.now(timezone.utc))


# Every candidate row is parsed into a full response object, charts and GeoJSON
# included, on each call to a public endpoint. Five hundred of those per request
# was a self-inflicted load; the newest hundred is more than enough to find the
# most recent verified run.
VERIFIED_DEMO_SCAN_LIMIT = 100


def latest_verified_demo(repository: RunRepository) -> VerifiedDemoResponse | None:
    candidates: list[VerifiedDemoResponse] = []
    for saved_at, run in repository.list_recent(limit=VERIFIED_DEMO_SCAN_LIMIT):
        manifest = _verified_manifest(run, saved_at=_saved_at(saved_at))
        if manifest is not None:
            candidates.append(manifest)
    if not candidates:
        return None
    return next(
        (candidate for candidate in candidates if candidate.observed_heatmap),
        candidates[0],
    )
