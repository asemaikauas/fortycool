from __future__ import annotations

import asyncio

from fortycool_agents.models import (
    AnalysisRequest,
    ConfidenceTier,
    RunStatus,
    SafetyVerdict,
)
from fortycool_agents.orchestrator import FortyCoolOrchestrator


def demo_request(**overrides) -> AnalysisRequest:
    payload = {
        "site": {
            "name": "Northern Virginia Demo Campus",
            "latitude": 39.01,
            "longitude": -77.46,
        },
        "analysis_modes": ["thermal_drift", "operations_12h", "investment"],
        "facility": {"it_capacity_mw": 24},
        "simulation": {"enabled": True, "seed": 42, "history_days": 60},
        "use_demo_defaults": True,
    }
    payload.update(overrides)
    return AnalysisRequest.model_validate(payload)


def run(request: AnalysisRequest):
    return asyncio.run(FortyCoolOrchestrator().run(request))


def test_demo_uses_verified_2022_baseline() -> None:
    assert demo_request().baseline_year == 2022


def test_full_demo_produces_evidence_backed_outputs() -> None:
    response = run(demo_request())

    assert response.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert response.confidence_tier == ConfidenceTier.INDICATIVE
    assert {chart.id for chart in response.charts} >= {
        "satellite_land_cover_context",
        "regional_control_map",
        "thermal_drift_timeseries",
        "historical_day_backtest",
        "baseline_vs_optimized",
    }
    assert {metric.id for metric in response.metrics} >= {
        "current_built_surface_share_percent",
        "built_surface_change_percentage_points",
        "regional_control_match_score",
        "local_thermal_drift_c",
        "forecast_savings_kwh",
        "thermal_drift_npv",
    }
    assert response.recommendations
    assert response.recommendations[0].verdict == SafetyVerdict.APPROVED_WITH_WARNING
    assert response.recommendations[0].expected_savings_kwh > 0

    evidence_ids = {item.id for item in response.evidence}
    assert all(metric.evidence_ids for metric in response.metrics)
    assert all(set(metric.evidence_ids) <= evidence_ids for metric in response.metrics)
    assert all(
        set(recommendation.evidence_ids) <= evidence_ids
        for recommendation in response.recommendations
    )
    assert not any("Evidence verification failed" in warning for warning in response.warnings)


def test_fixture_analysis_is_numerically_reproducible() -> None:
    first = run(demo_request())
    second = run(demo_request())

    first_metrics = {metric.id: metric.value for metric in first.metrics}
    second_metrics = {metric.id: metric.value for metric in second.metrics}
    assert first_metrics == second_metrics
    assert first.recommendations[0].expected_savings_kwh == second.recommendations[0].expected_savings_kwh


def test_missing_safety_constraints_blocks_action_without_demo_defaults() -> None:
    request = demo_request(
        analysis_modes=["operations_12h"],
        facility={
            "it_capacity_mw": 24,
            "typical_it_load_mw": 17,
            "reported_pue": 1.38,
        },
        economics={},
        constraints={},
        use_demo_defaults=False,
    )
    response = run(request)

    assert response.recommendations
    assert response.recommendations[0].verdict == SafetyVerdict.INSUFFICIENT_DATA
    assert response.recommendations[0].action == "hold_current_settings"
    assert any("constraints are incomplete" in warning for warning in response.warnings)


def test_planner_only_selects_requested_workflows() -> None:
    response = run(demo_request(analysis_modes=["thermal_drift"]))
    planner = response.trace[0]

    assert "calculate_thermal_drift" in planner.details["steps"]
    assert "retrieve_satellite_land_cover" in planner.details["steps"]
    assert "select_regional_controls" in planner.details["steps"]
    assert "simulate_or_ingest_bms" not in planner.details["steps"]
    assert not response.recommendations
