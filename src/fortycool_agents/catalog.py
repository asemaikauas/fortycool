from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    owner: str
    description: str
    deterministic: bool = True


TOOL_CATALOG = [
    ToolDefinition(
        name="get_site_thermal_history",
        owner="temperature_intelligence_agent",
        description="Retrieve annual site and matched-control thermal summaries.",
    ),
    ToolDefinition(
        name="get_temperature_forecast",
        owner="temperature_intelligence_agent",
        description="Retrieve the next 1-12 hours of thermal conditions.",
    ),
    ToolDefinition(
        name="simulate_bms",
        owner="asset_modeling_agent",
        description="Generate reproducible BMS telemetry for a configured data-center twin.",
    ),
    ToolDefinition(
        name="calculate_local_drift",
        owner="temperature_intelligence_agent",
        description="Calculate site-versus-control difference-in-differences.",
    ),
    ToolDefinition(
        name="predict_cooling_demand",
        owner="asset_modeling_agent",
        description="Train, backtest, and apply the cooling-demand model.",
    ),
    ToolDefinition(
        name="evaluate_operating_scenarios",
        owner="decision_agent",
        description="Enumerate candidate settings and rank the safe candidates.",
    ),
    ToolDefinition(
        name="review_recommendation_safety",
        owner="evidence_and_safety_agent",
        description="Verify constraints, confidence, and evidence before release.",
    ),
    ToolDefinition(
        name="calculate_financial_impact",
        owner="investment_analyst_agent",
        description="Translate thermal drift into energy cost and discounted exposure.",
    ),
    ToolDefinition(
        name="get_metric_evidence",
        owner="evidence_and_safety_agent",
        description="Resolve the lineage records supporting a metric or recommendation.",
    ),
]


def serialized_catalog() -> list[dict]:
    return [asdict(item) for item in TOOL_CATALOG]
