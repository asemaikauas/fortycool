from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    owner: str
    description: str
    invocation_path: str | None = None
    deterministic: bool = True


TOOL_CATALOG = [
    ToolDefinition(
        name="analyze_thermal_drift_workflow",
        owner="planning_agent",
        description="Run the complete historical site-versus-control workflow.",
        invocation_path="/agent-tools/thermal-drift",
    ),
    ToolDefinition(
        name="recommend_12h_workflow",
        owner="planning_agent",
        description="Run telemetry modeling, backtesting, optimization, and safety review.",
        invocation_path="/agent-tools/operations-12h",
    ),
    ToolDefinition(
        name="analyze_investment_workflow",
        owner="planning_agent",
        description="Run ThermalDrift and translate the result into indicative financial exposure.",
        invocation_path="/agent-tools/investment",
    ),
    ToolDefinition(
        name="get_site_thermal_history",
        owner="temperature_intelligence_agent",
        description="Retrieve annual site and matched-control thermal summaries.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="get_temperature_forecast",
        owner="temperature_intelligence_agent",
        description="Retrieve the next 1-12 hours of thermal conditions.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="simulate_bms",
        owner="asset_modeling_agent",
        description="Generate reproducible BMS telemetry for a configured data-center twin.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="calculate_local_drift",
        owner="temperature_intelligence_agent",
        description="Calculate site-versus-control difference-in-differences.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="predict_cooling_demand",
        owner="asset_modeling_agent",
        description="Train, backtest, and apply the cooling-demand model.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="evaluate_operating_scenarios",
        owner="decision_agent",
        description="Enumerate candidate settings and rank the safe candidates.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="review_recommendation_safety",
        owner="evidence_and_safety_agent",
        description="Verify constraints, confidence, and evidence before release.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="calculate_financial_impact",
        owner="investment_analyst_agent",
        description="Translate thermal drift into energy cost and discounted exposure.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="get_metric_evidence",
        owner="evidence_and_safety_agent",
        description="Resolve the lineage records supporting a metric or recommendation.",
        invocation_path=None,
    ),
    ToolDefinition(
        name="explain_completed_analysis",
        owner="copilot_agent",
        description=(
            "Use GPT-4o to answer an operator or investor question from a completed, "
            "evidence-backed analysis."
        ),
        invocation_path="/runs/{run_id}/copilot",
        deterministic=False,
    ),
]


def serialized_catalog() -> list[dict]:
    return [asdict(item) for item in TOOL_CATALOG]
