from __future__ import annotations

from typing import Callable
from uuid import uuid4

from .agents import (
    AssetModelingAgent,
    DecisionAgent,
    EvidenceAndSafetyAgent,
    InvestmentAnalystAgent,
    PlanningAgent,
    TemperatureIntelligenceAgent,
    UrbanChangeAgent,
)
from .context import RunContext
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    ConfidenceTier,
    RunStatus,
    SafetyVerdict,
    TraceEvent,
)
from .providers import build_thermal_provider, build_urban_provider
from .providers.fixture import ThermalDataProvider
from .providers.urban import UrbanContextProvider
from .telemetry import TelemetryStore


class FortyCoolOrchestrator:
    def __init__(
        self,
        provider: ThermalDataProvider | None = None,
        urban_provider: UrbanContextProvider | None = None,
        telemetry_store: TelemetryStore | None = None,
    ) -> None:
        self.provider = provider or build_thermal_provider()
        self.urban_provider = urban_provider or build_urban_provider(
            thermal_provider=self.provider
        )
        self.telemetry_store = telemetry_store or TelemetryStore()
        self.planner = PlanningAgent()
        self.urban_agent = UrbanChangeAgent(self.urban_provider)
        self.temperature_agent = TemperatureIntelligenceAgent(self.provider)
        self.asset_agent = AssetModelingAgent(self.provider, self.telemetry_store)
        self.decision_agent = DecisionAgent()
        self.investment_agent = InvestmentAnalystAgent()
        self.audit_agent = EvidenceAndSafetyAgent()

    async def run(
        self,
        request: AnalysisRequest,
        *,
        run_id: str | None = None,
        event_sink: Callable[[TraceEvent], None] | None = None,
    ) -> AnalysisResponse:
        context = RunContext(
            run_id=run_id or uuid4().hex,
            request=request.model_copy(deep=True),
            event_sink=event_sink,
        )
        self.planner.plan(context)
        await self.urban_agent.run(context)
        await self.temperature_agent.run(context)
        await self.asset_agent.run(context)
        await self.decision_agent.run(context)
        await self.investment_agent.run(context)
        await self.audit_agent.run(context)
        return self._response(context)

    @staticmethod
    def _response(context: RunContext) -> AnalysisResponse:
        evidence_failure = any(
            "Evidence verification failed" in item for item in context.warnings
        )
        incomplete = any(
            recommendation.verdict == SafetyVerdict.INSUFFICIENT_DATA
            for recommendation in context.recommendations
        )
        if evidence_failure:
            status = RunStatus.FAILED
        elif incomplete and not context.metrics:
            status = RunStatus.NEEDS_INPUT
        elif context.warnings or context.assumptions:
            status = RunStatus.COMPLETED_WITH_WARNINGS
        else:
            status = RunStatus.COMPLETED

        contains_simulated_evidence = any(
            item.data_class.value == "simulated" for item in context.evidence
        )
        if contains_simulated_evidence:
            tier = ConfidenceTier.INDICATIVE
        elif context.assumptions:
            tier = ConfidenceTier.SCREENING
        else:
            tier = ConfidenceTier.OPERATIONAL

        metric_map = {metric.id: metric for metric in context.metrics}
        drift = metric_map.get("local_thermal_drift_c")
        built_change = metric_map.get("built_surface_change_percentage_points")
        local_built_change = metric_map.get(
            "local_excess_built_surface_change_percentage_points"
        )
        land_cover_association = metric_map.get(
            "thermal_land_cover_association_correlation"
        )
        npv = metric_map.get("thermal_drift_npv")
        recommendation = next(
            (
                item
                for item in context.recommendations
                if item.action != "hold_current_settings"
            ),
            None,
        )
        summary_parts: list[str] = []
        if drift:
            summary_parts.append(
                f"Local thermal drift is {drift.value}{drift.unit} versus control"
            )
        if local_built_change:
            summary_parts.append(
                "local built surface changed by "
                f"{local_built_change.value} {local_built_change.unit} versus controls"
            )
        elif built_change:
            summary_parts.append(
                f"built surface changed by {built_change.value} {built_change.unit}"
            )
        if land_cover_association:
            summary_parts.append(
                "thermal/land-cover association is "
                f"{land_cover_association.value} {land_cover_association.unit}"
            )
        if npv:
            summary_parts.append(f"indicative exposure is {npv.value} {npv.unit}")
        if recommendation:
            summary_parts.append(
                f"the advisory 12-hour plan saves {recommendation.expected_savings_kwh:.1f} kWh"
            )
        if not summary_parts:
            summary_parts.append(
                "The requested workflow did not produce a decision-ready result"
            )
        summary = "; ".join(summary_parts) + "."

        return AnalysisResponse(
            run_id=context.run_id,
            status=status,
            confidence_tier=tier,
            summary=summary,
            metrics=context.metrics,
            recommendations=context.recommendations,
            charts=context.charts,
            evidence=context.evidence,
            trace=context.trace,
            assumptions=context.assumptions,
            warnings=context.warnings,
        )
