from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .context import RunContext
from .modeling import train_and_backtest
from .models import AnalysisMode, DataClass, EvidenceRef, TelemetrySource
from .providers.fixture import ThermalDataProvider
from .simulation import enrich_uploaded_bms, make_forecast_operating_frame, simulate_bms
from .telemetry import TelemetryStore
from .tools import (
    analyze_thermal_drift,
    apply_demo_defaults,
    calculate_investment_impact,
    optimize_operations,
    record_model_results,
    register_facility_assumptions,
)


class PlanningAgent:
    """Deterministic planner used until the team's LLM planner is connected."""

    def plan(self, context: RunContext) -> list[str]:
        modes = set(context.request.analysis_modes)
        plan = ["validate_request"]
        if AnalysisMode.THERMAL_DRIFT in modes or AnalysisMode.INVESTMENT in modes:
            plan.extend(["retrieve_annual_thermal_history", "calculate_thermal_drift"])
        if AnalysisMode.OPERATIONS_12H in modes:
            plan.extend(
                [
                    "retrieve_operational_weather",
                    "simulate_or_ingest_bms",
                    "train_and_backtest_models",
                    "evaluate_safe_actions",
                ]
            )
        if AnalysisMode.INVESTMENT in modes:
            plan.append("calculate_investment_impact")
        plan.append("verify_evidence")
        context.artifacts["plan"] = plan
        context.event(
            "planning_agent",
            f"Created a {len(plan)}-step execution plan",
            details={"steps": plan},
        )
        return plan


class TemperatureIntelligenceAgent:
    def __init__(self, provider: ThermalDataProvider) -> None:
        self.provider = provider

    async def run(self, context: RunContext) -> None:
        modes = set(context.request.analysis_modes)
        if not ({AnalysisMode.THERMAL_DRIFT, AnalysisMode.INVESTMENT} & modes):
            return
        annual = await self.provider.annual_history(
            context.request.site,
            context.request.baseline_year,
            2026,
            context.request.temperature_eligibility_threshold_c,
            seed=context.request.simulation.seed,
        )
        analyze_thermal_drift(context, annual)


class AssetModelingAgent:
    def __init__(self, provider: ThermalDataProvider, telemetry_store: TelemetryStore) -> None:
        self.provider = provider
        self.telemetry_store = telemetry_store

    async def run(self, context: RunContext) -> None:
        if AnalysisMode.OPERATIONS_12H not in context.request.analysis_modes:
            return
        if (
            context.request.telemetry.source == TelemetrySource.SIMULATED
            and not context.request.simulation.enabled
        ):
            context.warnings.append(
                "Uploaded/live BMS ingestion is not configured; enable simulation for this milestone"
            )
            context.event(
                "asset_modeling_agent",
                "Could not build operational model because no telemetry source was configured",
                status="blocked",
            )
            return

        register_facility_assumptions(context)
        uploaded = None
        if context.request.telemetry.source == TelemetrySource.UPLOADED:
            try:
                uploaded = self.telemetry_store.get(str(context.request.telemetry.upload_id))
            except KeyError:
                context.warnings.append("Uploaded telemetry was not found or expired")
                context.event(
                    "asset_modeling_agent",
                    "Could not load the requested telemetry upload",
                    status="blocked",
                )
                return
            reference_start = uploaded["timestamp"].min().to_pydatetime()
            reference_end = uploaded["timestamp"].max().to_pydatetime() + timedelta(hours=1)
        else:
            reference_end = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
            reference_start = reference_end - timedelta(days=context.request.simulation.history_days)
        history_thermal = await self.provider.history(
            context.request.site,
            reference_start,
            reference_end,
            seed=context.request.simulation.seed,
        )
        forecast_thermal = await self.provider.forecast(
            context.request.site,
            context.request.simulation.forecast_hours,
            seed=context.request.simulation.seed,
        )
        weather_evidence = context.add_evidence(
            EvidenceRef(
                id=f"operational-weather-{context.run_id}",
                source=history_thermal.source,
                description="Historical and forecast thermal inputs for the digital twin",
                data_class=history_thermal.data_class,
                activity_id=history_thermal.activity_ids[0],
                endpoint="/v1/heatmap",
                metadata={
                    "history_hours": len(history_thermal.samples),
                    "forecast_hours": len(forecast_thermal.samples),
                },
            )
        )
        if uploaded is None:
            history = simulate_bms(
                history_thermal, context.request.facility, seed=context.request.simulation.seed + 10
            )
            telemetry_data_class = DataClass.SIMULATED
            bms_source = "fortycool://simulator/data-center-digital-twin/v1"
            bms_description = "Reproducible simulated BMS telemetry driven by thermal inputs"
        else:
            history, enrichment_warnings = enrich_uploaded_bms(
                uploaded, history_thermal, context.request.facility
            )
            context.warnings.extend(enrichment_warnings)
            telemetry_data_class = DataClass.UPLOADED
            bms_source = f"upload://{context.request.telemetry.upload_id}"
            bms_description = "Validated user-uploaded BMS telemetry enriched with thermal inputs"
        forecast = make_forecast_operating_frame(
            forecast_thermal,
            context.request.facility,
            history,
            seed=context.request.simulation.seed + 20,
        )
        if uploaded is not None:
            recent_load = history["it_load_kw"].tail(len(forecast)).to_numpy()
            if len(recent_load) == len(forecast):
                forecast["it_load_kw"] = recent_load
        bms_evidence = context.add_evidence(
            EvidenceRef(
                id=f"bms-telemetry-{context.run_id}",
                source=bms_source,
                description=bms_description,
                data_class=telemetry_data_class,
                metadata={
                    "seed": context.request.simulation.seed,
                    "rows": len(history),
                    "facility_archetype": context.request.facility.archetype.value,
                },
            )
        )
        context.event(
            "asset_modeling_agent",
            f"Prepared {len(history)} BMS observations for modeling",
            evidence_ids=[weather_evidence, bms_evidence],
        )
        bundle = train_and_backtest(history)
        record_model_results(context, bundle, data_class=telemetry_data_class)
        context.artifacts["model_bundle"] = bundle
        context.artifacts["forecast_frame"] = forecast


class DecisionAgent:
    async def run(self, context: RunContext) -> None:
        if AnalysisMode.OPERATIONS_12H not in context.request.analysis_modes:
            return
        bundle = context.artifacts.get("model_bundle")
        forecast = context.artifacts.get("forecast_frame")
        if bundle is None or forecast is None:
            return
        constraints = apply_demo_defaults(context)
        optimize_operations(context, bundle, forecast, constraints)


class InvestmentAnalystAgent:
    async def run(self, context: RunContext) -> None:
        if AnalysisMode.INVESTMENT not in context.request.analysis_modes:
            return
        calculate_investment_impact(context)


class EvidenceAndSafetyAgent:
    async def run(self, context: RunContext) -> None:
        evidence_ids = {item.id for item in context.evidence}
        missing: list[str] = []
        for metric in context.metrics:
            if not metric.evidence_ids:
                missing.append(metric.id)
            elif any(item not in evidence_ids for item in metric.evidence_ids):
                missing.append(metric.id)
        for recommendation in context.recommendations:
            if recommendation.action != "hold_current_settings" and not recommendation.evidence_ids:
                missing.append(recommendation.id)
            elif any(item not in evidence_ids for item in recommendation.evidence_ids):
                missing.append(recommendation.id)
        if missing:
            context.warnings.append(
                "Evidence verification failed for: " + ", ".join(sorted(set(missing)))
            )
            context.event(
                "evidence_and_safety_agent",
                "Found outputs with missing or unresolved evidence",
                status="blocked",
                details={"output_ids": sorted(set(missing))},
            )
        else:
            context.event(
                "evidence_and_safety_agent",
                f"Verified evidence lineage for {len(context.metrics)} metrics and "
                f"{len(context.recommendations)} recommendations",
            )
