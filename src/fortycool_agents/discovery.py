from __future__ import annotations

import asyncio
import calendar
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import fmean
from typing import Any
from uuid import uuid4

from .discovery_catalog import public_catalog
from .models import (
    Chart,
    DataClass,
    DiscoveryCandidateResult,
    DiscoveryRequest,
    DiscoveryResponse,
    DiscoveryStatus,
    EvidenceRef,
    PublicSiteCandidate,
    TraceEvent,
)
from .providers.fixture import (
    AnnualThermalDataset,
    FixtureThermalProvider,
    ThermalDataProvider,
)
from .providers.live import FortyGuardThermalProvider
from .providers.urban import UrbanContextProvider


@dataclass(frozen=True)
class ScreeningObservation:
    candidate: PublicSiteCandidate
    baseline_site_temperature_c: float
    baseline_control_temperature_c: float
    latest_site_temperature_c: float
    latest_control_temperature_c: float
    local_drift_c: float
    data_class: DataClass
    source: str
    activity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class SiteDiscoveryAgent:
    """Two-stage discovery: cheap thermal screen, then deep validation."""

    def __init__(
        self,
        thermal_provider: ThermalDataProvider,
        urban_provider: UrbanContextProvider,
    ) -> None:
        self.thermal_provider = thermal_provider
        self.urban_provider = urban_provider
        concurrency = getattr(thermal_provider, "max_concurrency", 3)
        self._screen_semaphore = asyncio.Semaphore(concurrency)

    async def _screen_live(
        self,
        candidate: PublicSiteCandidate,
        baseline_year: int,
        end_year: int,
    ) -> ScreeningObservation:
        provider = self.thermal_provider
        if not isinstance(provider, FortyGuardThermalProvider):
            raise TypeError("live screening requires FortyGuardThermalProvider")

        async def retrieve(year: int) -> tuple[str | None, Any]:
            days = calendar.monthrange(year, provider.annual_reference_month)[1]
            start = datetime(
                year, provider.annual_reference_month, 1, tzinfo=timezone.utc
            )
            end = datetime(
                year, provider.annual_reference_month, days, tzinfo=timezone.utc
            )
            payload = provider._heatmap_payload(
                candidate.site,
                start=start,
                end=end,
                filter_type=4,
                analytic_type="tcm",
            )
            async with self._screen_semaphore:
                response = await provider.client.create_heatmap(payload)
            activity_id, _, features = provider._result(response)
            aggregate = provider._spatial_aggregate(
                features, candidate.site, "average_temperature"
            )
            return activity_id, aggregate

        baseline, latest = await asyncio.gather(
            retrieve(baseline_year), retrieve(end_year)
        )
        baseline_id, baseline_values = baseline
        latest_id, latest_values = latest
        baseline_gap = baseline_values.site_value - baseline_values.control_value
        latest_gap = latest_values.site_value - latest_values.control_value
        return ScreeningObservation(
            candidate=candidate,
            baseline_site_temperature_c=round(baseline_values.site_value, 4),
            baseline_control_temperature_c=round(baseline_values.control_value, 4),
            latest_site_temperature_c=round(latest_values.site_value, 4),
            latest_control_temperature_c=round(latest_values.control_value, 4),
            local_drift_c=round(latest_gap - baseline_gap, 4),
            data_class=DataClass.INFERRED,
            source=provider.source,
            activity_ids=[
                activity_id
                for activity_id in (baseline_id, latest_id)
                if activity_id is not None
            ],
            metadata={
                "analysis_window": calendar.month_name[provider.annual_reference_month],
                "control_method": "local_outer_ring",
                "site_tile_counts": [
                    baseline_values.site_tile_count,
                    latest_values.site_tile_count,
                ],
                "control_tile_counts": [
                    baseline_values.control_tile_count,
                    latest_values.control_tile_count,
                ],
            },
        )

    @staticmethod
    def _fixture_drift(candidate: PublicSiteCandidate, seed: int) -> float:
        demo_values = {
            "digital-realty-acc5": 0.28,
            "digital-realty-iad24": 0.16,
            "equinix-dc14": 0.04,
            "digital-realty-va3": -0.02,
        }
        if candidate.id in demo_values:
            return demo_values[candidate.id]
        digest = hashlib.sha256(f"{candidate.id}:{seed}".encode()).digest()
        return round((int.from_bytes(digest[:2], "big") / 65535 - 0.35) * 0.35, 4)

    def _screen_fixture(
        self,
        candidate: PublicSiteCandidate,
        baseline_year: int,
        end_year: int,
        seed: int,
    ) -> ScreeningObservation:
        drift = self._fixture_drift(candidate, seed)
        baseline_control = 26.0
        baseline_site = 26.4
        regional_change = 0.18 * (end_year - baseline_year)
        latest_control = baseline_control + regional_change
        latest_site = baseline_site + regional_change + drift
        return ScreeningObservation(
            candidate=candidate,
            baseline_site_temperature_c=round(baseline_site, 4),
            baseline_control_temperature_c=round(baseline_control, 4),
            latest_site_temperature_c=round(latest_site, 4),
            latest_control_temperature_c=round(latest_control, 4),
            local_drift_c=drift,
            data_class=DataClass.SIMULATED,
            source="fixture://site-discovery-screen/v1",
            activity_ids=[f"fixture-discovery-{candidate.id}-{seed}"],
            metadata={
                "analysis_window": "July",
                "control_method": "fixture_local_outer_ring",
            },
        )

    async def _screen(
        self,
        candidate: PublicSiteCandidate,
        request: DiscoveryRequest,
    ) -> ScreeningObservation:
        if isinstance(self.thermal_provider, FortyGuardThermalProvider):
            return await self._screen_live(
                candidate, request.baseline_year, request.end_year
            )
        return self._screen_fixture(
            candidate, request.baseline_year, request.end_year, request.seed
        )

    @staticmethod
    def _validated_drift(annual: AnnualThermalDataset) -> float:
        if len(annual.summaries) < 2:
            raise ValueError("full validation requires at least two years")
        baseline = annual.summaries[0]
        latest = annual.summaries[-1]
        return round(
            (latest.site_mean_temperature_c - baseline.site_mean_temperature_c)
            - (latest.control_mean_temperature_c - baseline.control_mean_temperature_c),
            4,
        )

    async def run(self, request: DiscoveryRequest) -> DiscoveryResponse:
        discovery_id = uuid4().hex
        candidates = request.candidates or public_catalog()
        evidence: list[EvidenceRef] = []
        trace = [
            TraceEvent(
                agent="site_discovery_agent",
                action=(
                    f"Planned a two-stage discovery across {len(candidates)} public sites"
                ),
                details={
                    "minimum_local_drift_c": request.minimum_local_drift_c,
                    "minimum_control_match_score": request.minimum_control_match_score,
                    "shortlist_size": request.shortlist_size,
                },
            )
        ]
        warnings: list[str] = []
        observations = await asyncio.gather(
            *(self._screen(candidate, request) for candidate in candidates),
            return_exceptions=True,
        )
        results: list[DiscoveryCandidateResult] = []
        for candidate, observation in zip(candidates, observations):
            catalog_evidence_id = f"catalog-{candidate.id}-{discovery_id}"
            evidence.append(
                EvidenceRef(
                    id=catalog_evidence_id,
                    source=candidate.operator_source_url,
                    description=(f"Public operator listing for {candidate.site.name}"),
                    data_class=DataClass.REPORTED,
                    metadata={
                        "operator": candidate.operator,
                        "address": candidate.address,
                        "latitude": candidate.site.latitude,
                        "longitude": candidate.site.longitude,
                        "coordinate_source": candidate.coordinate_source,
                    },
                )
            )
            if isinstance(observation, Exception):
                results.append(
                    DiscoveryCandidateResult(
                        candidate=candidate,
                        screening_local_drift_c=0.0,
                        screening_data_class=DataClass.INFERRED,
                        passed_drift_screen=False,
                        rejection_reasons=["thermal_screen_failed"],
                        evidence_ids=[catalog_evidence_id],
                    )
                )
                warnings.append(
                    f"{candidate.site.name} screen failed: {type(observation).__name__}"
                )
                continue
            screen_evidence_id = f"screen-{candidate.id}-{discovery_id}"
            evidence.append(
                EvidenceRef(
                    id=screen_evidence_id,
                    source=observation.source,
                    description=(
                        f"Baseline-versus-latest thermal screen for {candidate.site.name}"
                    ),
                    data_class=observation.data_class,
                    activity_id=(
                        observation.activity_ids[0]
                        if observation.activity_ids
                        else None
                    ),
                    endpoint=(
                        "/v1/heatmap"
                        if isinstance(self.thermal_provider, FortyGuardThermalProvider)
                        else None
                    ),
                    metadata={
                        "activity_ids": observation.activity_ids,
                        "baseline_year": request.baseline_year,
                        "end_year": request.end_year,
                        "baseline_site_temperature_c": observation.baseline_site_temperature_c,
                        "baseline_control_temperature_c": observation.baseline_control_temperature_c,
                        "latest_site_temperature_c": observation.latest_site_temperature_c,
                        "latest_control_temperature_c": observation.latest_control_temperature_c,
                        "formula": (
                            "(site_latest-site_baseline)-"
                            "(control_latest-control_baseline)"
                        ),
                        **observation.metadata,
                    },
                )
            )
            passed = observation.local_drift_c >= request.minimum_local_drift_c
            results.append(
                DiscoveryCandidateResult(
                    candidate=candidate,
                    screening_local_drift_c=observation.local_drift_c,
                    screening_data_class=observation.data_class,
                    passed_drift_screen=passed,
                    rejection_reasons=([] if passed else ["below_drift_threshold"]),
                    evidence_ids=[catalog_evidence_id, screen_evidence_id],
                )
            )

        results.sort(key=lambda item: item.screening_local_drift_c, reverse=True)
        shortlist = [item for item in results if item.passed_drift_screen][
            : request.shortlist_size
        ]
        shortlisted_ids = {item.candidate.id for item in shortlist}
        for result in results:
            if (
                result.passed_drift_screen
                and result.candidate.id not in shortlisted_ids
            ):
                result.rejection_reasons.append("not_shortlisted_for_deep_validation")

        trace.append(
            TraceEvent(
                agent="site_discovery_agent",
                action=(
                    f"Screened {len(results)} sites and shortlisted {len(shortlist)}"
                ),
                evidence_ids=[
                    evidence_id
                    for result in results
                    for evidence_id in result.evidence_ids
                    if evidence_id.startswith("screen-")
                ],
                details={
                    "ranking": [
                        {
                            "candidate_id": result.candidate.id,
                            "local_drift_c": result.screening_local_drift_c,
                            "passed": result.passed_drift_screen,
                        }
                        for result in results
                    ]
                },
            )
        )

        for result in shortlist:
            candidate = result.candidate
            try:
                urban = await self.urban_provider.analyze(
                    candidate.site,
                    request.baseline_year,
                    request.end_year,
                    seed=request.seed,
                )
                warnings.extend(urban.warnings)
                minimum_similarity = float(
                    urban.metadata.get("minimum_selected_similarity", 0.0)
                )
                result.control_match_score = round(
                    fmean(
                        control.similarity_score for control in urban.matched_controls
                    ),
                    4,
                )
                controls_passed = bool(
                    len(urban.matched_controls) >= 3
                    and minimum_similarity >= request.minimum_control_match_score
                )
                result.control_match_status = (
                    "accepted" if controls_passed else "rejected"
                )
                satellite_evidence_id = f"satellite-{candidate.id}-{discovery_id}"
                evidence.append(
                    EvidenceRef(
                        id=satellite_evidence_id,
                        source=urban.source,
                        description=(
                            f"Satellite control-quality validation for {candidate.site.name}"
                        ),
                        data_class=urban.data_class,
                        activity_id=(
                            urban.activity_ids[0] if urban.activity_ids else None
                        ),
                        endpoint=(
                            "/v1/satellite"
                            if urban.data_class == DataClass.OBSERVED
                            else None
                        ),
                        metadata={
                            "activity_ids": urban.activity_ids,
                            "mean_similarity": result.control_match_score,
                            "minimum_similarity": minimum_similarity,
                            "required_minimum_similarity": (
                                request.minimum_control_match_score
                            ),
                            **urban.metadata,
                        },
                    )
                )
                result.evidence_ids.append(satellite_evidence_id)
                if not controls_passed:
                    result.rejection_reasons.append("control_quality_gate_failed")
                    trace.append(
                        TraceEvent(
                            agent="site_discovery_agent",
                            action=(
                                f"Rejected {candidate.site.name} after control-quality validation"
                            ),
                            status="rejected",
                            evidence_ids=[satellite_evidence_id],
                        )
                    )
                    continue

                controls = [control.site for control in urban.matched_controls]
                annual = await self.thermal_provider.annual_history(
                    candidate.site,
                    request.baseline_year,
                    request.end_year,
                    request.temperature_eligibility_threshold_c,
                    seed=request.seed,
                    controls=controls,
                )
                warnings.extend(annual.warnings)
                result.validated_local_drift_c = self._validated_drift(annual)
                result.full_history_years = [item.year for item in annual.summaries]
                validation_evidence_id = f"validation-{candidate.id}-{discovery_id}"
                evidence.append(
                    EvidenceRef(
                        id=validation_evidence_id,
                        source=annual.source,
                        description=(
                            f"Full multi-year matched-control validation for {candidate.site.name}"
                        ),
                        data_class=annual.data_class,
                        activity_id=(
                            annual.activity_ids[0] if annual.activity_ids else None
                        ),
                        endpoint=(
                            "/v1/heatmap"
                            if annual.metadata.get("observed_years")
                            else None
                        ),
                        metadata={
                            "activity_ids": annual.activity_ids,
                            "years": result.full_history_years,
                            "validated_local_drift_c": result.validated_local_drift_c,
                            **annual.metadata,
                        },
                    )
                )
                result.evidence_ids.append(validation_evidence_id)
                backcast_years = annual.metadata.get("backcast_years", [])
                if backcast_years:
                    result.rejection_reasons.append("historical_coverage_gap")
                if result.validated_local_drift_c < request.minimum_local_drift_c:
                    result.rejection_reasons.append("full_validation_below_threshold")
                result.qualified = not result.rejection_reasons
                trace.append(
                    TraceEvent(
                        agent="site_discovery_agent",
                        action=(f"Completed full validation for {candidate.site.name}"),
                        status="qualified" if result.qualified else "rejected",
                        evidence_ids=[satellite_evidence_id, validation_evidence_id],
                        details={
                            "validated_local_drift_c": result.validated_local_drift_c,
                            "control_match_score": result.control_match_score,
                        },
                    )
                )
            except Exception as exc:
                result.control_match_status = "validation_failed"
                result.rejection_reasons.append("deep_validation_failed")
                warnings.append(
                    f"{candidate.site.name} deep validation failed: {type(exc).__name__}"
                )
                trace.append(
                    TraceEvent(
                        agent="site_discovery_agent",
                        action=f"Deep validation failed for {candidate.site.name}",
                        status="failed",
                        details={"error_type": type(exc).__name__},
                    )
                )

        winner = next((result for result in results if result.qualified), None)
        status = (
            DiscoveryStatus.QUALIFIED_CANDIDATE_FOUND
            if winner is not None
            else DiscoveryStatus.NO_QUALIFIED_CANDIDATE
        )
        if isinstance(self.thermal_provider, FixtureThermalProvider):
            warnings.append(
                "Discovery temperatures, drift, and deep validation are simulated in fixture mode."
            )
        else:
            warnings.append(
                "First-pass discovery uses July monthly tcm site-versus-local-ring differences; "
                "only shortlisted sites receive Satellite and full-history validation."
            )
        if winner is None:
            if not shortlist:
                summary = (
                    f"No site passed the initial {request.minimum_local_drift_c:.2f}°C "
                    "local-drift screen; deep validation was not run."
                )
            else:
                summary = (
                    "No shortlisted site passed both the regional-control quality gate "
                    "and full multi-year validation."
                )
        else:
            summary = (
                f"{winner.candidate.site.name} qualified with "
                f"{winner.validated_local_drift_c:.3f}°C validated local drift and a "
                f"{winner.control_match_score:.3f} control-match score."
            )
        chart_data = [
            {
                "candidate_id": result.candidate.id,
                "name": result.candidate.site.name,
                "operator": result.candidate.operator,
                "screening_local_drift_c": result.screening_local_drift_c,
                "minimum_local_drift_c": request.minimum_local_drift_c,
                "control_match_score": result.control_match_score,
                "validated_local_drift_c": result.validated_local_drift_c,
                "qualified": result.qualified,
                "rejection_reasons": result.rejection_reasons,
            }
            for result in results
        ]
        chart_data_class = (
            DataClass.SIMULATED
            if isinstance(self.thermal_provider, FixtureThermalProvider)
            else DataClass.INFERRED
        )
        evidence_ids = [item.id for item in evidence]
        charts = [
            Chart(
                id="site_discovery_ranking",
                title="ThermalDrift public-site discovery ranking",
                kind="bar",
                data=chart_data,
                data_class=chart_data_class,
                evidence_ids=evidence_ids,
            ),
            Chart(
                id="site_discovery_map",
                title="Public data-center discovery candidates",
                kind="point_map",
                data=[
                    {
                        "candidate_id": result.candidate.id,
                        "name": result.candidate.site.name,
                        "latitude": result.candidate.site.latitude,
                        "longitude": result.candidate.site.longitude,
                        "screening_local_drift_c": result.screening_local_drift_c,
                        "qualified": result.qualified,
                    }
                    for result in results
                ],
                data_class=chart_data_class,
                evidence_ids=evidence_ids,
            ),
        ]
        trace.append(
            TraceEvent(
                agent="site_discovery_agent",
                action=(f"Discovery finished with status {status.value}"),
                status="completed",
                evidence_ids=winner.evidence_ids if winner else [],
            )
        )
        return DiscoveryResponse(
            discovery_id=discovery_id,
            status=status,
            summary=summary,
            winner=winner,
            candidates=results,
            charts=charts,
            evidence=evidence,
            trace=trace,
            warnings=warnings,
            disclaimer=(
                "This is an independent screening analysis of public facility locations. "
                "It is not an operator performance claim, engineering assessment, or affiliation."
            ),
        )
