from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import (
    AnalysisRequest,
    Assumption,
    Chart,
    EvidenceRef,
    Metric,
    Recommendation,
    TraceEvent,
    WarningCode,
)


@dataclass
class RunContext:
    run_id: str
    request: AnalysisRequest
    metrics: list[Metric] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    degraded_stages: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    event_sink: Callable[[TraceEvent], None] | None = None

    def add_evidence(self, evidence: EvidenceRef) -> str:
        if not any(item.id == evidence.id for item in self.evidence):
            self.evidence.append(evidence)
        return evidence.id

    def warn(self, message: str, code: WarningCode | None = None) -> None:
        """Record a human-readable warning and, when it matters, a stable code.

        Callers that need to branch on a caveat read `warning_codes`; rewording
        a sentence must never change program behaviour.
        """

        self.warnings.append(message)
        if code is not None and code.value not in self.warning_codes:
            self.warning_codes.append(code.value)

    def mark_stage_degraded(self, stage: str) -> None:
        """Record that a pipeline stage did not produce its normal output."""

        if stage not in self.degraded_stages:
            self.degraded_stages.append(stage)

    def event(
        self,
        agent: str,
        action: str,
        *,
        status: str = "completed",
        evidence_ids: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            agent=agent,
            action=action,
            status=status,
            evidence_ids=evidence_ids or [],
            details=details or {},
        )
        self.trace.append(event)
        if self.event_sink is not None:
            self.event_sink(event)
