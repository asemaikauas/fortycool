from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import (
    AnalysisRequest,
    Assumption,
    Chart,
    EvidenceRef,
    Metric,
    Recommendation,
    TraceEvent,
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
    artifacts: dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: EvidenceRef) -> str:
        if not any(item.id == evidence.id for item in self.evidence):
            self.evidence.append(evidence)
        return evidence.id

    def event(
        self,
        agent: str,
        action: str,
        *,
        status: str = "completed",
        evidence_ids: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            TraceEvent(
                agent=agent,
                action=action,
                status=status,
                evidence_ids=evidence_ids or [],
                details=details or {},
            )
        )
