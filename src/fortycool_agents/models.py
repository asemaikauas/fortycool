from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisMode(str, Enum):
    THERMAL_DRIFT = "thermal_drift"
    OPERATIONS_12H = "operations_12h"
    INVESTMENT = "investment"


class DataClass(str, Enum):
    OBSERVED = "observed"
    UPLOADED = "uploaded"
    SIMULATED = "simulated"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    REPORTED = "reported"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"


class SafetyVerdict(str, Enum):
    APPROVED = "approved"
    APPROVED_WITH_WARNING = "approved_with_warning"
    HOLD = "hold_current_settings"
    INSUFFICIENT_DATA = "insufficient_data"


class ConfidenceTier(str, Enum):
    SCREENING = "screening"
    INDICATIVE = "indicative"
    OPERATIONAL = "operational"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FacilityArchetype(str, Enum):
    COLOCATION_WATER_COOLED = "colocation_water_cooled"
    HYPERSCALE_AIR_ECONOMIZED = "hyperscale_air_economized"
    HIGH_DENSITY_HYBRID = "high_density_hybrid"


class TelemetrySource(str, Enum):
    SIMULATED = "simulated"
    UPLOADED = "uploaded"


class CopilotAudience(str, Enum):
    OPERATOR = "operator"
    INVESTMENT_COMMITTEE = "investment_committee"
    TECHNICAL_REVIEWER = "technical_reviewer"


class DiscoveryStatus(str, Enum):
    QUALIFIED_CANDIDATE_FOUND = "qualified_candidate_found"
    NO_QUALIFIED_CANDIDATE = "no_qualified_candidate"


class SiteInput(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    aoi: dict[str, Any] | None = None
    timezone: str = "America/New_York"


class FacilityProfile(BaseModel):
    archetype: FacilityArchetype = FacilityArchetype.COLOCATION_WATER_COOLED
    it_capacity_mw: float | None = Field(default=None, gt=0, le=1000)
    typical_it_load_mw: float | None = Field(default=None, gt=0, le=1000)
    reported_pue: float | None = Field(default=None, ge=1.0, le=3.0)
    commissioning_year: int | None = Field(default=None, ge=1990, le=2100)
    redundancy: str | None = None

    @model_validator(mode="after")
    def validate_load(self) -> "FacilityProfile":
        if (
            self.it_capacity_mw is not None
            and self.typical_it_load_mw is not None
            and self.typical_it_load_mw > self.it_capacity_mw
        ):
            raise ValueError("typical_it_load_mw cannot exceed it_capacity_mw")
        return self


class EconomicsInput(BaseModel):
    electricity_price_per_kwh: float | None = Field(default=None, ge=0, le=5)
    demand_charge_per_kw_month: float | None = Field(default=None, ge=0, le=1000)
    horizon_years: int | None = Field(default=None, ge=1, le=40)
    discount_rate: float | None = Field(default=None, ge=0, le=0.5)
    annual_tariff_inflation: float | None = Field(default=None, ge=-0.1, le=0.5)


class SafetyConstraints(BaseModel):
    maximum_server_inlet_temperature_c: float | None = Field(default=None, ge=10, le=50)
    minimum_safety_margin_c: float | None = Field(default=None, ge=0, le=15)
    supply_air_setpoint_min_c: float | None = Field(default=None, ge=5, le=40)
    supply_air_setpoint_max_c: float | None = Field(default=None, ge=5, le=40)
    maximum_setpoint_change_c: float | None = Field(default=None, ge=0, le=10)
    minimum_model_confidence: float | None = Field(default=None, ge=0, le=1)
    forecast_max_age_minutes: int | None = Field(default=None, ge=1, le=1440)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SafetyConstraints":
        if (
            self.supply_air_setpoint_min_c is not None
            and self.supply_air_setpoint_max_c is not None
            and self.supply_air_setpoint_min_c > self.supply_air_setpoint_max_c
        ):
            raise ValueError("supply-air minimum cannot exceed maximum")
        return self


class SimulationConfig(BaseModel):
    enabled: bool = True
    seed: int = 42
    history_days: int = Field(default=60, ge=14, le=730)
    interval_minutes: int = Field(default=60, ge=15, le=60)
    forecast_hours: int = Field(default=12, ge=1, le=12)

    @field_validator("interval_minutes")
    @classmethod
    def interval_must_divide_hour(cls, value: int) -> int:
        if value != 60:
            raise ValueError("the first milestone currently supports hourly simulation only")
        return value


class TelemetryConfig(BaseModel):
    source: TelemetrySource = TelemetrySource.SIMULATED
    upload_id: str | None = None

    @model_validator(mode="after")
    def upload_requires_identifier(self) -> "TelemetryConfig":
        if self.source == TelemetrySource.UPLOADED and not self.upload_id:
            raise ValueError("telemetry.upload_id is required when source is uploaded")
        return self


class AnalysisRequest(BaseModel):
    site: SiteInput
    analysis_modes: list[AnalysisMode] = Field(
        default_factory=lambda: [AnalysisMode.THERMAL_DRIFT, AnalysisMode.OPERATIONS_12H]
    )
    facility: FacilityProfile = Field(default_factory=FacilityProfile)
    economics: EconomicsInput = Field(default_factory=EconomicsInput)
    constraints: SafetyConstraints = Field(default_factory=SafetyConstraints)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    baseline_year: int = Field(default=2022, ge=2019, le=2026)
    temperature_eligibility_threshold_c: float = Field(default=18.0, ge=-30, le=50)
    use_demo_defaults: bool = True

    @field_validator("analysis_modes")
    @classmethod
    def modes_cannot_be_empty(cls, value: list[AnalysisMode]) -> list[AnalysisMode]:
        if not value:
            raise ValueError("at least one analysis mode is required")
        return list(dict.fromkeys(value))


class EvidenceRef(BaseModel):
    id: str
    source: str
    description: str
    data_class: DataClass
    activity_id: str | None = None
    endpoint: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Metric(BaseModel):
    id: str
    label: str
    value: float | int | str
    unit: str
    confidence: float = Field(ge=0, le=1)
    data_class: DataClass
    evidence_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    id: str
    action: str
    description: str
    expected_savings_kwh: float = Field(ge=0)
    expected_peak_reduction_kw: float = Field(ge=0)
    predicted_max_inlet_temperature_c: float
    safety_margin_c: float
    confidence: float = Field(ge=0, le=1)
    verdict: SafetyVerdict
    constraints_checked: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Chart(BaseModel):
    id: str
    title: str
    kind: str
    data: list[dict[str, Any]]
    data_class: DataClass
    evidence_ids: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    action: str
    status: str = "completed"
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Assumption(BaseModel):
    field: str
    value: float | int | str | bool
    reason: str
    data_class: DataClass = DataClass.ASSUMED
    user_confirmed: bool = False


class AnalysisResponse(BaseModel):
    run_id: str
    status: RunStatus
    confidence_tier: ConfidenceTier
    summary: str
    metrics: list[Metric] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    charts: list[Chart] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VerifiedDemoResponse(BaseModel):
    run: AnalysisResponse
    saved_at: datetime
    verified_at: datetime
    thermal_years: list[int]
    observed_heatmap: bool
    operations_data_class: DataClass | None = None
    verification_notes: list[str] = Field(default_factory=list)


class TelemetryUpload(BaseModel):
    upload_id: str
    rows: int
    start_timestamp: datetime
    end_timestamp: datetime
    median_interval_minutes: float
    columns: list[str]
    warnings: list[str] = Field(default_factory=list)


class RunJobStatus(BaseModel):
    run_id: str
    state: JobState
    event_count: int = 0
    error: str | None = None
    response: AnalysisResponse | None = None


class CopilotRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    audience: CopilotAudience = CopilotAudience.OPERATOR

    @field_validator("question")
    @classmethod
    def question_cannot_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped


class CopilotDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=6000)
    key_findings: list[str] = Field(max_length=5)
    cautions: list[str] = Field(max_length=5)
    evidence_ids: list[str] = Field(max_length=20)
    suggested_questions: list[str] = Field(max_length=4)


class CopilotResponse(CopilotDraft):
    run_id: str
    model: str
    response_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PublicSiteCandidate(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    operator: str = Field(min_length=1, max_length=120)
    address: str = Field(min_length=1, max_length=240)
    site: SiteInput
    operator_source_url: str = Field(pattern=r"^https://")
    coordinate_source: str = Field(
        default="OpenStreetMap Nominatim geocode of the public operator address",
        min_length=1,
        max_length=240,
    )


class DiscoveryRequest(BaseModel):
    candidates: list[PublicSiteCandidate] | None = None
    baseline_year: int = Field(default=2022, ge=2019, le=2026)
    end_year: int = Field(default=2026, ge=2020, le=2026)
    minimum_local_drift_c: float = Field(default=0.10, ge=0, le=5)
    minimum_control_match_score: float = Field(default=0.65, ge=0, le=1)
    shortlist_size: int = Field(default=1, ge=1, le=3)
    require_historical_land_cover: bool = True
    temperature_eligibility_threshold_c: float = Field(default=18.0, ge=-30, le=50)
    seed: int = 42

    @model_validator(mode="after")
    def validate_discovery_scope(self) -> "DiscoveryRequest":
        if self.end_year <= self.baseline_year:
            raise ValueError("end_year must be later than baseline_year")
        if self.candidates is not None:
            if not 1 <= len(self.candidates) <= 12:
                raise ValueError("candidates must contain between 1 and 12 sites")
            candidate_ids = [candidate.id for candidate in self.candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("candidate IDs must be unique")
        return self


class DiscoveryCandidateResult(BaseModel):
    candidate: PublicSiteCandidate
    screening_local_drift_c: float
    screening_data_class: DataClass
    passed_drift_screen: bool
    control_match_score: float | None = Field(default=None, ge=0, le=1)
    control_match_status: str = "not_evaluated"
    historical_land_cover_status: str = "not_evaluated"
    validated_local_drift_c: float | None = None
    full_history_years: list[int] = Field(default_factory=list)
    qualified: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class DiscoveryResponse(BaseModel):
    discovery_id: str
    status: DiscoveryStatus
    summary: str
    winner: DiscoveryCandidateResult | None = None
    candidates: list[DiscoveryCandidateResult]
    charts: list[Chart] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
