from __future__ import annotations

import math
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# The analysis window used to end at a literal 2026 in three separate places, so
# in 2027 the "latest" year would silently stay 2026 forever and a newer
# baseline could not even be requested. The ceiling now follows the calendar.
EARLIEST_ANALYSIS_YEAR = 2019
SCHEMA_MAX_ANALYSIS_YEAR = 2100


def latest_analysis_year() -> int:
    """The newest calendar year an analysis window may reach."""

    return datetime.now(timezone.utc).year


# Retained for callers that imported the old name; it now tracks the calendar.
MAX_ANALYSIS_YEAR = latest_analysis_year()

# Guard rail for numbers that reach a report, a chart, or the copilot. A value
# outside this band is a corrupt input rather than a facility measurement, and
# persisting one poisons every later read of the run.
MAX_REPORTABLE_MAGNITUDE = 1e12


def reject_non_finite(value: Any, field_name: str) -> Any:
    """Reject NaN, infinities, and absurd magnitudes on a reported number."""

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number")
        if abs(value) > MAX_REPORTABLE_MAGNITUDE:
            raise ValueError(
                f"{field_name} exceeds the reportable magnitude limit "
                f"({MAX_REPORTABLE_MAGNITUDE:g})"
            )
    return value


def reject_control_characters(value: str, field_name: str) -> str:
    """Reject free text that could render as something else downstream.

    Bidi overrides and C0/C1 controls are not typographic detail here: they are
    the cheapest way to make one string display as another inside a PDF or a
    model prompt, and none of them belong in a facility name.
    """

    for character in value:
        if character in {" ", "\t"}:
            continue
        if unicodedata.category(character) in {"Cc", "Cf", "Co", "Cs", "Cn"}:
            raise ValueError(
                f"{field_name} may not contain control or formatting characters"
            )
    return value


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


class EvidenceGrade(str, Enum):
    """How well a reported number is backed, on an ordinal scale.

    A: measured by a named external source over the whole analysis window.
    B: derived or partially measured - real inputs, modelled combination.
    C: simulated, assumed, or without a sampling-error estimate.
    """

    A = "A"
    B = "B"
    C = "C"


class WarningCode(str, Enum):
    """Stable identifiers for caveats that consumers need to branch on.

    Run status and warning suppression used to be decided by searching for
    substrings inside human-readable warnings, so rewording a sentence changed
    program behaviour. These codes carry the meaning; the sentences stay for
    people to read.
    """

    EVIDENCE_VERIFICATION_FAILED = "evidence_verification_failed"
    CONTROL_STABILITY_UNVERIFIED = "control_stability_unverified"
    THERMAL_SERIES_BACKCAST = "thermal_series_backcast"
    THERMAL_SERIES_SIMULATED = "thermal_series_simulated"
    FIXTURE_SERIES_LOCATION_INDEPENDENT = "fixture_series_location_independent"
    SATELLITE_STAGE_UNAVAILABLE = "satellite_stage_unavailable"
    TELEMETRY_INTERPOLATED = "telemetry_interpolated"
    TELEMETRY_NOT_FOUND = "telemetry_not_found"
    SPATIAL_FALLBACK_ZONES = "spatial_fallback_zones"
    DRIFT_NOT_DISTINGUISHABLE = "drift_not_distinguishable_from_zero"
    ATTRIBUTION_WITHHELD = "attribution_withheld"
    CIRCULAR_SENSITIVITY_CONSTANT = "circular_sensitivity_constant"


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


# A facility AOI is a building footprint plus a control ring, not a region. The
# caps below are what a 60 m heatmap request can reasonably cover; anything
# larger is a mistake or an attempt to make one request cost a hundred.
MAX_AOI_VERTICES = 2_000
MAX_AOI_SPAN_DEGREES = 1.0


def _aoi_coordinates(geometry: Any) -> list[tuple[float, float]]:
    """Flatten every coordinate pair reachable from a GeoJSON geometry."""

    points: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and all(isinstance(item, (int, float)) for item in node[:2])
                and not isinstance(node[0], bool)
            ):
                points.append((float(node[0]), float(node[1])))
                return
            for item in node:
                walk(item)

    walk(geometry)
    return points


class SiteInput(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    aoi: dict[str, Any] | None = None
    timezone: str = "America/New_York"

    @field_validator("name")
    @classmethod
    def name_must_be_printable(cls, value: str) -> str:
        return reject_control_characters(value, "site.name")

    @field_validator("aoi")
    @classmethod
    def aoi_must_be_a_bounded_polygon(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Validate the caller-supplied AOI at the door rather than upstream.

        An unrecognised shape used to surface as a 500 from inside the provider,
        and an unbounded polygon became a planet-scale 60 m heatmap request.
        Both are request errors, so they belong here as a 422.
        """

        if value is None:
            return value
        aoi_type = value.get("type")
        if aoi_type not in {"FeatureCollection", "Feature", "Polygon"}:
            raise ValueError(
                "site.aoi must be a GeoJSON Polygon, Feature, or FeatureCollection"
            )
        points = _aoi_coordinates(value)
        if not points:
            raise ValueError("site.aoi did not contain any coordinates")
        if len(points) > MAX_AOI_VERTICES:
            raise ValueError(
                f"site.aoi may not contain more than {MAX_AOI_VERTICES:,} vertices"
            )
        longitudes = [point[0] for point in points]
        latitudes = [point[1] for point in points]
        if not all(-180 <= longitude <= 180 for longitude in longitudes) or not all(
            -90 <= latitude <= 90 for latitude in latitudes
        ):
            raise ValueError("site.aoi coordinates fall outside valid WGS84 bounds")
        span = max(
            max(longitudes) - min(longitudes), max(latitudes) - min(latitudes)
        )
        if span > MAX_AOI_SPAN_DEGREES:
            raise ValueError(
                "site.aoi bounding box may not span more than "
                f"{MAX_AOI_SPAN_DEGREES} degrees; supply a facility-scale area"
            )
        return value


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
    # Cooling-energy response to one degree of sustained local warming, as a
    # fraction of the cooling-plant load. It is an operator input, not a
    # property the analysis discovers, so it is stated here and recorded as an
    # assumption when it falls back to the default.
    temperature_sensitivity_per_c: float | None = Field(
        default=None, ge=0, le=0.5
    )


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
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
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
    baseline_year: int = Field(
        default=2022, ge=EARLIEST_ANALYSIS_YEAR, le=SCHEMA_MAX_ANALYSIS_YEAR
    )
    temperature_eligibility_threshold_c: float = Field(default=18.0, ge=-30, le=50)
    use_demo_defaults: bool = True

    @field_validator("analysis_modes")
    @classmethod
    def modes_cannot_be_empty(cls, value: list[AnalysisMode]) -> list[AnalysisMode]:
        if not value:
            raise ValueError("at least one analysis mode is required")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def baseline_year_must_leave_a_window(self) -> "AnalysisRequest":
        """A baseline equal to the latest year leaves a one-point series.

        The drift estimator needs at least two annual observations, so a
        baseline at the end of the window is a request error and must be
        rejected at the door rather than raising from inside the agents.
        """

        latest = latest_analysis_year()
        if self.baseline_year >= latest:
            raise ValueError(
                "baseline_year must be earlier than the latest analysis year "
                f"({latest}) so the drift series has at least two points"
            )
        return self


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
    # Ordinal grade shown to users. `confidence` is a hand-set internal score
    # and was being rendered as a percentage next to money, which reads as a
    # probability it never was.
    evidence_grade: EvidenceGrade | None = None
    # Two-sided 95% interval where the estimator supports one. Absent means the
    # quantity has no sampling-error estimate, not that it is exact.
    interval_low: float | None = None
    interval_high: float | None = None

    @field_validator("value", "interval_low", "interval_high")
    @classmethod
    def value_must_be_finite(cls, value: Any) -> Any:
        return reject_non_finite(value, "metric value")


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

    @field_validator(
        "expected_savings_kwh",
        "expected_peak_reduction_kw",
        "predicted_max_inlet_temperature_c",
        "safety_margin_c",
    )
    @classmethod
    def numbers_must_be_finite(cls, value: float) -> float:
        return reject_non_finite(value, "recommendation value")


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
    # Machine-readable caveats. `warnings` stays a list of sentences for
    # people; consumers that need to branch read these instead of matching on
    # prose, and `degraded_stages` names any pipeline stage that did not run.
    warning_codes: list[str] = Field(default_factory=list)
    degraded_stages: list[str] = Field(default_factory=list)


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
    # The draft caps what the model may return. The response also carries the
    # run's own warnings and safety verdicts, appended by the service so they
    # cannot be dropped or softened, so it needs the wider bound.
    cautions: list[str] = Field(default_factory=list, max_length=40)
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
    baseline_year: int = Field(
        default=2022, ge=EARLIEST_ANALYSIS_YEAR, le=SCHEMA_MAX_ANALYSIS_YEAR
    )
    end_year: int = Field(
        default_factory=latest_analysis_year, ge=2020, le=SCHEMA_MAX_ANALYSIS_YEAR
    )
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
        latest = latest_analysis_year()
        if self.end_year > latest:
            raise ValueError(
                f"end_year cannot exceed the current calendar year ({latest})"
            )
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
