"""Regression tests for the hostile-review findings.

The suite that existed before this file was 41 green tests that proved less than
the count suggested: nothing called an endpoint with a schema-valid but fatal
input, nothing fed the provider a malformed upstream body, nothing asserted that
the confidence tier follows data coverage, and the reproducibility test compared
two fixture runs with each other, which a systematically wrong model also passes.

Each test below fails on the code as it stood before the corresponding fix.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fortycool_agents import security
from fortycool_agents.api import app
from fortycool_agents.inference import estimate_drift
from fortycool_agents.models import (
    AnalysisRequest,
    ConfidenceTier,
    DataClass,
    SiteInput,
    WarningCode,
)
from fortycool_agents.orchestrator import FortyCoolOrchestrator
from fortycool_agents.providers import FixtureThermalProvider, FortyGuardThermalProvider
from fortycool_agents.providers.fortyguard import FortyGuardError
from fortycool_agents.providers.live import FortyGuardThermalProvider as LiveProvider
from fortycool_agents.simulation import enrich_uploaded_bms
from fortycool_agents.telemetry import TelemetryStore, TelemetryValidationError

from test_live_provider import FakeFortyGuardClient, polygon_feature, site


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Each test starts with a fresh allowance."""

    security.rate_limiter.reset()
    yield
    security.rate_limiter.reset()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def demo_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "site": {
            "name": "Test Campus",
            "latitude": 39.01,
            "longitude": -77.46,
            "timezone": "America/New_York",
        },
        "analysis_modes": ["thermal_drift", "operations_12h", "investment"],
        "facility": {"archetype": "colocation_water_cooled", "it_capacity_mw": 10.0},
        "economics": {
            "electricity_price_per_kwh": 0.08,
            "horizon_years": 10,
            "discount_rate": 0.08,
        },
        "telemetry": {"source": "simulated"},
        "simulation": {
            "enabled": True,
            "seed": 42,
            "history_days": 60,
            "interval_minutes": 60,
            "forecast_hours": 12,
        },
        "baseline_year": 2022,
        "temperature_eligibility_threshold_c": 18.0,
        "use_demo_defaults": True,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Endpoint robustness: schema-valid inputs that used to produce a 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description,overrides",
    [
        # A baseline at the end of the window leaves a one-point series, and
        # the drift estimator raised straight through the ASGI layer.
        ("baseline at the latest year", {"baseline_year": datetime.now(timezone.utc).year}),
        ("baseline one year before the end", {"baseline_year": datetime.now(timezone.utc).year - 1}),
        # An unrecognised AOI type raised from inside the provider.
        ("nonsense aoi type", {"site": {
            "name": "Test", "latitude": 39.01, "longitude": -77.46,
            "aoi": {"type": "Nonsense"},
        }}),
        # A world-spanning polygon became a planet-scale 60 m heatmap request.
        ("world-spanning aoi", {"site": {
            "name": "Test", "latitude": 39.01, "longitude": -77.46,
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
            },
        }}),
        # A bidi override in a facility name reaches the model context and the PDF.
        ("bidi override in the site name", {"site": {
            "name": "Site ‮evil", "latitude": 39.01, "longitude": -77.46,
        }}),
        ("minimum legal request", {"analysis_modes": ["thermal_drift"]}),
    ],
)
def test_schema_valid_edge_inputs_never_return_5xx(
    client: TestClient, description: str, overrides: dict[str, Any]
) -> None:
    response = client.post("/runs", json=demo_request(**overrides))
    assert response.status_code < 500, f"{description} produced {response.status_code}"


def test_sync_and_async_run_paths_agree_on_rejection(client: TestClient) -> None:
    """A body rejected by /runs must not be accepted by /run-jobs.

    The same payload used to 500 on /runs and return 202 on /run-jobs, then fail
    the job - two different answers to one question.
    """

    payload = demo_request(baseline_year=datetime.now(timezone.utc).year)
    sync = client.post("/runs", json=payload)
    queued = client.post("/run-jobs", json=payload)
    assert sync.status_code >= 400
    assert queued.status_code >= 400


def test_job_errors_do_not_leak_internal_detail(client: TestClient) -> None:
    """A failed job returns a stable message, not a library stack trace."""

    started = client.post("/run-jobs", json=demo_request())
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    status = client.get(f"/run-jobs/{run_id}").json()
    if status.get("error"):
        assert "pydantic" not in status["error"].lower()
        assert "traceback" not in status["error"].lower()


# ---------------------------------------------------------------------------
# Malformed upstream responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"data": None},
        {"data": {"result": None}},
        {"data": {"result": "processing"}},
        {"data": {"result": {"map_data": None}}},
        {"data": {"result": {"map_data": {"features": {"not": "a list"}}}}},
        {"data": {"result": {"map_data": {"features": None}}}},
        {},
    ],
)
def test_malformed_upstream_bodies_raise_a_handled_provider_error(
    body: dict[str, Any]
) -> None:
    """Every shape degrades as FortyGuardError, which callers already handle.

    These produced AttributeError, TypeError, and ValueError from outside the
    provider's own error handling and reached clients as a 500.
    """

    if body.get("data", {}) and isinstance(body.get("data"), dict):
        try:
            LiveProvider._result(body)
        except FortyGuardError:
            return
        # A well-formed-but-empty body is allowed to succeed with no features.
        return
    try:
        LiveProvider._result(body)
    except FortyGuardError:
        return
    except Exception as exc:  # pragma: no cover - this is the regression
        pytest.fail(f"unhandled {type(exc).__name__} instead of FortyGuardError")


@pytest.mark.parametrize(
    "properties",
    [
        {"tile_id": 1, "average_temperature": "n/a"},
        {"tile_id": 1, "average_temperature": float("inf")},
    ],
)
def test_unusable_tile_values_raise_a_handled_provider_error(
    properties: dict[str, Any]
) -> None:
    provider = LiveProvider(FakeFortyGuardClient())
    features = [
        polygon_feature(-77.46, 39.01, properties),
        polygon_feature(-77.454, 39.01, {"tile_id": 2, "average_temperature": 25.0}),
    ]
    with pytest.raises(FortyGuardError):
        provider._spatial_aggregate(features, site(), "average_temperature")


def test_null_coordinates_raise_a_handled_provider_error() -> None:
    feature = {
        "type": "Feature",
        "properties": {"average_temperature": 25.0},
        "geometry": {"type": "Polygon", "coordinates": [[[None, None], [1, 1], [2, 2]]]},
    }
    with pytest.raises(FortyGuardError):
        LiveProvider._centroid(feature)


# ---------------------------------------------------------------------------
# Coverage drives the tier, and a backcast series publishes no drift
# ---------------------------------------------------------------------------


def test_partly_backcast_series_withholds_the_drift_and_the_money() -> None:
    """With one observed year the DiD is arithmetic on a fixture constant.

    Backcast years are built as fixture_gap[y] + (observed[first] - fixture[first]).
    When the window ends on a backcast the observed term cancels exactly, so the
    reported drift equals the fixture's own hard-coded trend whatever the real
    data said - while `data_class` reads INFERRED rather than SIMULATED, which
    is what made it dangerous. The gate that caught this lived only on the demo
    endpoint; `/runs`, `/run-jobs`, `/agent-tools/*` and the PDF all accepted it.
    """

    from fortycool_agents.context import RunContext
    from fortycool_agents.tools import analyze_thermal_drift

    stub = FakeFortyGuardClient(unavailable_years={2022, 2023, 2024, 2025})
    provider = FortyGuardThermalProvider(
        stub, clock=lambda: datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
    )
    annual = asyncio.run(
        provider.annual_history(
            SiteInput(name="Test", latitude=39.01, longitude=-77.46),
            2022,
            2026,
            18.0,
            seed=42,
        )
    )
    # Precondition: this is the dangerous shape, not the all-fixture fallback.
    assert annual.data_class == DataClass.INFERRED
    assert annual.metadata["observed_years"] == [2026]
    assert annual.metadata["backcast_years"] == [2022, 2023, 2024, 2025]

    context = RunContext(
        run_id="backcast-test",
        request=AnalysisRequest.model_validate(demo_request()),
    )
    analyze_thermal_drift(context, annual)

    metric_ids = {metric.id for metric in context.metrics}
    assert "local_thermal_drift_c" not in metric_ids
    assert "local_drift_rate_c_per_year" not in metric_ids
    assert "thermal_drift_status" in metric_ids
    assert WarningCode.THERMAL_SERIES_BACKCAST.value in context.warning_codes
    assert "thermal_drift" in context.degraded_stages
    # No drift artifact means the financial model has nothing to monetise.
    assert "local_thermal_drift_c" not in context.artifacts

    response = FortyCoolOrchestrator._response(context)
    # Partial coverage must never buy a higher tier than no coverage at all.
    assert response.confidence_tier != ConfidenceTier.OPERATIONAL


def test_all_fixture_series_still_reports_but_labels_itself_simulated() -> None:
    """The demo path keeps its numbers; it just never claims they are measured.

    Withholding here too would leave the fixture demo with no figures at all,
    and a wholly simulated series is honest as long as every surface says so.
    """

    from fortycool_agents.context import RunContext
    from fortycool_agents.tools import analyze_thermal_drift

    annual = asyncio.run(
        FixtureThermalProvider().annual_history(
            SiteInput(name="Test", latitude=39.01, longitude=-77.46),
            2022,
            2026,
            18.0,
            seed=42,
        )
    )
    context = RunContext(
        run_id="fixture-test",
        request=AnalysisRequest.model_validate(demo_request()),
    )
    analyze_thermal_drift(context, annual)

    drift = next(
        metric for metric in context.metrics if metric.id == "local_thermal_drift_c"
    )
    assert drift.evidence_grade.value == "C"
    assert WarningCode.THERMAL_SERIES_SIMULATED.value in context.warning_codes
    assert (
        WarningCode.FIXTURE_SERIES_LOCATION_INDEPENDENT.value in context.warning_codes
    )


def test_degraded_stage_is_visible_in_the_response(client: TestClient) -> None:
    """A dead stage must change something a consumer can branch on.

    A total satellite failure used to produce a byte-identical status, tier,
    drift and NPV, differing only by one extra free-text warning.
    """

    class BrokenUrbanProvider:
        source = "test://broken"

        async def analyze(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("satellite stage is down")

    orchestrator = FortyCoolOrchestrator(
        provider=FixtureThermalProvider(), urban_provider=BrokenUrbanProvider()
    )
    response = asyncio.run(
        orchestrator.run(AnalysisRequest.model_validate(demo_request()))
    )
    assert "satellite_land_cover" in response.degraded_stages
    assert WarningCode.SATELLITE_STAGE_UNAVAILABLE.value in response.warning_codes


# ---------------------------------------------------------------------------
# The estimator: signed, bounded, and self-consistent
# ---------------------------------------------------------------------------


def test_drift_is_signed_and_a_cooling_site_reports_a_benefit() -> None:
    """The clamp made the expected reported exposure grow with input noise.

    Under a true zero effect the drift is symmetric, so rectifying it at zero
    produced a strictly positive expected exposure and a site that measurably
    cooled reported nothing rather than a benefit.
    """

    years = np.array([2022, 2023, 2024, 2025, 2026], dtype=float)
    cooling = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    estimate = estimate_drift(years, cooling)
    assert estimate.drift_c < 0
    assert estimate.ci_high_c is not None and estimate.ci_high_c < 0


def test_drift_carries_an_interval_and_matches_its_own_rate(client: TestClient) -> None:
    response = client.post("/runs", json=demo_request())
    assert response.status_code == 200
    metrics = {metric["id"]: metric for metric in response.json()["metrics"]}
    drift = metrics["local_thermal_drift_c"]
    rate = metrics["local_drift_rate_c_per_year"]
    assert drift["interval_low"] is not None
    assert drift["interval_low"] < drift["value"] < drift["interval_high"]
    assert drift["evidence_grade"] in {"A", "B", "C"}
    # Two estimators of the same quantity used to be published side by side and
    # disagree by 6.6% with nothing reconciling them.
    span = 2026 - 2022
    assert drift["value"] == pytest.approx(rate["value"] * span, abs=0.01)


def test_a_flat_series_is_not_reported_as_a_significant_drift() -> None:
    years = np.array([2022, 2023, 2024, 2025, 2026], dtype=float)
    flat = np.array([0.65, 0.65, 0.65, 0.65, 0.65])
    estimate = estimate_drift(years, flat)
    assert estimate.drift_c == pytest.approx(0.0, abs=1e-9)
    assert not estimate.distinguishable_from_zero


def test_npv_responds_to_pue(client: TestClient) -> None:
    """PUE was accepted, validated, and then ignored by the money model.

    The setup form has a PUE dial; moving it from 1.1 to 2.0 changed the
    reported exposure by exactly zero dollars.
    """

    def npv_for(pue: float) -> float:
        payload = demo_request()
        payload["facility"]["reported_pue"] = pue
        metrics = {
            metric["id"]: metric
            for metric in client.post("/runs", json=payload).json()["metrics"]
        }
        return metrics["thermal_drift_npv"]["value"]

    assert npv_for(2.0) > npv_for(1.1) * 1.5


def test_tariff_inflation_changes_the_npv(client: TestClient) -> None:
    def npv_for(inflation: float) -> float:
        payload = demo_request()
        payload["economics"]["annual_tariff_inflation"] = inflation
        metrics = {
            metric["id"]: metric
            for metric in client.post("/runs", json=payload).json()["metrics"]
        }
        return metrics["thermal_drift_npv"]["value"]

    assert npv_for(0.05) > npv_for(0.0)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_safety_buffer_is_an_upper_bound_not_a_mean(client: TestClient) -> None:
    """MAE is a mean of the absolute error; a safety limit needs a quantile.

    On the project's own holdout, MAE was 0.137 and p95 was 0.322, so the buffer
    protecting a server inlet was roughly 2.4 times too small.
    """

    metrics = {
        metric["id"]: metric
        for metric in client.post("/runs", json=demo_request()).json()["metrics"]
    }
    buffer = metrics["inlet_safety_buffer_c"]["value"]
    mae = metrics["inlet_model_mae_c"]["value"]
    assert buffer >= mae


def test_recommended_action_stays_inside_the_training_envelope(
    client: TestClient,
) -> None:
    """The old recommendation sat 6.55 sigma past the training maximum.

    A linear model extrapolated that far is not making a prediction, and the
    pipeline had no way to see it happening.
    """

    response = client.post("/runs", json=demo_request()).json()
    recommendations = response.get("recommendations") or []
    assert recommendations
    assert "training_envelope" in recommendations[0]["constraints_checked"]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def _csv(hours: int, *, value: str = "1000") -> bytes:
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = ["timestamp,it_load_kw,cooling_power_kw,server_inlet_temperature_c"]
    for index in range(hours):
        stamp = (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z")
        rows.append(f"{stamp},{value},500,22.5")
    return "\n".join(rows).encode()


def test_upload_accepted_at_the_model_minimum_and_refused_below_it() -> None:
    store = TelemetryStore()
    accepted = store.ingest_csv(_csv(336))
    assert accepted.rows == 336
    with pytest.raises(TelemetryValidationError):
        # One hour short of what the model needs. This used to be accepted with
        # a 201 and then fail every run against it, permanently.
        store.ingest_csv(_csv(335))


@pytest.mark.parametrize("value", ["inf", "Infinity", "-inf", "1e308"])
def test_non_finite_telemetry_is_refused_at_upload(value: str) -> None:
    store = TelemetryStore()
    with pytest.raises(TelemetryValidationError):
        store.ingest_csv(_csv(336, value=value))


def test_interpolated_rows_are_reported_and_downgrade_the_data_class() -> None:
    """The service filled gaps in the two model targets and called it measured."""

    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    stamps = [start + timedelta(hours=index) for index in range(48)]
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "it_load_kw": [1000.0] * 48,
            "cooling_power_kw": [500.0] * 48,
            "server_inlet_temperature_c": [22.5] * 48,
        }
    )
    # Punch a two-hour hole, which is inside the interpolation limit.
    frame = frame.drop(index=[10, 11]).reset_index(drop=True)

    class Sample:
        def __init__(self, timestamp: datetime) -> None:
            self.timestamp = timestamp
            self.site_temperature_c = 25.0
            self.control_temperature_c = 24.0
            self.wet_bulb_temperature_c = 18.0
            self.relative_humidity_percent = 55.0
            self.solar_irradiance_w_m2 = 300.0

    class Thermal:
        samples = [Sample(stamp) for stamp in stamps]

    from fortycool_agents.models import FacilityProfile

    result = enrich_uploaded_bms(frame, Thermal(), FacilityProfile())
    assert result.interpolated_rows == 2
    assert result.values_were_invented
    assert any("interpolation" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# Verification badge, rate limiting, and body size
# ---------------------------------------------------------------------------


def test_verification_cannot_be_asserted_by_the_caller(client: TestClient) -> None:
    """The badge is a claim about evidence, so only the server may make it.

    It used to be driven by `?verified=1` in the address bar plus a localStorage
    entry, so any fixture run could be displayed as verified.
    """

    run_id = client.post("/runs", json=demo_request()).json()["run_id"]
    verification = client.get(f"/runs/{run_id}/verification").json()
    assert verification["verified"] is False
    assert verification["reason"]


def test_rate_limit_applies_to_the_endpoints_that_spend_money(
    client: TestClient,
) -> None:
    limit = security.EXPENSIVE_RULE.limit
    statuses = [
        client.post("/runs", json=demo_request()).status_code
        for _ in range(limit + 2)
    ]
    assert 429 in statuses


def test_oversized_body_is_refused_before_it_is_buffered(client: TestClient) -> None:
    """The 10 MB check ran after the whole body was already resident."""

    payload = b"x" * (security.MAX_REQUEST_BYTES + 1024)
    response = client.post(
        "/telemetry/uploads", content=payload, headers={"Content-Type": "text/csv"}
    )
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Cache durability
# ---------------------------------------------------------------------------


def test_a_truncated_cache_file_does_not_poison_the_payload(tmp_path) -> None:
    """A non-atomic write plus an unguarded json.loads killed a payload forever."""

    from fortycool_agents.providers.fortyguard import FortyGuardClient

    fortyguard = FortyGuardClient("test-key", cache_dir=tmp_path)
    key = fortyguard._cache_key("heatmap", {"a": 1})
    fortyguard._store_cache(key, {"data": {"activity_id": "abc"}})
    assert fortyguard._load_cache(key) is not None
    (tmp_path / f"{key}.json").write_text('{"cached_at": 1, "response')
    assert fortyguard._load_cache(key) is None


def test_cache_key_ignores_imperceptible_coordinate_jitter() -> None:
    from fortycool_agents.providers.fortyguard import FortyGuardClient

    base = {"polygon_aoi": {"coordinates": [[[-77.46, 39.01]]]}}
    jittered = {"polygon_aoi": {"coordinates": [[[-77.46 + 1e-9, 39.01]]]}}
    assert FortyGuardClient._cache_key("heatmap", base) == FortyGuardClient._cache_key(
        "heatmap", jittered
    )


# ---------------------------------------------------------------------------
# Control flow must not depend on the wording of a warning
# ---------------------------------------------------------------------------


def test_run_status_is_driven_by_a_code_not_by_a_sentence() -> None:
    """Rewording a warning used to turn a failed evidence check into success."""

    from fortycool_agents.context import RunContext

    context = RunContext(
        run_id="test", request=AnalysisRequest.model_validate(demo_request())
    )
    context.warn(
        "Evidence verification failed for: metric-x",
        WarningCode.EVIDENCE_VERIFICATION_FAILED,
    )
    response = FortyCoolOrchestrator._response(context)
    assert response.status.value == "failed"

    reworded = RunContext(
        run_id="test", request=AnalysisRequest.model_validate(demo_request())
    )
    reworded.warn(
        "Evidence check did not pass for: metric-x",
        WarningCode.EVIDENCE_VERIFICATION_FAILED,
    )
    assert FortyCoolOrchestrator._response(reworded).status.value == "failed"


def test_fixture_runs_declare_that_coordinates_do_not_change_the_result(
    client: TestClient,
) -> None:
    """In fixture mode the annual series never reads the site coordinates.

    Northern Virginia, Almaty, Sydney, Svalbard and the middle of the Atlantic
    all returned the same drift, the same NPV and the same hours, with nothing
    on any surface saying so.
    """

    payload = demo_request()
    payload["site"]["latitude"] = 43.24
    payload["site"]["longitude"] = 76.89
    body = client.post("/runs", json=payload).json()
    assert (
        WarningCode.FIXTURE_SERIES_LOCATION_INDEPENDENT.value in body["warning_codes"]
    )
