import fortycool_agents.api as api_module
from fastapi.testclient import TestClient

from fortycool_agents.api import app
from fortycool_agents.models import CopilotResponse


client = TestClient(app)


def test_browser_frontend_cors_preflight_is_accepted() -> None:
    response = client.options(
        "/runs",
        headers={
            "Origin": "https://frontend.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://frontend.example"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_integrated_dashboard_is_served_by_the_api() -> None:
    redirect = client.get("/", follow_redirects=False)
    page = client.get("/dashboard/pages/site_setup.html")
    browser_client = client.get("/dashboard/js/api.js")

    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/dashboard/pages/site_setup.html"
    assert page.status_code == 200
    assert "Northern Virginia Demo Campus" in page.text
    assert "Uploaded BMS CSV" in page.text
    assert browser_client.status_code == 200
    assert 'request("/run-jobs"' in browser_client.text
    assert '"completed_with_warnings"' in browser_client.text


def test_dashboard_request_contract_produces_all_mvp_outputs() -> None:
    payload = {
        "site": {
            "name": "Northern Virginia Demo Campus",
            "latitude": 39.01,
            "longitude": -77.46,
            "timezone": "America/New_York",
        },
        "analysis_modes": ["thermal_drift", "operations_12h", "investment"],
        "facility": {
            "archetype": "colocation_water_cooled",
            "it_capacity_mw": 10,
            "reported_pue": 1.25,
        },
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
        "temperature_eligibility_threshold_c": 18,
        "use_demo_defaults": True,
    }

    created = client.post("/run-jobs", json=payload)

    assert created.status_code == 202
    run_id = created.json()["run_id"]
    run = client.get(f"/runs/{run_id}")
    assert run.status_code == 200
    body = run.json()
    metric_ids = {item["id"] for item in body["metrics"]}
    chart_ids = {item["id"] for item in body["charts"]}
    assert {
        "local_thermal_drift_c",
        "local_drift_rate_c_per_year",
        "temperature_eligible_hours_lost",
        "forecast_savings_kwh",
        "forecast_safety_margin_c",
        "thermal_drift_npv",
    } <= metric_ids
    assert {
        "thermal_drift_timeseries",
        "temperature_forecast_12h",
        "historical_day_backtest",
        "baseline_vs_optimized",
    } <= chart_ids
    assert body["recommendations"]
    assert body["evidence"]


def test_health_and_tool_catalog() -> None:
    health = client.get("/health")
    tools = client.get("/tools")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert tools.status_code == 200
    assert {item["name"] for item in tools.json()} >= {
        "calculate_local_drift",
        "discover_thermal_drift_site",
        "evaluate_operating_scenarios",
        "explain_completed_analysis",
        "review_recommendation_safety",
    }

    copilot_schema = client.get("/schemas/copilot-request")
    assert copilot_schema.status_code == 200
    assert "question" in copilot_schema.json()["properties"]

    discovery_schema = client.get("/schemas/discovery-request")
    assert discovery_schema.status_code == 200
    assert "minimum_local_drift_c" in discovery_schema.json()["properties"]


def test_public_catalog_and_discovery_tool_are_typed_and_evidence_backed() -> None:
    catalog = client.get("/discovery/catalog")
    discovery = client.post("/agent-tools/site-discovery", json={})

    assert catalog.status_code == 200
    assert len(catalog.json()) == 4
    assert all(item["operator_source_url"].startswith("https://") for item in catalog.json())
    assert discovery.status_code == 200
    body = discovery.json()
    assert body["status"] == "qualified_candidate_found"
    assert body["winner"]["qualified"] is True
    evidence_ids = {item["id"] for item in body["evidence"]}
    assert set(body["winner"]["evidence_ids"]) <= evidence_ids


def test_run_can_be_retrieved_with_evidence() -> None:
    payload = {
        "site": {
            "name": "Northern Virginia Demo Campus",
            "latitude": 39.01,
            "longitude": -77.46,
        },
        "analysis_modes": ["thermal_drift"],
        "simulation": {"enabled": True, "seed": 7},
    }
    created = client.post("/runs", json=payload)

    assert created.status_code == 200
    body = created.json()
    fetched = client.get(f"/runs/{body['run_id']}")
    evidence_id = body["metrics"][0]["evidence_ids"][0]
    evidence = client.get(f"/runs/{body['run_id']}/evidence/{evidence_id}")

    assert fetched.status_code == 200
    assert evidence.status_code == 200
    assert evidence.json()["id"] == evidence_id


def test_investment_memo_is_a_downloadable_pdf() -> None:
    payload = {
        "site": {
            "name": "Memo Demo",
            "latitude": 39.01,
            "longitude": -77.46,
        },
        "analysis_modes": ["thermal_drift", "operations_12h", "investment"],
        "simulation": {"enabled": True, "seed": 31},
    }
    created = client.post("/runs", json=payload)
    run_id = created.json()["run_id"]

    memo = client.get(f"/runs/{run_id}/memo.pdf")

    assert memo.status_code == 200
    assert memo.headers["content-type"] == "application/pdf"
    assert f"fortycool-thermaldrift-{run_id[:8]}.pdf" in memo.headers[
        "content-disposition"
    ]
    assert memo.content.startswith(b"%PDF-")
    assert memo.content.rstrip().endswith(b"%%EOF")
    assert len(memo.content) > 5_000


def test_missing_run_memo_returns_not_found() -> None:
    response = client.get("/runs/does-not-exist/memo.pdf")

    assert response.status_code == 404


def test_background_job_exposes_trace_stream_and_persists_result() -> None:
    payload = {
        "site": {
            "name": "Background Job Demo",
            "latitude": 39.01,
            "longitude": -77.46,
        },
        "analysis_modes": ["thermal_drift"],
        "simulation": {"enabled": True, "seed": 13},
    }
    created = client.post("/run-jobs", json=payload)

    assert created.status_code == 202
    run_id = created.json()["run_id"]
    job = client.get(f"/run-jobs/{run_id}")
    stream = client.get(f"/run-jobs/{run_id}/stream")
    persisted = client.get(f"/runs/{run_id}")

    assert job.status_code == 200
    assert job.json()["state"] == "completed"
    assert job.json()["event_count"] > 0
    assert stream.status_code == 200
    assert "event: trace" in stream.text
    assert "event: terminal" in stream.text
    assert persisted.status_code == 200


def test_copilot_endpoint_grounds_answer_in_persisted_run(monkeypatch) -> None:
    payload = {
        "site": {
            "name": "Copilot Demo",
            "latitude": 39.01,
            "longitude": -77.46,
        },
        "analysis_modes": ["thermal_drift"],
        "simulation": {"enabled": True, "seed": 19},
    }
    created = client.post("/runs", json=payload)
    run_id = created.json()["run_id"]
    evidence_id = created.json()["evidence"][0]["id"]

    class StubCopilot:
        configured = True
        model = "gpt-4o"

        async def answer(self, run, request):
            assert run.run_id == run_id
            assert request.question == "What should the operator know?"
            return CopilotResponse(
                run_id=run_id,
                model="gpt-4o",
                response_id="resp_api_test",
                answer="The analysis is indicative and advisory.",
                key_findings=["The site was compared with a matched control."],
                cautions=["Historical inputs are simulated in fixture mode."],
                evidence_ids=[evidence_id],
                suggested_questions=["Which facility inputs should be confirmed?"],
            )

    monkeypatch.setattr(api_module, "copilot_service", StubCopilot())
    response = client.post(
        f"/runs/{run_id}/copilot",
        json={
            "question": "What should the operator know?",
            "audience": "operator",
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4o"
    assert response.json()["evidence_ids"] == [evidence_id]
