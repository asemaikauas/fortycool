import fortycool_agents.api as api_module
from fastapi.testclient import TestClient

from fortycool_agents.api import app
from fortycool_agents.models import CopilotResponse


client = TestClient(app)


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
