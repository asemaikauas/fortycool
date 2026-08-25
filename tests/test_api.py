from fastapi.testclient import TestClient

from fortycool_agents.api import app


client = TestClient(app)


def test_health_and_tool_catalog() -> None:
    health = client.get("/health")
    tools = client.get("/tools")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert tools.status_code == 200
    assert {item["name"] for item in tools.json()} >= {
        "calculate_local_drift",
        "evaluate_operating_scenarios",
        "review_recommendation_safety",
    }


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
