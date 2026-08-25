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
