from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from fortycool_agents.api import app


client = TestClient(app)


def telemetry_csv(*, include_controls: bool = True) -> bytes:
    columns = [
        "timestamp",
        "it_load_kw",
        "cooling_power_kw",
        "server_inlet_temperature_c",
    ]
    if include_controls:
        columns.extend(
            [
                "supply_air_setpoint_c",
                "chilled_water_supply_c",
                "fan_speed_percent",
            ]
        )
    lines = [",".join(columns)]
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for index in range(15 * 24):
        timestamp = start + timedelta(hours=index)
        it_load = 16_500 + 700 * math.sin(2 * math.pi * index / 24)
        supply = 18.5 + (index % 3) * 0.5
        chilled = 6.5 + (index % 2) * 0.5
        fan = 68 + (index % 4) * 2
        cooling = 4_900 + 0.08 * (it_load - 16_000) - 170 * (supply - 18.5)
        cooling -= 90 * (chilled - 6.5)
        cooling += 14 * (fan - 68)
        inlet = supply + 4.2 + 0.025 * (70 - fan)
        values = [
            timestamp.isoformat(),
            f"{it_load:.3f}",
            f"{cooling:.3f}",
            f"{inlet:.3f}",
        ]
        if include_controls:
            values.extend([f"{supply:.3f}", f"{chilled:.3f}", f"{fan:.3f}"])
        lines.append(",".join(values))
    return "\n".join(lines).encode()


def test_csv_upload_can_drive_operational_model() -> None:
    upload = client.post(
        "/telemetry/uploads", content=telemetry_csv(), headers={"content-type": "text/csv"}
    )
    assert upload.status_code == 201
    metadata = upload.json()
    assert metadata["rows"] == 15 * 24

    analysis = client.post(
        "/runs",
        json={
            "site": {
                "name": "Uploaded Telemetry Demo",
                "latitude": 39.01,
                "longitude": -77.46,
            },
            "analysis_modes": ["operations_12h"],
            "facility": {
                "it_capacity_mw": 24,
                "typical_it_load_mw": 17,
                "reported_pue": 1.38,
            },
            "constraints": {
                "maximum_server_inlet_temperature_c": 27,
                "minimum_safety_margin_c": 2,
                "supply_air_setpoint_min_c": 18,
                "supply_air_setpoint_max_c": 22,
                "maximum_setpoint_change_c": 1,
                "minimum_model_confidence": 0.7,
                "forecast_max_age_minutes": 180,
            },
            "telemetry": {"source": "uploaded", "upload_id": metadata["upload_id"]},
            "use_demo_defaults": False,
        },
    )
    assert analysis.status_code == 200
    body = analysis.json()
    assert any(item["data_class"] == "uploaded" for item in body["evidence"])
    assert any(item["id"] == "historical_day_backtest" for item in body["charts"])
    assert body["recommendations"]


def test_invalid_csv_is_rejected_with_actionable_error() -> None:
    response = client.post(
        "/telemetry/uploads",
        content=b"timestamp,it_load_kw\n2026-08-01T00:00:00Z,1000\n",
        headers={"content-type": "text/csv"},
    )
    assert response.status_code == 422
    assert "missing required columns" in response.json()["detail"]


def test_missing_control_history_causes_safety_hold() -> None:
    upload = client.post(
        "/telemetry/uploads",
        content=telemetry_csv(include_controls=False),
        headers={"content-type": "text/csv"},
    )
    upload_id = upload.json()["upload_id"]
    analysis = client.post(
        "/agent-tools/operations-12h",
        json={
            "site": {
                "name": "Minimal Telemetry Demo",
                "latitude": 39.01,
                "longitude": -77.46,
            },
            "facility": {
                "it_capacity_mw": 24,
                "typical_it_load_mw": 17,
                "reported_pue": 1.38,
            },
            "telemetry": {"source": "uploaded", "upload_id": upload_id},
            "use_demo_defaults": True,
        },
    )
    assert analysis.status_code == 200
    recommendation = analysis.json()["recommendations"][0]
    assert recommendation["verdict"] == "insufficient_data"
    assert recommendation["action"] == "hold_current_settings"


def test_agent_contract_schemas_are_exposed() -> None:
    request_schema = client.get("/schemas/analysis-request")
    response_schema = client.get("/schemas/analysis-response")

    assert request_schema.status_code == 200
    assert response_schema.status_code == 200
    assert "site" in request_schema.json()["properties"]
    assert "evidence" in response_schema.json()["properties"]
