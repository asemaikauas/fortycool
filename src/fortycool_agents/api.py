from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .catalog import serialized_catalog
from .models import AnalysisRequest, AnalysisResponse
from .orchestrator import FortyCoolOrchestrator


app = FastAPI(
    title="FortyCool Agent Service",
    version="0.1.0",
    description="Evidence-first thermal decision tools for a simulated data-center digital twin.",
)
orchestrator = FortyCoolOrchestrator()
run_store: dict[str, AnalysisResponse] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "thermal_provider": type(orchestrator.provider).__name__}


@app.get("/tools")
async def tools() -> list[dict]:
    return serialized_catalog()


@app.post("/runs", response_model=AnalysisResponse)
async def create_run(request: AnalysisRequest) -> AnalysisResponse:
    response = await orchestrator.run(request)
    run_store[response.run_id] = response
    return response


@app.get("/runs/{run_id}", response_model=AnalysisResponse)
async def get_run(run_id: str) -> AnalysisResponse:
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="run not found")
    return run_store[run_id]


@app.get("/runs/{run_id}/events")
async def get_run_events(run_id: str) -> list[dict]:
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="run not found")
    return [event.model_dump(mode="json") for event in run_store[run_id].trace]


@app.get("/runs/{run_id}/evidence/{evidence_id}")
async def get_evidence(run_id: str, evidence_id: str) -> dict:
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="run not found")
    evidence = next(
        (item for item in run_store[run_id].evidence if item.id == evidence_id), None
    )
    if evidence is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    return evidence.model_dump(mode="json")
