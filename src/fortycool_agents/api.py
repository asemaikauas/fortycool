from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from .catalog import serialized_catalog
from .jobs import RunJobManager
from .models import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResponse,
    RunJobStatus,
    TelemetryUpload,
)
from .orchestrator import FortyCoolOrchestrator
from .storage import RunRepository
from .telemetry import TelemetryStore, TelemetryValidationError


app = FastAPI(
    title="FortyCool Agent Service",
    version="0.1.0",
    description="Evidence-first thermal decision tools for a hybrid data-center digital twin.",
)
telemetry_store = TelemetryStore()
orchestrator = FortyCoolOrchestrator(telemetry_store=telemetry_store)
run_repository = RunRepository()
job_manager = RunJobManager(orchestrator, run_repository)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "thermal_provider": type(orchestrator.provider).__name__}


@app.get("/tools")
async def tools() -> list[dict]:
    return serialized_catalog()


@app.get("/schemas/analysis-request")
async def analysis_request_schema() -> dict:
    return AnalysisRequest.model_json_schema()


@app.get("/schemas/analysis-response")
async def analysis_response_schema() -> dict:
    return AnalysisResponse.model_json_schema()


@app.post("/telemetry/uploads", response_model=TelemetryUpload, status_code=status.HTTP_201_CREATED)
async def upload_telemetry(request: Request) -> TelemetryUpload:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="send the CSV as a raw body with Content-Type: text/csv",
        )
    try:
        return telemetry_store.ingest_csv(await request.body())
    except TelemetryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/telemetry/uploads/{upload_id}", response_model=TelemetryUpload)
async def get_telemetry_upload(upload_id: str) -> TelemetryUpload:
    try:
        return telemetry_store.metadata(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="telemetry upload not found") from exc


@app.delete("/telemetry/uploads/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_telemetry_upload(upload_id: str) -> Response:
    if not telemetry_store.delete(upload_id):
        raise HTTPException(status_code=404, detail="telemetry upload not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/runs", response_model=AnalysisResponse)
async def create_run(request: AnalysisRequest) -> AnalysisResponse:
    response = await orchestrator.run(request)
    run_repository.save(response)
    return response


@app.post("/run-jobs", response_model=RunJobStatus, status_code=status.HTTP_202_ACCEPTED)
async def create_run_job(
    request: AnalysisRequest, background_tasks: BackgroundTasks
) -> RunJobStatus:
    job = job_manager.create()
    background_tasks.add_task(job_manager.execute, job.run_id, request)
    return job


@app.get("/run-jobs/{run_id}", response_model=RunJobStatus)
async def get_run_job(run_id: str) -> RunJobStatus:
    try:
        return job_manager.snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run job not found") from exc


@app.get("/run-jobs/{run_id}/stream")
async def stream_run_job(run_id: str) -> StreamingResponse:
    try:
        job_manager.snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run job not found") from exc
    return StreamingResponse(
        job_manager.stream(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_agent_tool(
    request: AnalysisRequest, modes: list[AnalysisMode]
) -> AnalysisResponse:
    scoped_request = request.model_copy(deep=True, update={"analysis_modes": modes})
    response = await orchestrator.run(scoped_request)
    run_repository.save(response)
    return response


@app.post("/agent-tools/thermal-drift", response_model=AnalysisResponse)
async def thermal_drift_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(request, [AnalysisMode.THERMAL_DRIFT])


@app.post("/agent-tools/operations-12h", response_model=AnalysisResponse)
async def operations_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(request, [AnalysisMode.OPERATIONS_12H])


@app.post("/agent-tools/investment", response_model=AnalysisResponse)
async def investment_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(
        request, [AnalysisMode.THERMAL_DRIFT, AnalysisMode.INVESTMENT]
    )


@app.get("/runs/{run_id}", response_model=AnalysisResponse)
async def get_run(run_id: str) -> AnalysisResponse:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    return response


@app.get("/runs/{run_id}/events")
async def get_run_events(run_id: str) -> list[dict]:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    return [event.model_dump(mode="json") for event in response.trace]


@app.get("/runs/{run_id}/evidence/{evidence_id}")
async def get_evidence(run_id: str, evidence_id: str) -> dict:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    evidence = next(
        (item for item in response.evidence if item.id == evidence_id), None
    )
    if evidence is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    return evidence.model_dump(mode="json")
