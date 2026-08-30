from __future__ import annotations

from pathlib import Path

import json
import logging
import os
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .catalog import serialized_catalog
from .copilot import (
    CopilotError,
    CopilotQuotaError,
    CopilotUnavailableError,
    FortyCoolCopilot,
)
from .database import Database
from .demo import latest_verified_demo, verified_manifest_for
from .discovery import SiteDiscoveryAgent
from .discovery_catalog import public_catalog
from .jobs import RunJobManager
from .memo import build_investment_memo
from .models import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResponse,
    CopilotRequest,
    CopilotResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    RunJobStatus,
    TelemetryUpload,
    VerifiedDemoResponse,
)
from .orchestrator import FortyCoolOrchestrator
from .security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    api_key_required,
    guard_analysis,
    guard_copilot,
    guard_read,
    guard_upload,
)
from .storage import RunRepository
from .telemetry import TelemetryStore, TelemetryValidationError


logger = logging.getLogger(__name__)

# The interactive API console and the schema dump are convenient locally and a
# free map of every money-spending endpoint once the service is reachable from
# the internet. Off unless the operator asks for them.
_DOCS_ENABLED = os.getenv("FORTYCOOL_ENABLE_DOCS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FORTYCOOL_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(
    title="FortyCool Agent Service",
    version="0.1.0",
    description="Evidence-first thermal decision tools for a hybrid data-center digital twin.",
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)
# Outermost first: headers are attached to every response, including the ones
# the size limiter short-circuits.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
if _CORS_ORIGINS:
    # CORS is deliberately opt-in. The public demo uses ``*`` without browser
    # credentials; production operators can instead provide a comma-separated
    # allowlist of exact frontend origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Turn a rejected input into a 422 instead of an unhandled 500.

    Several schema-valid requests reached code that raises `ValueError` for an
    input it cannot use - a baseline year with no window after it, telemetry an
    hour short of the model's minimum, an interval the weather provider does not
    cover. Each surfaced as a 500 with a stack trace, which tells a client
    nothing and looks like a service fault rather than a bad request.
    """

    correlation_id = uuid4().hex[:12]
    logger.warning("rejected request %s: %s", correlation_id, exc, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "correlation_id": correlation_id},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a correlation id, never the internal exception text.

    Job errors used to be handed back verbatim, including pydantic field paths,
    offending input values, and a link to the library docs.
    """

    correlation_id = uuid4().hex[:12]
    logger.exception("unhandled error %s", correlation_id, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "the service could not complete this request",
            "correlation_id": correlation_id,
        },
    )
database = Database()
telemetry_store = TelemetryStore(database=database)

# Provider construction reads environment variables and, for Earth Engine, opens
# an authenticated session. Any of that failing used to raise during module
# import, which killed the process with a bare traceback before `/health` could
# ever answer, so a misconfiguration was indistinguishable from a crash.
provider_error: str | None = None
try:
    orchestrator = FortyCoolOrchestrator(telemetry_store=telemetry_store)
except Exception as exc:  # configuration faults must not kill the process
    provider_error = f"{type(exc).__name__}: {exc}"
    logger.error("provider configuration failed: %s", provider_error)
    from .providers.fixture import FixtureThermalProvider
    from .providers.urban import FixtureUrbanProvider

    orchestrator = FortyCoolOrchestrator(
        provider=FixtureThermalProvider(),
        urban_provider=FixtureUrbanProvider(),
        telemetry_store=telemetry_store,
    )

run_repository = RunRepository(database=database)
job_manager = RunJobManager(orchestrator, run_repository)
copilot_service = FortyCoolCopilot()
discovery_agent = SiteDiscoveryAgent(
    orchestrator.provider,
    orchestrator.urban_provider,
)
web_root = Path(__file__).with_name("web")

# Upstream fan-out for one anonymous discovery request reached roughly ninety
# paid provider activities. An authenticated caller keeps the full range.
UNAUTHENTICATED_MAX_CANDIDATES = int(
    os.getenv("FORTYCOOL_OPEN_MAX_CANDIDATES", 4)
)
UNAUTHENTICATED_MAX_SHORTLIST = int(os.getenv("FORTYCOOL_OPEN_MAX_SHORTLIST", 1))


@app.get("/", include_in_schema=False)
async def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/pages/site_setup.html")


@app.get("/dashboard/api-config.js", include_in_schema=False)
async def dashboard_api_config() -> Response:
    """Inject the backend origin when Vercel serves only the dashboard."""

    api_base = os.getenv("FORTYCOOL_PUBLIC_API_BASE", "").rstrip("/")
    return Response(
        content=f"window.FORTYCOOL_API_BASE = {json.dumps(api_base)};\n",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "degraded" if provider_error else "ok",
        "provider_error": provider_error or "",
        "thermal_provider": type(orchestrator.provider).__name__,
        "urban_provider": type(orchestrator.urban_provider).__name__,
        "copilot": "configured" if copilot_service.configured else "not_configured",
        "copilot_model": copilot_service.model,
        "authentication": "required" if api_key_required() else "open",
        "database": database.backend,
    }


@app.get("/tools")
async def tools() -> list[dict]:
    return serialized_catalog()


@app.get("/schemas/analysis-request")
async def analysis_request_schema() -> dict:
    return AnalysisRequest.model_json_schema()


@app.get("/schemas/analysis-response")
async def analysis_response_schema() -> dict:
    return AnalysisResponse.model_json_schema()


@app.get("/schemas/copilot-request")
async def copilot_request_schema() -> dict:
    return CopilotRequest.model_json_schema()


@app.get("/schemas/copilot-response")
async def copilot_response_schema() -> dict:
    return CopilotResponse.model_json_schema()


@app.get("/schemas/discovery-request")
async def discovery_request_schema() -> dict:
    return DiscoveryRequest.model_json_schema()


@app.get("/schemas/discovery-response")
async def discovery_response_schema() -> dict:
    return DiscoveryResponse.model_json_schema()


@app.get("/discovery/catalog")
async def discovery_catalog() -> list[dict]:
    return [candidate.model_dump(mode="json") for candidate in public_catalog()]


@app.get("/demo/verified-run", response_model=VerifiedDemoResponse)
async def verified_demo_run() -> VerifiedDemoResponse:
    demo = latest_verified_demo(run_repository)
    if demo is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "no completed run passed the FortyGuard history, Dynamic World history, "
                "and no-backcast verification gates"
            ),
        )
    return demo


@app.post(
    "/agent-tools/site-discovery",
    response_model=DiscoveryResponse,
    dependencies=[Depends(guard_analysis)],
)
async def site_discovery_tool(request: DiscoveryRequest) -> DiscoveryResponse:
    if not api_key_required():
        candidate_count = len(request.candidates or [])
        if (
            candidate_count > UNAUTHENTICATED_MAX_CANDIDATES
            or request.shortlist_size > UNAUTHENTICATED_MAX_SHORTLIST
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "this instance accepts at most "
                    f"{UNAUTHENTICATED_MAX_CANDIDATES} candidates and a shortlist of "
                    f"{UNAUTHENTICATED_MAX_SHORTLIST} without an API key; each "
                    "candidate costs a paid upstream activity"
                ),
            )
    return await discovery_agent.run(request)


@app.post(
    "/telemetry/uploads",
    response_model=TelemetryUpload,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(guard_upload)],
)
async def upload_telemetry(request: Request) -> TelemetryUpload:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="send the CSV as a raw body with Content-Type: text/csv",
        )
    body = await request.body()
    try:
        # pandas parsing is synchronous and takes about a fifth of a second for
        # a 5 MB upload. On the event loop that stalls every other request,
        # including the health check and every open SSE stream.
        return await run_in_threadpool(telemetry_store.ingest_csv, body)
    except TelemetryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/telemetry/uploads/{upload_id}",
    response_model=TelemetryUpload,
    dependencies=[Depends(guard_read)],
)
async def get_telemetry_upload(upload_id: str) -> TelemetryUpload:
    try:
        return telemetry_store.metadata(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="telemetry upload not found") from exc


@app.delete(
    "/telemetry/uploads/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(guard_upload)],
)
async def delete_telemetry_upload(upload_id: str) -> Response:
    if not telemetry_store.delete(upload_id):
        raise HTTPException(status_code=404, detail="telemetry upload not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/runs",
    response_model=AnalysisResponse,
    dependencies=[Depends(guard_analysis)],
)
async def create_run(request: AnalysisRequest) -> AnalysisResponse:
    response = await orchestrator.run(request)
    await run_in_threadpool(run_repository.save, response)
    return response


@app.post(
    "/run-jobs",
    response_model=RunJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(guard_analysis)],
)
async def create_run_job(
    request: AnalysisRequest, background_tasks: BackgroundTasks
) -> RunJobStatus:
    job = job_manager.create()
    background_tasks.add_task(job_manager.execute, job.run_id, request)
    return job


@app.get(
    "/run-jobs/{run_id}",
    response_model=RunJobStatus,
    dependencies=[Depends(guard_read)],
)
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
    await run_in_threadpool(run_repository.save, response)
    return response


@app.post(
    "/agent-tools/thermal-drift",
    response_model=AnalysisResponse,
    dependencies=[Depends(guard_analysis)],
)
async def thermal_drift_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(request, [AnalysisMode.THERMAL_DRIFT])


@app.post(
    "/agent-tools/operations-12h",
    response_model=AnalysisResponse,
    dependencies=[Depends(guard_analysis)],
)
async def operations_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(request, [AnalysisMode.OPERATIONS_12H])


@app.post(
    "/agent-tools/investment",
    response_model=AnalysisResponse,
    dependencies=[Depends(guard_analysis)],
)
async def investment_tool(request: AnalysisRequest) -> AnalysisResponse:
    return await _run_agent_tool(
        request, [AnalysisMode.THERMAL_DRIFT, AnalysisMode.INVESTMENT]
    )


@app.get(
    "/runs/{run_id}",
    response_model=AnalysisResponse,
    dependencies=[Depends(guard_read)],
)
async def get_run(run_id: str) -> AnalysisResponse:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    return response


@app.get("/runs/{run_id}/verification", dependencies=[Depends(guard_read)])
async def get_run_verification(run_id: str) -> dict:
    """Say whether THIS run passed the evidence gates, from the server.

    The dashboard used to stamp a VERIFIED badge whenever `?verified=1` was in
    the URL, so any run - including one built entirely from fixture data - could
    be made to display as verified by editing the address bar. The gate itself
    was always real; nothing ever asked it about the run on screen.
    """

    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    manifest = verified_manifest_for(response)
    if manifest is None:
        return {
            "run_id": run_id,
            "verified": False,
            "reason": (
                "this run did not pass the FortyGuard history, Dynamic World history, "
                "and no-backcast verification gates"
            ),
        }
    return {
        "run_id": run_id,
        "verified": True,
        "verified_at": manifest.verified_at.isoformat(),
        "thermal_years": manifest.thermal_years,
        "observed_heatmap": manifest.observed_heatmap,
        "verification_notes": manifest.verification_notes,
    }


@app.get("/runs/{run_id}/memo.pdf", dependencies=[Depends(guard_read)])
async def get_investment_memo(run_id: str, request: Request) -> Response:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    # reportlab rendering is synchronous; keep it off the event loop.
    pdf = await run_in_threadpool(
        build_investment_memo,
        response,
        base_url=str(request.base_url).rstrip("/"),
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="fortycool-thermaldrift-{run_id[:8]}.pdf"'
            ),
            "Cache-Control": "no-store",
        },
    )


@app.post(
    "/runs/{run_id}/copilot",
    response_model=CopilotResponse,
    dependencies=[Depends(guard_copilot)],
)
async def ask_copilot(run_id: str, request: CopilotRequest) -> CopilotResponse:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return await copilot_service.answer(response, request)
    except CopilotQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CopilotError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/runs/{run_id}/events", dependencies=[Depends(guard_read)])
async def get_run_events(run_id: str) -> list[dict]:
    response = run_repository.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="run not found")
    return [event.model_dump(mode="json") for event in response.trace]


@app.get("/runs/{run_id}/evidence/{evidence_id}", dependencies=[Depends(guard_read)])
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


app.mount("/dashboard", StaticFiles(directory=web_root, html=True), name="dashboard")
