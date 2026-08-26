/**
 * FortyCool browser client.
 *
 * The dashboard is served by the FastAPI process at /dashboard, so API calls
 * are same-origin by default. A separate frontend can still set
 * window.FORTYCOOL_API_BASE before this file loads.
 */
const FortyCoolAPI = (() => {
  const BASE_URL = (window.FORTYCOOL_API_BASE || "").replace(/\/$/, "");

  async function request(path, { method = "GET", body, headers } = {}) {
    const response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...(headers || {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      const raw = await response.text().catch(() => "");
      let detail = raw;
      try {
        detail = JSON.parse(raw).detail || raw;
      } catch (_) {
        // Keep the original response text.
      }
      throw new Error(`${method} ${path} returned ${response.status}: ${String(detail).slice(0, 500)}`);
    }
    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/json") ? response.json() : response.text();
  }

  function byId(items) {
    return Object.fromEntries((items || []).map((item) => [item.id, item]));
  }

  function normalizeRun(run) {
    return {
      ...run,
      metricsById: byId(run.metrics),
      chartsById: byId(run.charts),
      evidenceById: byId(run.evidence),
      isTerminal: ["completed", "completed_with_warnings", "needs_input", "failed"].includes(run.status),
      isSuccessful: ["completed", "completed_with_warnings"].includes(run.status),
    };
  }

  function escapeHTML(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatNumber(value, maximumFractionDigits = 2) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(number);
  }

  const health = () => request("/health");
  const tools = () => request("/tools");
  const analysisRequestSchema = () => request("/schemas/analysis-request");
  const analysisResponseSchema = () => request("/schemas/analysis-response");
  const discoveryCatalog = () => request("/discovery/catalog");
  const runSiteDiscovery = (payload = {}) => request("/agent-tools/site-discovery", { method: "POST", body: payload });

  const startRun = (analysisRequest) => request("/run-jobs", { method: "POST", body: analysisRequest });
  const getRunJobStatus = (runId) => request(`/run-jobs/${encodeURIComponent(runId)}`);
  const streamRunJob = (runId) => new EventSource(`${BASE_URL}/run-jobs/${encodeURIComponent(runId)}/stream`);
  const getRun = async (runId) => normalizeRun(await request(`/runs/${encodeURIComponent(runId)}`));
  const getRunEvents = (runId) => request(`/runs/${encodeURIComponent(runId)}/events`);
  const getEvidence = (runId, evidenceId) =>
    request(`/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(evidenceId)}`);
  const askCopilot = (runId, question, audience = "operator") =>
    request(`/runs/${encodeURIComponent(runId)}/copilot`, {
      method: "POST",
      body: { question, audience },
    });

  const runThermalDrift = (payload) => request("/agent-tools/thermal-drift", { method: "POST", body: payload });
  const runOperations12h = (payload) => request("/agent-tools/operations-12h", { method: "POST", body: payload });
  const runInvestment = (payload) => request("/agent-tools/investment", { method: "POST", body: payload });

  async function uploadTelemetry(file) {
    const response = await fetch(`${BASE_URL}/telemetry/uploads`, {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: await file.text(),
    });
    if (!response.ok) {
      const raw = await response.text().catch(() => "");
      throw new Error(`Telemetry upload returned ${response.status}: ${raw.slice(0, 500)}`);
    }
    return response.json();
  }
  const getUpload = (uploadId) => request(`/telemetry/uploads/${encodeURIComponent(uploadId)}`);
  const deleteUpload = (uploadId) => request(`/telemetry/uploads/${encodeURIComponent(uploadId)}`, { method: "DELETE" });

  const RUN_HISTORY_KEY = "fortycool_run_history";
  function recordRunLocally(runId, meta) {
    const list = JSON.parse(localStorage.getItem(RUN_HISTORY_KEY) || "[]")
      .filter((item) => item.run_id !== runId);
    list.unshift({ run_id: runId, started_at: new Date().toISOString(), ...meta });
    localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(list.slice(0, 100)));
    localStorage.setItem("fortycool_last_run_id", runId);
  }
  function listLocalRuns() {
    return JSON.parse(localStorage.getItem(RUN_HISTORY_KEY) || "[]");
  }

  return {
    BASE_URL,
    health,
    tools,
    analysisRequestSchema,
    analysisResponseSchema,
    discoveryCatalog,
    runSiteDiscovery,
    startRun,
    getRunJobStatus,
    streamRunJob,
    getRun,
    getRunEvents,
    getEvidence,
    askCopilot,
    runThermalDrift,
    runOperations12h,
    runInvestment,
    uploadTelemetry,
    getUpload,
    deleteUpload,
    normalizeRun,
    escapeHTML,
    formatNumber,
    recordRunLocally,
    listLocalRuns,
  };
})();
