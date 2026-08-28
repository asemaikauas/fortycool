(function () {
  const el = (id) => document.getElementById(id);
  const latInput = el("latInput");
  const lonInput = el("lonInput");
  const locationInput = el("locationInput");
  const sampleSitePicker = el("sampleSitePicker");
  const telemetrySource = el("telemetrySource");
  const telemetryFile = el("telemetryFile");
  let telemetryUpload = null;

  function log(message, className = "") {
    const terminal = el("terminalLog");
    const line = document.createElement("div");
    line.className = className;
    line.textContent = message;
    terminal.insertBefore(line, terminal.lastElementChild);
    terminal.scrollTop = terminal.scrollHeight;
  }

  function refreshCoordsLabel() {
    el("coordsLabel").textContent = `LAT: ${latInput.value || "—"} | LNG: ${lonInput.value || "—"}`;
  }

  function setSelectedSite(option) {
    if (!option?.dataset.lat) return;
    latInput.value = option.dataset.lat;
    lonInput.value = option.dataset.lon;
    locationInput.value = option.dataset.name || option.textContent.trim();
    locationInput.dataset.timezone = option.dataset.timezone || "America/New_York";
    refreshCoordsLabel();
  }

  async function loadProviderStatus() {
    try {
      const health = await FortyCoolAPI.health();
      el("syncLabel").innerHTML = `<span class="material-symbols-outlined text-[14px]">satellite_alt</span> ${FortyCoolAPI.escapeHTML(health.thermal_provider)} + ${FortyCoolAPI.escapeHTML(health.urban_provider)}`;
      log(`[SYS] Providers: ${health.thermal_provider} / ${health.urban_provider}`, "text-status-observed");
      log(`[SYS] Copilot: ${health.copilot} (${health.copilot_model})`);
      if (health.provider_error) {
        log(`[WARN] Provider configuration failed: ${health.provider_error}`, "text-safety-warning");
      }
      // Say it on the screen, before anyone runs anything. In fixture mode the
      // annual series is a fixed synthetic trend that never reads the entered
      // coordinates, so the headline figures are identical for every site.
      if (String(health.thermal_provider).startsWith("Fixture")) {
        const banner = el("fixtureModeBanner");
        if (banner) banner.classList.remove("hidden");
        log(
          "[SYS] Fixture mode: thermal history is a fixed demo series and does not depend on the coordinates you enter.",
          "text-status-simulated"
        );
      }
    } catch (error) {
      el("syncLabel").textContent = "Backend unavailable";
      log(`[ERR] ${error.message}`, "text-safety-critical");
    }
  }

  async function loadCatalog() {
    try {
      const candidates = await FortyCoolAPI.discoveryCatalog();
      candidates.forEach((candidate) => {
        const option = document.createElement("option");
        option.value = candidate.id;
        option.textContent = `${candidate.operator} — ${candidate.site.name}`;
        option.dataset.name = candidate.site.name;
        option.dataset.lat = candidate.site.latitude;
        option.dataset.lon = candidate.site.longitude;
        option.dataset.timezone = candidate.site.timezone;
        sampleSitePicker.appendChild(option);
      });
    } catch (error) {
      log(`[WARN] Public site catalog unavailable: ${error.message}`, "text-safety-warning");
    }
  }

  function toggleTelemetryPanel() {
    const uploaded = telemetrySource.value === "uploaded";
    el("telemetryUploadPanel").classList.toggle("hidden", !uploaded);
    el("telemetryUploadPanel").classList.toggle("flex", uploaded);
  }

  async function ensureTelemetryUpload() {
    if (telemetrySource.value !== "uploaded") return null;
    if (telemetryUpload) return telemetryUpload;
    if (!telemetryFile.files?.[0]) {
      throw new Error("Choose a BMS CSV file or switch telemetry back to simulated.");
    }
    el("telemetryUploadStatus").textContent = "Validating and uploading…";
    telemetryUpload = await FortyCoolAPI.uploadTelemetry(telemetryFile.files[0]);
    el("telemetryUploadStatus").textContent = `${telemetryUpload.rows} rows accepted · ${telemetryUpload.median_interval_minutes} min median interval`;
    log(`[SYS] Uploaded BMS telemetry ${telemetryUpload.upload_id}`, "text-status-observed");
    return telemetryUpload;
  }

  function numberValue(id) {
    return Number.parseFloat(el(id).value);
  }

  function buildRequest(upload) {
    const latitude = numberValue("latInput");
    const longitude = numberValue("lonInput");
    const capacity = numberValue("itLoadSlider");
    const reportedPue = numberValue("pueInput");
    const electricityPrice = numberValue("priceInput");
    if (!locationInput.value.trim() || !Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      throw new Error("Enter a site name and valid coordinates.");
    }
    return {
      site: {
        name: locationInput.value.trim(),
        latitude,
        longitude,
        timezone: locationInput.dataset.timezone || "America/New_York",
      },
      analysis_modes: ["thermal_drift", "operations_12h", "investment"],
      facility: {
        archetype: "colocation_water_cooled",
        it_capacity_mw: capacity,
        reported_pue: reportedPue,
      },
      economics: {
        electricity_price_per_kwh: electricityPrice,
        horizon_years: 10,
        discount_rate: 0.08,
      },
      telemetry: upload
        ? { source: "uploaded", upload_id: upload.upload_id }
        : { source: "simulated" },
      simulation: {
        enabled: !upload,
        seed: 42,
        history_days: 60,
        interval_minutes: 60,
        forecast_hours: 12,
      },
      baseline_year: Number.parseInt(el("baselineYearInput").value, 10),
      temperature_eligibility_threshold_c: numberValue("thresholdInput"),
      use_demo_defaults: true,
    };
  }

  latInput.addEventListener("input", refreshCoordsLabel);
  lonInput.addEventListener("input", refreshCoordsLabel);
  sampleSitePicker.addEventListener("change", (event) => setSelectedSite(event.target.selectedOptions[0]));
  telemetrySource.addEventListener("change", () => {
    telemetryUpload = null;
    toggleTelemetryPanel();
  });
  telemetryFile.addEventListener("change", () => {
    telemetryUpload = null;
    el("telemetryUploadStatus").textContent = telemetryFile.files?.[0]
      ? `${telemetryFile.files[0].name} ready to validate`
      : "Select at least 14 days of BMS telemetry.";
  });
  el("revertBtn").addEventListener("click", () => window.location.reload());

  el("loadDemoBtn").addEventListener("click", async () => {
    const button = el("loadDemoBtn");
    button.disabled = true;
    button.textContent = "VERIFYING SAVED RUN…";
    try {
      const demo = await FortyCoolAPI.getVerifiedDemo();
      const runId = demo.run.run_id;
      FortyCoolAPI.recordRunLocally(runId, {
        site_name: "Verified ThermalDrift Evidence Demo",
        telemetry_source: demo.operations_data_class || "not_available",
      });
      log(`[SYS] Verified saved run ${runId}: FortyGuard ${demo.thermal_years[0]}-${demo.thermal_years.at(-1)} + Dynamic World`, "text-status-observed");
      // No `verified` flag in the URL. The command centre asks the server
      // whether this specific run passed the gates.
      window.location.href = `command_center.html?run_id=${encodeURIComponent(runId)}`;
    } catch (error) {
      // A fresh deployment has no stored verified run: the qualifying run is
      // produced by a live keyed analysis and lives in the gitignored local
      // database, so a clean clone always answers 404 here. Say that plainly
      // instead of printing a raw HTTP error.
      const missing = /returned 404/.test(error.message);
      log(
        missing
          ? "[SYS] No verified run is stored on this deployment. A verified run is created by a live analysis with FortyGuard and Earth Engine credentials; the local database that holds it is not committed to the repository."
          : `[ERR] ${error.message}`,
        missing ? "text-safety-warning" : "text-safety-critical"
      );
      button.disabled = false;
      button.innerHTML = '<span class="material-symbols-outlined text-[17px]">verified</span> LOAD VERIFIED THERMALDRIFT DEMO';
    }
  });

  el("analyzeBtn").addEventListener("click", async () => {
    const button = el("analyzeBtn");
    button.disabled = true;
    button.textContent = "STARTING AGENTS…";
    try {
      const upload = await ensureTelemetryUpload();
      const request = buildRequest(upload);
      log(`[SYS] Starting complete analysis for ${request.site.name}…`, "text-status-observed");
      const job = await FortyCoolAPI.startRun(request);
      if (!job.run_id) throw new Error("The backend accepted the request without returning a run ID.");
      FortyCoolAPI.recordRunLocally(job.run_id, {
        site_name: request.site.name,
        latitude: request.site.latitude,
        longitude: request.site.longitude,
        telemetry_source: request.telemetry.source,
      });
      log(`[SYS] Run ${job.run_id} queued. Opening agent trace…`, "text-status-observed");
      window.location.href = `command_center.html?run_id=${encodeURIComponent(job.run_id)}`;
    } catch (error) {
      log(`[ERR] ${error.message}`, "text-safety-critical");
      button.disabled = false;
      button.textContent = "ANALYZE SITE";
    }
  });

  locationInput.dataset.timezone = "America/New_York";
  refreshCoordsLabel();
  toggleTelemetryPanel();
  loadProviderStatus();
  loadCatalog();
})();
