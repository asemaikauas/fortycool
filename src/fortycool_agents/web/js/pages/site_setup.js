(function () {
  const el = (id) => document.getElementById(id);
  const latInput = el("latInput");
  const lonInput = el("lonInput");
  const locationInput = el("locationInput");
  const sampleSitePicker = el("sampleSitePicker");
  const telemetrySource = el("telemetrySource");
  const telemetryFile = el("telemetryFile");
  let telemetryUpload = null;
  let siteMap = null;
  let siteMarker = null;
  let analysisArea = null;

  const ANALYSIS_RADIUS_METERS = 1500;

  function showNotice(message) {
    const notice = el("setupNotice");
    notice.textContent = message;
    notice.classList.remove("hidden");
  }

  function clearNotice() {
    const notice = el("setupNotice");
    notice.textContent = "";
    notice.classList.add("hidden");
  }

  function refreshCoordsLabel() {
    el("coordsLabel").textContent = `LAT: ${latInput.value || "—"} | LNG: ${lonInput.value || "—"}`;
  }

  function inputCoordinates() {
    const latitude = Number.parseFloat(latInput.value);
    const longitude = Number.parseFloat(lonInput.value);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
    return [latitude, longitude];
  }

  function markerLabel() {
    return locationInput.value.trim() || "Selected site";
  }

  function syncMapFromInputs({ recenter = false } = {}) {
    const coordinates = inputCoordinates();
    if (!siteMap || !coordinates) return;
    siteMarker.setLatLng(coordinates);
    siteMarker.unbindTooltip().bindTooltip(markerLabel(), {
      direction: "top",
      offset: [0, -10],
    });
    analysisArea.setLatLng(coordinates);
    if (recenter) siteMap.flyTo(coordinates, Math.max(siteMap.getZoom(), 13));
  }

  function initializeSiteMap() {
    const coordinates = inputCoordinates() || [39.01, -77.46];
    if (!window.L) {
      showNotice("The map could not load. You can still enter coordinates manually.");
      return;
    }

    siteMap = L.map("siteMap", {
      zoomControl: true,
      attributionControl: true,
    }).setView(coordinates, 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(siteMap);
    siteMap.attributionControl.setPrefix(false);

    analysisArea = L.circle(coordinates, {
      radius: ANALYSIS_RADIUS_METERS,
      color: "#82d9ff",
      weight: 2,
      opacity: 0.9,
      fillColor: "#36b8f4",
      fillOpacity: 0.12,
      interactive: false,
    }).addTo(siteMap);
    siteMarker = L.circleMarker(coordinates, {
      radius: 8,
      color: "#e9fbff",
      weight: 3,
      fillColor: "#00a9e8",
      fillOpacity: 1,
    }).addTo(siteMap);
    syncMapFromInputs();

    siteMap.on("click", (event) => {
      latInput.value = event.latlng.lat.toFixed(6);
      lonInput.value = event.latlng.lng.toFixed(6);
      sampleSitePicker.value = "";
      refreshCoordsLabel();
      syncMapFromInputs();
    });
    window.requestAnimationFrame(() => siteMap.invalidateSize());
  }

  function setSelectedSite(option) {
    if (!option?.dataset.lat) return;
    latInput.value = option.dataset.lat;
    lonInput.value = option.dataset.lon;
    locationInput.value = option.dataset.name || option.textContent.trim();
    locationInput.dataset.timezone = option.dataset.timezone || "America/New_York";
    refreshCoordsLabel();
    syncMapFromInputs({ recenter: true });
  }

  async function loadProviderStatus() {
    try {
      const health = await FortyCoolAPI.health();
      el("syncLabel").innerHTML = `<span class="material-symbols-outlined text-[14px]">satellite_alt</span> ${FortyCoolAPI.escapeHTML(health.thermal_provider)} + ${FortyCoolAPI.escapeHTML(health.urban_provider)}`;
      if (health.provider_error) {
        showNotice("Live-data providers are unavailable. Check the backend configuration before running an analysis.");
      }
      // Say it on the screen, before anyone runs anything. In fixture mode the
      // annual series is a fixed synthetic trend that never reads the entered
      // coordinates, so the headline figures are identical for every site.
      if (String(health.thermal_provider).startsWith("Fixture")) {
        const banner = el("fixtureModeBanner");
        if (banner) banner.classList.remove("hidden");
      }
    } catch (error) {
      el("syncLabel").textContent = "Backend unavailable";
      showNotice(`The backend is unavailable: ${error.message}`);
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
      showNotice(`The sample-site catalog could not load: ${error.message}`);
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

  latInput.addEventListener("input", () => {
    refreshCoordsLabel();
    syncMapFromInputs();
  });
  lonInput.addEventListener("input", () => {
    refreshCoordsLabel();
    syncMapFromInputs();
  });
  locationInput.addEventListener("input", () => syncMapFromInputs());
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
    clearNotice();
    button.disabled = true;
    button.textContent = "VERIFYING SAVED RUN…";
    try {
      const demo = await FortyCoolAPI.getVerifiedDemo();
      const runId = demo.run.run_id;
      FortyCoolAPI.recordRunLocally(runId, {
        site_name: "Verified ThermalDrift Evidence Demo",
        telemetry_source: demo.operations_data_class || "not_available",
      });
      // No `verified` flag in the URL. The command centre asks the server
      // whether this specific run passed the gates.
      window.location.href = `command_center.html?run_id=${encodeURIComponent(runId)}`;
    } catch (error) {
      // A fresh deployment has no stored verified run: the qualifying run is
      // produced by a live keyed analysis and lives in the gitignored local
      // database, so a clean clone always answers 404 here. Say that plainly
      // instead of printing a raw HTTP error.
      const missing = /returned 404/.test(error.message);
      showNotice(
        missing
          ? "No verified run is stored on this deployment. Run a new live analysis to create one."
          : `The verified demo could not load: ${error.message}`
      );
      button.disabled = false;
      button.innerHTML = '<span class="material-symbols-outlined text-[17px]">verified</span> LOAD VERIFIED THERMALDRIFT DEMO';
    }
  });

  el("analyzeBtn").addEventListener("click", async () => {
    const button = el("analyzeBtn");
    clearNotice();
    button.disabled = true;
    button.textContent = "STARTING AGENTS…";
    try {
      const upload = await ensureTelemetryUpload();
      const request = buildRequest(upload);
      const job = await FortyCoolAPI.startRun(request);
      if (!job.run_id) throw new Error("The backend accepted the request without returning a run ID.");
      FortyCoolAPI.recordRunLocally(job.run_id, {
        site_name: request.site.name,
        latitude: request.site.latitude,
        longitude: request.site.longitude,
        telemetry_source: request.telemetry.source,
      });
      window.location.href = `command_center.html?run_id=${encodeURIComponent(job.run_id)}`;
    } catch (error) {
      showNotice(error.message);
      button.disabled = false;
      button.textContent = "ANALYZE SITE";
    }
  });

  locationInput.dataset.timezone = "America/New_York";
  refreshCoordsLabel();
  initializeSiteMap();
  toggleTelemetryPanel();
  loadProviderStatus();
  loadCatalog();
})();
