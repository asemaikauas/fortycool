(function () {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run_id") || localStorage.getItem("fortycool_last_run_id");
  const el = (id) => document.getElementById(id);
  const seenTrace = new Set();
  let map = null;
  let mapLayer = null;
  let runReady = false;

  const dataClassStyles = {
    observed: "text-status-observed",
    uploaded: "text-safety-normal",
    simulated: "text-status-simulated",
    inferred: "text-status-inferred",
    assumed: "text-status-assumed",
    reported: "text-on-surface-variant",
  };

  function humanize(value) {
    return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function signedMetric(metric, suffix = "") {
    if (!metric) return "—";
    const value = Number(metric.value);
    const sign = Number.isFinite(value) && value > 0 ? "+" : "";
    return `${sign}${FortyCoolAPI.formatNumber(metric.value, 3)}${suffix || ` ${metric.unit}`}`;
  }

  // Grade plus interval, never a hand-set score rendered as a percentage. The
  // confidence field is a literal chosen in code; printing it as "56%" beside a
  // dollar figure reads to an investment committee as a probability.
  function provenanceLine(metric, fallback) {
    if (!metric) return fallback;
    const parts = [humanize(metric.data_class)];
    if (metric.evidence_grade) parts.push(`evidence ${metric.evidence_grade}`);
    if (metric.interval_low !== null && metric.interval_low !== undefined) {
      parts.push(
        `95% CI ${FortyCoolAPI.formatNumber(metric.interval_low, 3)} to ${FortyCoolAPI.formatNumber(metric.interval_high, 3)}`
      );
    }
    return parts.join(" · ");
  }

  function appendTraceStep(step) {
    const key = `${step.timestamp || ""}|${step.agent || ""}|${step.action || ""}`;
    if (seenTrace.has(key)) return;
    seenTrace.add(key);
    const status = String(step.status || "processing").toLowerCase();
    const isProblem = ["blocked", "failed", "unavailable", "error"].includes(status);
    const isWarning = status.includes("warning") || status === "insufficient_data";
    const dotColor = isProblem ? "bg-safety-critical" : isWarning ? "bg-safety-warning" : "bg-status-observed";
    const labelColor = isProblem ? "text-safety-critical" : isWarning ? "text-safety-warning" : "text-status-observed";
    const details = step.details && Object.keys(step.details).length
      ? Object.entries(step.details).map(([keyName, value]) => `${keyName}: ${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · ")
      : "";
    const div = document.createElement("div");
    div.className = "flex gap-4 relative z-10";
    div.innerHTML = `
      <div class="w-5 h-5 rounded-full bg-surface border-2 border-outline-variant flex items-center justify-center mt-0.5"><div class="w-2 h-2 rounded-full ${dotColor}"></div></div>
      <div class="min-w-0">
        <div class="text-label-sm font-label-sm ${labelColor} uppercase">${FortyCoolAPI.escapeHTML(status)}</div>
        <div class="text-body-md font-body-md text-primary">${FortyCoolAPI.escapeHTML(humanize(step.agent))}</div>
        <div class="text-label-sm font-label-sm text-on-surface-variant mt-1">${FortyCoolAPI.escapeHTML(step.action || "Agent step")}</div>
        ${details ? `<div class="text-[10px] font-label-sm text-outline mt-1 break-words">${FortyCoolAPI.escapeHTML(details)}</div>` : ""}
      </div>`;
    el("agentTraceContainer").appendChild(div);
  }

  function openTraceStream() {
    const stream = FortyCoolAPI.streamRunJob(runId);
    let terminalReceived = false;
    stream.addEventListener("trace", (event) => {
      try {
        appendTraceStep(JSON.parse(event.data));
      } catch (error) {
        console.warn("Could not parse trace event", error);
      }
    });
    stream.addEventListener("terminal", (event) => {
      terminalReceived = true;
      stream.close();
      const terminal = JSON.parse(event.data);
      if (terminal.state === "failed") {
        el("runStatusBadge").textContent = `Run failed: ${terminal.error || "unknown error"}`;
        return;
      }
      loadFinalRun();
    });
    stream.onerror = () => {
      stream.close();
      if (!terminalReceived && !runReady) pollUntilTerminal();
    };
  }

  async function pollUntilTerminal() {
    for (let attempt = 0; attempt < 90 && !runReady; attempt += 1) {
      try {
        const job = await FortyCoolAPI.getRunJobStatus(runId);
        el("runStatusBadge").textContent = `Run ${runId.slice(0, 8)} · ${humanize(job.state)} · ${job.event_count} events`;
        if (job.state === "completed") return loadFinalRun();
        if (job.state === "failed") {
          el("runStatusBadge").textContent = `Run failed: ${job.error || "unknown error"}`;
          return;
        }
      } catch (_) {
        // A persisted run may exist even if the in-memory job record expired.
        try {
          await loadFinalRun();
          return;
        } catch (_) {
          // Keep waiting while the run is in flight.
        }
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
  }

  function temperatureColor(value, min, max) {
    const ratio = max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
    return `hsl(${220 - ratio * 220} 88% 52%)`;
  }

  function renderMap(run) {
    const heatmapChart = run.chartsById.fortyguard_heatmap;
    const controlChart = run.chartsById.regional_control_map;
    const featureCollection = heatmapChart?.data?.[0]?.feature_collection;
    const empty = el("heatmapEmptyState");
    const tag = el("heatmapStatusTag");
    const mapDataClass = heatmapChart?.data_class || controlChart?.data_class;
    tag.textContent = mapDataClass ? `${humanize(mapDataClass)} · 60 M` : "NO SPATIAL DATA";
    tag.className = `px-2 py-1 text-label-sm font-label-sm border-l-2 bg-surface-container-high ${dataClassStyles[mapDataClass] || "text-on-surface-variant"}`;

    if (!window.L || (!featureCollection && !controlChart?.data?.length)) {
      empty.style.display = "flex";
      empty.textContent = "Spatial map is available when the provider returns GeoJSON.";
      return;
    }
    empty.style.display = "none";
    if (!map) {
      map = L.map("heatmapTiles", { zoomControl: true, attributionControl: true });
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap",
      }).addTo(map);
    }
    if (mapLayer) map.removeLayer(mapLayer);
    if (featureCollection) {
      const temperatures = (featureCollection.features || [])
        .map((feature) => Number(feature.properties?.average_temperature))
        .filter(Number.isFinite);
      const min = Math.min(...temperatures);
      const max = Math.max(...temperatures);
      mapLayer = L.geoJSON(featureCollection, {
        style: (feature) => {
          const value = Number(feature.properties?.average_temperature);
          return {
            color: "rgba(255,255,255,.28)",
            weight: 0.4,
            fillColor: Number.isFinite(value) ? temperatureColor(value, min, max) : "transparent",
            fillOpacity: Number.isFinite(value) ? 0.72 : 0,
          };
        },
        onEachFeature: (feature, layer) => {
          const value = feature.properties?.average_temperature;
          if (value !== undefined) layer.bindTooltip(`${FortyCoolAPI.formatNumber(value, 2)}°C`);
        },
      }).addTo(map);
    } else {
      const group = L.featureGroup();
      controlChart.data.forEach((point) => {
        const color = point.role === "site" ? "#0070f3" : "#f5a623";
        L.circleMarker([point.latitude, point.longitude], { radius: point.role === "site" ? 8 : 6, color, fillOpacity: 0.82 })
          .bindTooltip(`${point.location} · ${humanize(point.role)} · match ${FortyCoolAPI.formatNumber(point.similarity_score, 3)}`)
          .addTo(group);
      });
      mapLayer = group.addTo(map);
    }
    const bounds = mapLayer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18] });
    window.setTimeout(() => map.invalidateSize(), 0);
  }

  function linePath(values, min, max) {
    if (!values?.length) return "";
    const span = max - min || 1;
    return values.map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 100;
      const y = 95 - ((value - min) / span) * 90;
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    }).join(" ");
  }

  function renderTemperatureTimeline(run) {
    const chart = run.chartsById.temperature_forecast_12h;
    if (!chart?.data?.length) return;
    const site = chart.data.map((item) => Number(item.site_temperature_c));
    const control = chart.data.map((item) => Number(item.control_temperature_c));
    const values = [...site, ...control].filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    el("coolingObservedPath").setAttribute("d", linePath(site, min, max));
    el("coolingSimulatedPath").setAttribute("d", linePath(control, min, max));
  }

  function renderKpis(run) {
    const drift = run.metricsById.local_thermal_drift_c;
    const driftStatus = run.metricsById.thermal_drift_status;
    const hours = run.metricsById.temperature_eligible_hours_lost;
    const built = run.metricsById.local_excess_built_surface_change_percentage_points
      || run.metricsById.built_surface_change_percentage_points;
    const npv = run.metricsById.thermal_drift_npv;
    // A withheld drift is a result, not a blank. Showing an em dash where the
    // gate refused to publish a number reads as "still loading".
    el("kpiDrift").textContent = drift ? signedMetric(drift) : (driftStatus ? "Withheld" : "—");
    el("kpiDriftClass").textContent = drift
      ? provenanceLine(drift, "Awaiting analysis")
      : (driftStatus ? "Not identified from observed data" : "Awaiting analysis");
    el("kpiDriftClass").className = `text-label-sm font-label-sm ${dataClassStyles[drift?.data_class] || "text-on-surface-variant"}`;
    el("kpiHours").textContent = hours ? FortyCoolAPI.formatNumber(hours.value, 0) : "—";
    el("kpiHoursClass").textContent = provenanceLine(hours, "hours / year");
    el("kpiBuilt").textContent = built ? signedMetric(built, " pp") : "—";
    el("kpiNpv").textContent = npv ? `$${FortyCoolAPI.formatNumber(npv.value, 0)}` : "—";
    el("kpiNpvClass").textContent = provenanceLine(npv, "Indicative NPV");
  }

  // Warnings and assumptions used to appear on this screen only as counts.
  function renderCaveats(run) {
    const panel = el("caveatPanel");
    const list = el("caveatList");
    const warnings = run.warnings || [];
    const assumptions = run.assumptions || [];
    if (!warnings.length && !assumptions.length) {
      panel.classList.add("hidden");
      return;
    }
    panel.classList.remove("hidden");
    el("caveatHeading").textContent =
      `What this run does not establish (${warnings.length} ${warnings.length === 1 ? "warning" : "warnings"})`;
    const COLLAPSED = 3;
    let expanded = false;
    const paint = () => {
      const shown = expanded ? warnings : warnings.slice(0, COLLAPSED);
      list.innerHTML = shown
        .map((item) => `<li>${FortyCoolAPI.escapeHTML(item)}</li>`)
        .join("");
      const assumptionBlock = el("assumptionBlock");
      if (expanded && assumptions.length) {
        assumptionBlock.classList.remove("hidden");
        el("assumptionHeading").textContent =
          `Unconfirmed assumptions (${assumptions.length})`;
        el("assumptionList").innerHTML = assumptions
          .map(
            (item) =>
              `<li>${FortyCoolAPI.escapeHTML(item.field)} = ${FortyCoolAPI.escapeHTML(item.value)} — ${FortyCoolAPI.escapeHTML(item.reason)}</li>`
          )
          .join("");
      } else {
        assumptionBlock.classList.add("hidden");
      }
      const hidden = warnings.length - shown.length;
      el("caveatToggle").textContent = expanded
        ? "Show less"
        : `Show all (${hidden} more, ${assumptions.length} assumptions)`;
      el("caveatToggle").classList.toggle(
        "hidden",
        warnings.length <= COLLAPSED && !assumptions.length
      );
    };
    el("caveatToggle").onclick = () => {
      expanded = !expanded;
      paint();
    };
    paint();
  }

  function renderRecommendation(run) {
    const recommendation = run.recommendations?.[0];
    if (!recommendation) {
      el("recommendationHeadline").textContent = "No operational recommendation produced";
      el("recommendationTag").textContent = "NO ACTION";
      return;
    }
    el("recommendationHeadline").textContent = recommendation.description;
    el("recommendationTag").textContent = humanize(recommendation.verdict);
    el("recSavings").textContent = `${FortyCoolAPI.formatNumber(recommendation.expected_savings_kwh, 1)} kWh / 12h`;
    el("recPeakReduction").textContent = `${FortyCoolAPI.formatNumber(recommendation.expected_peak_reduction_kw, 1)} kW`;
    const reviewButton = el("executeBtn");
    reviewButton.disabled = false;
    reviewButton.onclick = () => {
      window.location.href = `optimization.html?run_id=${encodeURIComponent(runId)}`;
    };
  }

  function renderSafety(run) {
    const recommendation = run.recommendations?.[0];
    const verdict = recommendation?.verdict || (run.status === "failed" ? "failed" : "insufficient_data");
    const safe = ["approved", "approved_with_warning"].includes(verdict);
    el("safetyStatus").textContent = humanize(verdict);
    el("safetyConfidence").innerHTML = `${FortyCoolAPI.escapeHTML(humanize(run.confidence_tier))} <span class="text-on-surface-variant mx-1">|</span> ${recommendation ? `${FortyCoolAPI.formatNumber(recommendation.safety_margin_c, 2)}°C` : "—"}`;
    el("safetyPanel").classList.toggle("border-l-safety-normal", safe);
    el("safetyPanel").classList.toggle("border-l-safety-critical", !safe);
  }

  function renderEvidence(run) {
    el("evidenceCount").textContent = `Evidence Chain (${run.evidence.length} records)`;
    el("evidenceUpdatedAt").textContent = `Warnings ${run.warnings.length} · Assumptions ${run.assumptions.length}`;
    const body = el("evidenceTableBody");
    body.innerHTML = "";
    run.evidence.forEach((evidence) => {
      const row = document.createElement("tr");
      row.className = "border-t border-outline-variant hover:bg-surface-variant";
      const evidenceUrl = `${FortyCoolAPI.BASE_URL}/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(evidence.id)}`;
      row.innerHTML = `
        <td class="py-2 pr-3"><a class="font-label-md text-primary hover:underline" href="${evidenceUrl}" target="_blank" rel="noopener">${FortyCoolAPI.escapeHTML(evidence.id)}</a></td>
        <td class="py-2 pr-3" title="${FortyCoolAPI.escapeHTML(evidence.description)}">${FortyCoolAPI.escapeHTML(evidence.source)}</td>
        <td class="py-2 pr-3 text-on-surface-variant">${FortyCoolAPI.escapeHTML(new Date(evidence.retrieved_at).toLocaleString())}</td>
        <td class="py-2 ${dataClassStyles[evidence.data_class] || "text-on-surface-variant"}">${FortyCoolAPI.escapeHTML(humanize(evidence.data_class))}</td>`;
      body.appendChild(row);
    });
  }

  function renderCopilot(run) {
    const button = el("copilotAskBtn");
    button.disabled = !run.isSuccessful;
    button.onclick = async () => {
      const question = el("copilotQuestion").value.trim() || "Explain the investment risk, operational impact, main cautions, and supporting evidence.";
      const answerBox = el("copilotAnswer");
      button.disabled = true;
      button.textContent = "THINKING…";
      answerBox.classList.remove("hidden");
      answerBox.textContent = "Grounding the answer in this run's evidence…";
      try {
        const response = await FortyCoolAPI.askCopilot(runId, question, el("copilotAudience").value);
        answerBox.innerHTML = `<div class="text-primary mb-2">${FortyCoolAPI.escapeHTML(response.answer)}</div>
          ${response.key_findings?.length ? `<div class="text-label-sm font-label-sm text-status-observed mt-2">FINDINGS</div><ul class="list-disc pl-5">${response.key_findings.map((item) => `<li>${FortyCoolAPI.escapeHTML(item)}</li>`).join("")}</ul>` : ""}
          ${response.cautions?.length ? `<div class="text-label-sm font-label-sm text-safety-warning mt-2">CAUTIONS</div><ul class="list-disc pl-5">${response.cautions.map((item) => `<li>${FortyCoolAPI.escapeHTML(item)}</li>`).join("")}</ul>` : ""}
          <div class="text-[10px] font-label-sm text-outline mt-2">Evidence: ${FortyCoolAPI.escapeHTML(response.evidence_ids.join(", "))}</div>`;
      } catch (error) {
        answerBox.textContent = error.message;
      } finally {
        button.disabled = false;
        button.textContent = "ASK";
      }
    };
  }

  function updateRunLinks() {
    document.querySelectorAll('a[href$=".html"]').forEach((link) => {
      if (link.getAttribute("href").includes("site_setup")) return;
      const url = new URL(link.href);
      url.searchParams.set("run_id", runId);
      if (params.get("verified") === "1") url.searchParams.set("verified", "1");
      link.href = url.href;
    });
  }

  // The badge is a claim about evidence, so only the server may make it. It
  // used to be driven by `?verified=1` in the address bar plus a localStorage
  // entry, which meant any run, including a wholly fixture one, could be shown
  // as verified by editing the URL.
  async function renderRunActions(run) {
    const memoButton = el("downloadMemoBtn");
    memoButton.href = FortyCoolAPI.memoUrl(run.run_id);
    memoButton.download = `fortycool-thermaldrift-${run.run_id.slice(0, 8)}.pdf`;
    memoButton.classList.remove("hidden");
    memoButton.classList.add("inline-flex");

    const badge = el("verifiedDemoBadge");
    badge.classList.add("hidden");
    let verification = null;
    try {
      verification = await FortyCoolAPI.getRunVerification(run.run_id);
    } catch (_) {
      return; // No verification available: show no claim at all.
    }
    if (!verification?.verified) return;
    badge.classList.remove("hidden");
    const years = verification.thermal_years;
    badge.textContent = years?.length
      ? `VERIFIED FORTYGUARD ${years[0]}-${years[years.length - 1]}`
      : "VERIFIED SAVED RUN";
    badge.title = (verification.verification_notes || []).join(" ");
  }

  async function loadFinalRun() {
    const run = await FortyCoolAPI.getRun(runId);
    runReady = true;
    const historyRecord = FortyCoolAPI.listLocalRuns().find((item) => item.run_id === runId);
    const degraded = (run.degraded_stages || []).length
      ? ` · ${run.degraded_stages.length} degraded stage(s)`
      : "";
    el("runStatusBadge").textContent =
      `${humanize(run.status)} · ${humanize(run.confidence_tier)} · ${run.warnings.length} warnings${degraded}`;
    el("siteName").textContent = historyRecord?.site_name || "FortyCool Site Analysis";
    renderRunActions(run);
    run.trace.forEach(appendTraceStep);
    renderCaveats(run);
    renderKpis(run);
    renderMap(run);
    renderTemperatureTimeline(run);
    renderRecommendation(run);
    renderSafety(run);
    renderEvidence(run);
    renderCopilot(run);
    return run;
  }

  if (!runId) {
    el("runStatusBadge").textContent = "No run loaded — start from Site Setup";
    return;
  }
  localStorage.setItem("fortycool_last_run_id", runId);
  el("runStatusBadge").textContent = `Run ${runId.slice(0, 8)} · loading`;
  updateRunLinks();
  loadFinalRun().catch(() => openTraceStream());
})();
