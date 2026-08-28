(function () {
  const runId = new URLSearchParams(window.location.search).get("run_id")
    || localStorage.getItem("fortycool_last_run_id");
  const el = (id) => document.getElementById(id);

  function humanize(value) {
    return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
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

  function renderTemperatureChart(chart) {
    if (!chart?.data?.length) return;
    const site = chart.data.map((item) => Number(item.site_mean_temperature_c));
    const control = chart.data.map((item) => Number(item.control_mean_temperature_c));
    const values = [...site, ...control].filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    el("siteTrendPath").setAttribute("d", linePath(site, min, max));
    el("controlTrendPath").setAttribute("d", linePath(control, min, max));
    el("driftYears").innerHTML = chart.data.map((item) => `<span>${FortyCoolAPI.escapeHTML(item.year)}</span>`).join("");
  }

  function renderLandCoverChart(chart) {
    if (!chart?.data?.length) return;
    const site = chart.data.map((item) => Number(item.site_built_surface_percent));
    const control = chart.data.map((item) => Number(item.control_built_surface_percent));
    const values = [...site, ...control].filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    el("siteBuiltPath").setAttribute("d", linePath(site, min, max));
    el("controlBuiltPath").setAttribute("d", linePath(control, min, max));
    el("landCoverClass").textContent = humanize(chart.data_class);
  }

  function metricText(metric, digits = 3, positiveSign = false) {
    if (!metric) return "—";
    const numeric = Number(metric.value);
    const sign = positiveSign && Number.isFinite(numeric) && numeric > 0 ? "+" : "";
    return `${sign}${FortyCoolAPI.formatNumber(metric.value, digits)}`;
  }

  function renderEvidence(run, metrics) {
    const evidenceIds = [...new Set(metrics.flatMap((metric) => metric?.evidence_ids || []))];
    if (!evidenceIds.length) return;
    const links = evidenceIds.map((evidenceId) => {
      const evidence = run.evidenceById[evidenceId];
      const url = `${FortyCoolAPI.BASE_URL}/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(evidenceId)}`;
      const label = evidence?.activity_id ? `${evidenceId} · activity ${evidence.activity_id}` : evidenceId;
      return `<a class="block text-primary hover:underline mt-2 break-all" href="${url}" target="_blank" rel="noopener">${FortyCoolAPI.escapeHTML(label)}</a>`;
    }).join("");
    el("thermalEvidence").innerHTML = `<div class="text-primary">AUDITABLE EVIDENCE</div>${links}`;
  }

  function updateRunLinks() {
    document.querySelectorAll('a[href$=".html"]').forEach((link) => {
      if (link.getAttribute("href").includes("site_setup")) return;
      const url = new URL(link.href);
      url.searchParams.set("run_id", runId);
      if (new URLSearchParams(window.location.search).get("verified") === "1") {
        url.searchParams.set("verified", "1");
      }
      link.href = url.href;
    });
  }

  async function load() {
    if (!runId) return;
    try {
      const run = await FortyCoolAPI.getRun(runId);
      const historyRecord = FortyCoolAPI.listLocalRuns().find((item) => item.run_id === runId);
      el("siteLabel").textContent = historyRecord?.site_name || "FortyCool Site Analysis";
      const memoButton = el("downloadMemoBtn");
      memoButton.href = FortyCoolAPI.memoUrl(runId);
      memoButton.download = `fortycool-thermaldrift-${runId.slice(0, 8)}.pdf`;
      memoButton.classList.remove("hidden");
      memoButton.classList.add("inline-flex");
      const metrics = run.metricsById;
      const drift = metrics.local_thermal_drift_c;
      const driftRate = metrics.local_drift_rate_c_per_year;
      const hours = metrics.temperature_eligible_hours_lost;
      const npv = metrics.thermal_drift_npv;
      const controlScore = metrics.regional_control_match_score;
      const controlStatus = metrics.regional_control_match_status;
      const builtChange = metrics.local_excess_built_surface_change_percentage_points
        || metrics.built_surface_change_percentage_points;
      const association = metrics.thermal_land_cover_association_correlation;
      const thermalChart = run.chartsById.thermal_drift_timeseries;
      const landCoverChart = run.chartsById.historical_land_cover_timeseries;

      // The comparison is named from the run, not hardcoded. This line asserted
      // "matched regional controls" even on runs where the candidates were
      // rejected and the disclosed local outer ring was used instead, while the
      // very next line printed that rejection.
      const controlMethod = String(controlStatus?.value || "").toLowerCase();
      const controlName =
        controlMethod === "accepted"
          ? "matched regional controls"
          : "the disclosed local outer ring";
      const interval =
        drift && drift.interval_low !== null && drift.interval_low !== undefined
          ? ` (95% CI ${FortyCoolAPI.formatNumber(drift.interval_low, 3)} to ${FortyCoolAPI.formatNumber(drift.interval_high, 3)})`
          : "";
      el("driftSubtitle").textContent = drift
        ? `Site vs ${controlName} · ${metricText(drift, 3, true)}°C local drift${interval} · ${humanize(run.confidence_tier)} confidence tier`
        : (metrics.thermal_drift_status
            ? `Drift withheld: ${metrics.thermal_drift_status.caveats?.[0] || "not identified from observed data"}`
            : "Site vs control, drift was not computed");
      el("thermalDataClass").textContent = thermalChart ? `${humanize(thermalChart.data_class)} · ${thermalChart.data.length} YEARS` : "NO THERMAL SERIES";
      el("driftValue").textContent = metricText(driftRate, 3, true);
      el("eligibleHours").textContent = metricText(hours, 0);
      el("financialExposure").textContent = npv ? `$${FortyCoolAPI.formatNumber(npv.value, 0)}` : "—";
      el("financialClass").textContent = npv
        ? [
            humanize(npv.data_class),
            npv.evidence_grade ? `evidence ${npv.evidence_grade}` : null,
            npv.interval_low !== null && npv.interval_low !== undefined
              ? `95% CI $${FortyCoolAPI.formatNumber(npv.interval_low, 0)} to $${FortyCoolAPI.formatNumber(npv.interval_high, 0)}`
              : null,
          ]
            .filter(Boolean)
            .join(" · ")
        : "Not calculated";
      el("controlScore").textContent = metricText(controlScore, 3);
      el("controlStatus").textContent = controlStatus ? `${humanize(controlStatus.value)} quality gate` : "Local outer ring used";
      el("builtChange").textContent = builtChange ? `${metricText(builtChange, 3, true)} pp` : "—";
      el("association").textContent = association
        ? `${metricText(association, 3, true)} (95% CI ${FortyCoolAPI.formatNumber(association.interval_low, 3)} to ${FortyCoolAPI.formatNumber(association.interval_high, 3)})`
        : (metrics.thermal_land_cover_association_status
            ? "Withheld: too few observed years, or an interpolated series"
            : "Not enough paired years");

      renderTemperatureChart(thermalChart);
      renderLandCoverChart(landCoverChart);
      renderEvidence(run, [drift, driftRate, hours, npv, controlScore, builtChange, association]);
    } catch (error) {
      el("driftSubtitle").textContent = `Run unavailable: ${error.message}`;
    }
  }

  if (runId) {
    localStorage.setItem("fortycool_last_run_id", runId);
    updateRunLinks();
    load();
  }
})();
