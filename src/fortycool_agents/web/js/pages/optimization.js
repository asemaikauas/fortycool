(function () {
  const runId = new URLSearchParams(window.location.search).get("run_id")
    || localStorage.getItem("fortycool_last_run_id");
  const el = (id) => document.getElementById(id);

  function humanize(value) {
    return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function points(values, min, max) {
    if (!values?.length) return "";
    const span = max - min || 1;
    return values.map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 100;
      const y = 95 - ((value - min) / span) * 90;
      return `${x},${y}`;
    }).join(" ");
  }

  function renderDecisionCard(recommendation, optimizerEvidence) {
    const div = document.createElement("div");
    const safe = ["approved", "approved_with_warning"].includes(recommendation.verdict);
    const evaluated = optimizerEvidence?.metadata?.candidate_count;
    const retained = optimizerEvidence?.metadata?.safe_candidate_count;
    div.className = `p-stack-md bg-surface border ${safe ? "border-primary" : "border-safety-critical"}`;
    div.innerHTML = `
      <div class="flex justify-between items-start mb-stack-sm"><span class="text-label-sm font-label-sm text-status-inferred bg-status-inferred/10 px-unit py-[2px] border-l border-status-inferred">${FortyCoolAPI.escapeHTML(humanize(recommendation.verdict))}</span><span class="text-label-sm font-label-sm text-on-surface-variant">Conf: ${Math.round(recommendation.confidence * 100)}%</span></div>
      <h4 class="text-body-lg font-body-lg text-primary font-medium mb-unit">${FortyCoolAPI.escapeHTML(humanize(recommendation.action))}</h4>
      <p class="text-label-sm font-label-sm text-on-surface-variant mb-stack-md">${FortyCoolAPI.escapeHTML(recommendation.description)}</p>
      <div class="grid grid-cols-2 gap-unit text-label-sm font-label-sm"><div><p class="text-on-surface-variant">Candidates</p><p class="text-primary font-medium">${evaluated ?? "—"} evaluated</p></div><div><p class="text-on-surface-variant">Safe</p><p class="text-primary font-medium">${retained ?? "—"} retained</p></div></div>`;
    return div;
  }

  function renderForecastChart(chart) {
    if (!chart?.data?.length) return;
    const baseline = chart.data.map((item) => Number(item.baseline_cooling_kw));
    const optimized = chart.data.map((item) => Number(item.optimized_cooling_kw));
    const values = [...baseline, ...optimized].filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    el("baselineChartLine").setAttribute("points", points(baseline, min, max));
    el("optimizedChartLine").setAttribute("points", points(optimized, min, max));
  }

  function renderBacktest(run) {
    const chart = run.chartsById.historical_day_backtest;
    const coolingMae = run.metricsById.cooling_model_mae_kw;
    const inletMae = run.metricsById.inlet_model_mae_c;
    if (!chart?.data?.length) return;
    const actual = chart.data.map((item) => Number(item.actual_cooling_kw));
    const predicted = chart.data.map((item) => Number(item.predicted_cooling_kw));
    const values = [...actual, ...predicted].filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    el("actualBacktestLine").setAttribute("points", points(actual, min, max));
    el("predictedBacktestLine").setAttribute("points", points(predicted, min, max));
    el("backtestSubtitle").textContent = `${chart.title} · ${humanize(chart.data_class)} · ${chart.data.length} held-out hours`;
    el("coolingMae").textContent = coolingMae ? `${FortyCoolAPI.formatNumber(coolingMae.value, 1)} kW` : "—";
    el("inletMae").textContent = inletMae ? `${FortyCoolAPI.formatNumber(inletMae.value, 3)}°C` : "—";
  }

  function renderRecommendation(run, recommendation) {
    el("selectedActionTitle").textContent = humanize(recommendation.action);
    el("selectedActionDesc").textContent = recommendation.description;
    el("selectedActionSavings").textContent = `${FortyCoolAPI.formatNumber(recommendation.expected_savings_kwh, 1)} kWh`;
    const checklist = el("constraintChecklist");
    checklist.innerHTML = "";
    (recommendation.constraints_checked || []).forEach((constraint) => {
      const item = document.createElement("div");
      item.className = "flex items-center gap-unit text-label-sm font-label-sm";
      item.innerHTML = `<span class="material-symbols-outlined text-[16px] text-safety-normal">check_circle</span><span class="text-on-surface-variant">${FortyCoolAPI.escapeHTML(humanize(constraint))}</span>`;
      checklist.appendChild(item);
    });
    if (!recommendation.constraints_checked?.length) {
      checklist.textContent = "No safe action was approved.";
    }

    const exportButton = el("executeActionBtn");
    exportButton.disabled = false;
    exportButton.onclick = () => {
      const evidence = recommendation.evidence_ids.map((id) => run.evidenceById[id]).filter(Boolean);
      const blob = new Blob([JSON.stringify({ run_id: runId, advisory_only: true, recommendation, evidence }, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `fortycool-advisory-${runId.slice(0, 8)}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    };
  }

  function updateRunLinks() {
    document.querySelectorAll('a[href$=".html"]').forEach((link) => {
      if (link.getAttribute("href").includes("site_setup")) return;
      const url = new URL(link.href);
      url.searchParams.set("run_id", runId);
      link.href = url.href;
    });
  }

  async function load() {
    try {
      const run = await FortyCoolAPI.getRun(runId);
      const historyRecord = FortyCoolAPI.listLocalRuns().find((item) => item.run_id === runId);
      el("siteLabel").textContent = historyRecord?.site_name || "FortyCool Site Analysis";
      el("systemStatus").textContent = `${humanize(run.status)} · ${humanize(run.confidence_tier)}`;
      const recommendation = run.recommendations?.[0];
      if (!recommendation) {
        el("candidateEmptyState").textContent = "The run did not produce an operational recommendation.";
        return;
      }
      const optimizerEvidence = run.evidence.find((item) => item.id.startsWith("optimizer-"));
      el("candidateEmptyState").remove();
      el("candidateList").appendChild(renderDecisionCard(recommendation, optimizerEvidence));
      renderRecommendation(run, recommendation);
      renderForecastChart(run.chartsById.baseline_vs_optimized);
      renderBacktest(run);
    } catch (error) {
      el("systemStatus").textContent = "Run not ready";
      el("candidateEmptyState").textContent = error.message;
    }
  }

  if (runId) {
    localStorage.setItem("fortycool_last_run_id", runId);
    updateRunLinks();
    load();
  }
})();
