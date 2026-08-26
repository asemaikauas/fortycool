(function () {
  const body = document.getElementById("historyTableBody");
  const runs = FortyCoolAPI.listLocalRuns();

  if (!runs.length) return; // default "no runs yet" row stays

  body.innerHTML = "";
  runs.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-surface-container-low transition-colors group";
    tr.innerHTML = `
      <td class="px-gutter py-stack-md text-body-md font-body-md text-on-surface whitespace-nowrap">${new Date(r.started_at).toLocaleString()}</td>
      <td class="px-gutter py-stack-md text-label-md font-label-md text-on-surface-variant"><div class="text-primary">${FortyCoolAPI.escapeHTML(r.site_name || "Site analysis")}</div><div class="text-[10px] break-all">${FortyCoolAPI.escapeHTML(r.run_id)}</div></td>
      <td class="px-gutter py-stack-md" data-status-cell>
        <span class="inline-flex items-center px-2 py-0.5 border border-outline-variant text-on-surface-variant font-label-sm text-label-sm">checking...</span>
      </td>
      <td class="px-gutter py-stack-md text-label-md font-label-md" data-confidence-cell>—</td>
      <td class="px-gutter py-stack-md text-right">
        <a class="inline-flex items-center gap-unit px-2 py-1 border border-outline-variant text-on-surface font-label-sm text-label-sm hover:border-primary hover:text-primary transition-colors bg-surface"
           href="command_center.html?run_id=${encodeURIComponent(r.run_id)}">
          <span class="material-symbols-outlined text-[14px]">play_arrow</span> Open
        </a>
      </td>`;
    body.appendChild(tr);

    // Best-effort status refresh — fine if some of these 404 for very old/expired runs.
    FortyCoolAPI.getRun(r.run_id)
      .then((run) => {
        const statusCell = tr.querySelector("[data-status-cell]");
        const ok = run.isSuccessful;
        statusCell.innerHTML = `<span class="inline-flex items-center px-2 py-0.5 border ${
          ok ? "border-safety-normal text-safety-normal bg-safety-normal/10" : "border-safety-warning text-safety-warning bg-safety-warning/10"
        } font-label-sm text-label-sm">${FortyCoolAPI.escapeHTML(run.status || "unknown")}</span>`;
        tr.querySelector("[data-confidence-cell]").textContent = (run.confidence_tier || "—").replaceAll("_", " ");
      })
      .catch(() => {
        tr.querySelector("[data-status-cell]").innerHTML =
          `<span class="inline-flex items-center px-2 py-0.5 border border-outline-variant text-on-surface-variant font-label-sm text-label-sm">unavailable</span>`;
      });
  });
})();
