/* Quantization Canary - frontend
   Loads docs/data.json once, renders the home/evidence pages on
   demand, switches sections via URL hash. Vanilla JS, Chart.js
   (CDN) for the time series, jsdiff (CDN) for character-level
   diff highlighting in the drill-down. */

(function () {
  "use strict";

  const VERDICT_META = {
    green: { emoji: "🟢", label: "Stable", detail: "Fingerprint within noise floor.", cls: "status-green" },
    yellow: { emoji: "🟡", label: "Possible drift", detail: "Soft signal — watch this space.", cls: "status-yellow" },
    red: { emoji: "🔴", label: "Change detected", detail: "High-confidence change. See evidence below.", cls: "status-red" },
  };

  const SECTIONS = ["home", "evidence", "methodology"];

  let DATA = null;
  let chartInstance = null;

  // ─── boot ─────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", function () {
    setupNav();
    showSection(currentSection());
    fetch("data.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        DATA = data;
        renderHome();
        renderEvidence();
        renderFooter();
      })
      .catch((e) => {
        console.error("Failed to load data.json", e);
        renderEmpty();
      });
  });

  // ─── nav / hash routing ───────────────────────────────
  function setupNav() {
    document.querySelectorAll("nav a[data-section]").forEach((a) => {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        const target = a.dataset.section;
        location.hash = target;
        showSection(target);
      });
    });
    window.addEventListener("hashchange", () => showSection(currentSection()));
  }

  function currentSection() {
    const h = (location.hash || "#home").replace("#", "");
    return SECTIONS.includes(h) ? h : "home";
  }

  function showSection(name) {
    SECTIONS.forEach((s) => {
      const el = document.getElementById(s);
      if (el) el.classList.toggle("hidden", s !== name);
    });
    document.querySelectorAll("nav a[data-section]").forEach((a) => {
      a.classList.toggle("active", a.dataset.section === name);
    });
  }

  // ─── home ─────────────────────────────────────────────
  function renderHome() {
    if (!DATA || !DATA.latest) {
      renderEmpty();
      return;
    }

    const verdict = DATA.latest.verdict;
    const meta = VERDICT_META[verdict] || VERDICT_META.green;

    const card = document.getElementById("status-card");
    card.className = "status-card " + meta.cls;
    document.getElementById("status-emoji").textContent = meta.emoji;
    document.getElementById("status-label").textContent = meta.label;
    document.getElementById("status-detail").textContent = meta.detail;

    const lastTickDate = DATA.latest.date;
    const lastTickTime = DATA.latest.finished_at || "";
    const probesCount = DATA.n_probes;
    const baselineV = DATA.baseline_version;
    const probeV = DATA.probe_set_version;

    document.getElementById("meta-line").innerHTML =
      `Last tick <strong>${escapeHtml(lastTickDate)}</strong> ` +
      `(${escapeHtml(lastTickTime.split("T")[1] ? lastTickTime.split("T")[1].split(".")[0] + " UTC" : "")})  ` +
      ` &middot; ${probesCount} probes  ` +
      ` &middot; baseline ${escapeHtml(baselineV)}  ` +
      ` &middot; probe set ${escapeHtml(probeV)}`;

    renderChart();
  }

  function renderChart() {
    const empty = document.getElementById("chart-empty");
    const canvas = document.getElementById("similarity-chart");
    if (!DATA.history || DATA.history.length === 0) {
      empty.style.display = "block";
      canvas.style.display = "none";
      return;
    }
    empty.style.display = "none";
    canvas.style.display = "block";

    const labels = DATA.history.map((h) => h.date);
    const dValues = DATA.history.map((h) => h.D);
    const targetD = DATA.cusum && DATA.cusum.target_D !== undefined ? DATA.cusum.target_D : 0;
    const dStd = DATA.calibration_distributions && DATA.calibration_distributions.d_std
      ? DATA.calibration_distributions.d_std
      : 0;
    const upperBand = labels.map(() => targetD + dStd);
    const lowerBand = labels.map(() => Math.max(0, targetD - dStd));

    if (chartInstance) {
      chartInstance.destroy();
    }
    chartInstance = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "noise floor (+1σ)",
            data: upperBand,
            borderColor: "rgba(140,140,140,0.4)",
            backgroundColor: "rgba(140,140,140,0.12)",
            fill: "+1",
            pointRadius: 0,
            borderWidth: 1,
            borderDash: [4, 4],
          },
          {
            label: "noise floor (−1σ)",
            data: lowerBand,
            borderColor: "rgba(140,140,140,0.4)",
            backgroundColor: "rgba(140,140,140,0.12)",
            fill: false,
            pointRadius: 0,
            borderWidth: 1,
            borderDash: [4, 4],
          },
          {
            label: "D(t) — daily fingerprint distance",
            data: dValues,
            borderColor: "#0a5ec7",
            backgroundColor: "#0a5ec7",
            borderWidth: 2.5,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.18,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 13 } } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.dataset.label.startsWith("D(")) {
                  return `D=${ctx.parsed.y.toFixed(4)}`;
                }
                return null;
              },
            },
          },
        },
        scales: {
          x: { ticks: { font: { size: 12 } } },
          y: {
            beginAtZero: true,
            ticks: { font: { size: 12 } },
            title: { display: true, text: "Normalized edit distance", font: { size: 13 } },
          },
        },
      },
    });
  }

  // ─── evidence ─────────────────────────────────────────
  function renderEvidence() {
    const tbody = document.getElementById("evidence-tbody");
    const empty = document.getElementById("evidence-empty");
    tbody.innerHTML = "";

    if (!DATA || !DATA.latest || !DATA.latest.per_prompt || DATA.latest.per_prompt.length === 0) {
      empty.style.display = "block";
      return;
    }
    empty.style.display = "none";

    DATA.latest.per_prompt.forEach((pp, idx) => {
      const tr = document.createElement("tr");
      const driftPct = (pp.d !== undefined && pp.d !== null) ? (pp.d * 100).toFixed(1) + "%" : "—";
      const noiseMu = pp.baseline_mu !== undefined ? (pp.baseline_mu * 100).toFixed(1) : "?";
      const noiseSig = pp.baseline_sigma !== undefined ? (pp.baseline_sigma * 100).toFixed(1) : "?";
      const noiseStr = `${noiseMu}% &pm; ${noiseSig}%`;
      const vrd = verdictFromZ(pp.z);
      tr.innerHTML =
        `<td>${escapeHtml(pp.id)}</td>` +
        `<td>${escapeHtml(pp.category)}</td>` +
        `<td class="numeric">${driftPct}</td>` +
        `<td class="numeric">${noiseStr}</td>` +
        `<td class="numeric">${formatNum(pp.z, 2)}</td>` +
        `<td class="verdict-cell ${vrd.cls}">${vrd.dot} ${vrd.label}</td>`;
      tr.dataset.idx = idx;
      tr.addEventListener("click", () => toggleDrilldown(tr, pp));
      tbody.appendChild(tr);
    });
  }

  function toggleDrilldown(row, pp) {
    const next = row.nextElementSibling;
    if (next && next.classList.contains("drilldown")) {
      next.remove();
      row.classList.remove("expanded");
      return;
    }
    row.classList.add("expanded");

    const dd = document.createElement("tr");
    dd.classList.add("drilldown");
    const td = document.createElement("td");
    td.colSpan = 6;

    const baselineRef = pp.baseline_reference || "";
    const samples = pp.samples || [];
    const driftPct = (pp.d * 100).toFixed(1);
    const noiseMuPct = pp.baseline_mu !== undefined ? (pp.baseline_mu * 100).toFixed(1) : "?";
    const noiseSigPct = pp.baseline_sigma !== undefined ? (pp.baseline_sigma * 100).toFixed(1) : "?";

    // Plain-English interpretation of the numbers
    let interpretation;
    const zAbs = Math.abs(pp.z);
    if (zAbs < 1.0) {
      interpretation = `This level of drift is <strong>well within the noise floor</strong> — normal T=0 jitter.`;
    } else if (zAbs < 2.0) {
      interpretation = `This drift is <strong>mildly unusual</strong> but still within the noise floor.`;
    } else if (zAbs < 3.0) {
      interpretation = `This drift is <strong>moderately unusual</strong> — worth watching over the next few days.`;
    } else {
      interpretation = `This drift is <strong>highly unusual</strong> — significantly outside the calibration noise floor.`;
    }
    if (pp.z < -0.5) {
      interpretation = `Today's output is <strong>closer to baseline than usual</strong> — less drift than the calibration average. Not concerning.`;
    }

    const diffHtml = samples
      .map((s, i) => {
        const html = renderInlineDiff(baselineRef, s);
        return `<div class="diff-pair">
                  <div>
                    <h4>Baseline reference</h4>
                    <div class="diff-pane">${escapeHtml(baselineRef)}</div>
                  </div>
                  <div>
                    <h4>Today's sample ${i + 1}</h4>
                    <div class="diff-pane">${html}</div>
                  </div>
                </div>`;
      })
      .join("");

    td.innerHTML =
      `<div class="drilldown-content">
        <h4>Prompt</h4>
        <div class="prompt-text">${escapeHtml(pp.prompt || "")}</div>
        <h4>What happened</h4>
        <p class="interpretation">
          Today's output is <strong>${driftPct}% different</strong> from baseline.
          During calibration, this prompt typically varied by
          <strong>${noiseMuPct}%</strong> (&pm;${noiseSigPct}%).
          ${interpretation}
        </p>
        <details class="raw-stats">
          <summary>Raw stats</summary>
          <div class="prompt-text">μ = ${formatNum(pp.baseline_mu, 5)} &nbsp; σ = ${formatNum(pp.baseline_sigma, 5)} &nbsp; d = ${formatNum(pp.d, 5)} &nbsp; z = ${formatNum(pp.z, 3)}</div>
        </details>
        <h4>Diff: baseline reference vs today's samples</h4>
        ${diffHtml}
        <button class="replay-button" data-id="${escapeHtml(pp.id)}">Copy "replay" snippet</button>
      </div>`;

    td.querySelector(".replay-button").addEventListener("click", function (e) {
      e.stopPropagation();
      copyReplaySnippet(pp);
      this.textContent = "Copied!";
      setTimeout(() => (this.textContent = 'Copy "replay" snippet'), 1400);
    });

    dd.appendChild(td);
    row.parentNode.insertBefore(dd, row.nextSibling);
  }

  function renderInlineDiff(baseline, sample) {
    if (typeof Diff === "undefined" || !Diff.diffChars) {
      // Library missing - fall back to plain text
      return escapeHtml(sample);
    }
    const parts = Diff.diffChars(baseline, sample);
    return parts
      .map((p) => {
        const text = escapeHtml(p.value);
        if (p.added) return `<span class="diff-add">${text}</span>`;
        if (p.removed) return `<span class="diff-del">${text}</span>`;
        return text;
      })
      .join("");
  }

  function copyReplaySnippet(pp) {
    const snippet = `# Replay this probe yourself
# pip install anthropic
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="${DATA.model}",
    max_tokens=${pp.max_tokens || 500},
    temperature=0,
    messages=[{"role": "user", "content": ${JSON.stringify(pp.prompt || "")}}],
)
print(response.content[0].text)`;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(snippet).catch(() => {});
    }
  }

  // ─── empty state ──────────────────────────────────────
  function renderEmpty() {
    const card = document.getElementById("status-card");
    card.className = "status-card status-loading";
    document.getElementById("status-emoji").textContent = "⏳";
    document.getElementById("status-label").textContent = "No data yet";
    document.getElementById("status-detail").textContent =
      "Run the worker once to populate docs/data.json.";
    document.getElementById("meta-line").textContent = "";
    document.getElementById("evidence-empty").style.display = "block";
  }

  function renderFooter() {
    const f = document.getElementById("footer-meta");
    if (!DATA) return;
    f.textContent =
      `data.json generated ${DATA.generated_at || "?"} ` +
      `· model ${DATA.model || "?"} ` +
      `· baseline ${DATA.baseline_version || "?"}`;
  }

  // ─── small helpers ────────────────────────────────────
  function escapeHtml(s) {
    if (s === undefined || s === null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNum(n, digits) {
    if (n === undefined || n === null || isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  }

  // Per-prompt status uses standard z-score thresholds (2.0 / 3.0),
  // NOT the aggregate yellow_z / red_z from calibration. The aggregate
  // thresholds are calibrated for the mean-of-30-z-scores, which has a
  // much tighter distribution than individual z-scores. Using them here
  // would cause many individual prompts to show "Anomaly" even on
  // perfectly stable days, because individual z > 0.35 is commonplace
  // while aggregate Z > 0.35 is rare.
  //
  // With 30 prompts, ~1-2 prompts above |z|=2.0 per day is expected
  // from normal jitter — visitors should not be alarmed by a couple of
  // yellow rows.
  function verdictFromZ(z) {
    const az = Math.abs(z);
    if (az >= 3.0) return { dot: "🔴", label: "Anomaly", cls: "v-red" };
    if (az >= 2.0) return { dot: "🟡", label: "Watch", cls: "v-yellow" };
    return { dot: "🟢", label: "Stable", cls: "v-green" };
  }
})();
