// Evaluation view: win rate per level and model (first try vs after learning) and a detail table.

const SERIES = ["--series-1", "--series-2", "--series-3"]; // fixed order: the color follows the model
let loaded = false;

export async function showEval(root) {
  if (loaded) return;
  loaded = true;
  const report = await (await fetch("/api/eval")).json();
  if (!report.rows.length) {
    root.innerHTML = `<section class="card"><h2>Evaluation</h2>
      <p>No results yet. Run <code>uv run python -m maze_agent.eval</code>, then reload.</p></section>`;
    return;
  }
  const models = report.config.models;
  const color = (model) => `var(${SERIES[models.indexOf(model)]})`;
  const totalCost = report.runs.reduce((sum, r) => sum + r.cost, 0);
  root.innerHTML = `
    <section class="card">
      <h2>Evaluation <small>${report.runs.length} runs · ${report.config.repeats} chains per level ·
        up to ${report.config.max_attempts} attempts that learn from failures · $${totalCost.toFixed(2)} ·
        ${report.generated.replace("T", " ")}</small></h2>
      <div class="legend">${models.map((m) => `<span><i style="background:${color(m)}"></i>${label(m)}</span>`).join("")}
        <span><i class="solid"></i>first try</span><span><i class="pale"></i>after learning</span></div>
      <div class="bars">${levels(report.rows).map((n) => levelBars(n, report.rows, models, color)).join("")}</div>
    </section>
    <section class="card">
      <h2>Details</h2>
      <div class="table-wrap">${table(report.rows)}</div>
    </section>`;
}

function levels(rows) {
  return [...new Set(rows.map((r) => r.level))].sort((a, b) => a - b);
}

function levelBars(level, rows, models, color) {
  const bars = models.map((model) => {
    const row = rows.find((r) => r.level === level && r.model === model);
    if (!row) return "";
    const tip = `${label(model)}, level ${level}: ${pct(row.first_try)} first try, ${pct(row.with_learning)} after learning`;
    return `<div class="bar" title="${tip}">
      <div class="track">
        <div class="fill pale" style="width:${row.with_learning * 100}%;background:${color(model)}"></div>
        <div class="fill" style="width:${row.first_try * 100}%;background:${color(model)}"></div>
      </div>
      <span class="value">${pct(row.first_try)} → ${pct(row.with_learning)}</span>
    </div>`;
  });
  return `<div class="level-row"><span class="level-name">Level ${level}</span><div>${bars.join("")}</div></div>`;
}

function table(rows) {
  const head = ["Model", "Level", "First try", "After learning", "Steps (optimal)", "Lives lost", "Cost / attempt", "Cost / win"];
  const body = rows.map((r) => `<tr>
    <td>${label(r.model)}</td><td>${r.level}</td><td>${pct(r.first_try)}</td><td>${pct(r.with_learning)}</td>
    <td>${r.steps === null ? "–" : Math.round(r.steps)} (${r.optimal})</td><td>${r.lives_lost.toFixed(1)}</td>
    <td>$${r.cost_per_attempt.toFixed(4)}</td><td>${r.cost_per_win === null ? "–" : "$" + r.cost_per_win.toFixed(4)}</td>
  </tr>`);
  return `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body.join("")}</tbody></table>`;
}

function label(model) {
  return model.split(":")[1];
}

function pct(x) {
  return `${Math.round(x * 100)}%`;
}
