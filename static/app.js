import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
import { draw, facing } from "./draw.js";
import { showEval } from "./eval.js";

const $ = (id) => document.getElementById(id);
const ARROWS = { north: "⬆", south: "⬇", east: "➡", west: "⬅", wait: "⏸" };
const HURT = { spikes: "spikes!", dog: "bitten!" };

const game = { levels: [], current: 1, cleared: new Set(), source: null, thread: null, live: "" };
let view = null; // the level being shown: maze, world, cat position, stats

async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function init() {
  const info = await getJson("/api/levels");
  game.levels = info.levels;
  for (const m of info.models) {
    const option = new Option(m.available ? m.label : `${m.label} (no API key)`, m.id, false, m.id === info.default_model);
    option.disabled = !m.available;
    $("model").append(option);
  }
  $("model").onchange = () => { stop(); loadLevel(game.current); };
  $("play").onclick = () => (game.source ? stop() : start());
  $("forget").onclick = forget;
  $("help").onsubmit =(e) => { e.preventDefault(); answer($("help-text").value); };
  document.querySelectorAll("[data-answer]").forEach((b) => (b.onclick = () => answer(b.dataset.answer)));
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  await renderGraph();
  await loadLevel(1);
  requestAnimationFrame(animate);
}

function showTab(tab) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("current", b.dataset.tab === tab));
  $("play-tab").hidden = tab !== "play";
  $("eval-tab").hidden = tab !== "eval";
  if (tab === "eval") showEval($("eval-tab"));
}

async function renderGraph() {
  mermaid.initialize({ startOnLoad: false, theme: "dark" });
  const { mermaid: text } = await getJson("/api/graph");
  const plain = text.split("\n").filter((line) => !line.trim().startsWith("classDef")).join("\n");
  const { svg } = await mermaid.render("graph-svg", plain);
  $("graph").innerHTML = svg;
}

function highlight(node) {
  document.querySelectorAll("#graph .node").forEach((el) => {
    el.classList.toggle("active", el.id.startsWith(`flowchart-${node}-`));
  });
}

function renderLevels() {
  $("levels").innerHTML = "";
  for (const lv of game.levels) {
    const li = document.createElement("li");
    li.textContent = `${lv.number}. ${lv.title}`;
    li.classList.toggle("current", lv.number === game.current);
    li.classList.toggle("cleared", game.cleared.has(lv.number));
    li.onclick = () => { stop(); loadLevel(lv.number); };
    $("levels").append(li);
  }
}

async function loadLevel(number) {
  const data = await getJson(`/api/levels/${number}`);
  game.current = number;
  const start = data.maze.start;
  view = {
    maze: data.maze, world: data.world, cat: start, drawnCat: [...start],
    trail: new Set([start.join()]), maxLives: data.lives, lives: data.lives, steps: 0, tokens: 0, cost: 0,
    catFacing: -1, dogFacing: data.world.dogs.map(() => 1),
  };
  $("level-title").textContent = `Level ${number}: ${data.title}`;
  $("thoughts").innerHTML = "";
  setBanner("", "");
  showHelp(null);
  updateStats();
  renderLevels();
  highlight("__start__");
  await renderLessons();
}

// ---------- running ----------

function start() {
  loadLevel(game.current).then(() => {
    const model = encodeURIComponent($("model").value);
    listen(new EventSource(`/api/levels/${game.current}/play?model=${model}`));
    highlight("think");
  });
}

/** Resume the paused graph with the player's advice (Command(resume=...) on the server). */
function answer(text) {
  if (!text.trim() || !game.thread) return;
  showHelp(null);
  addNote(`🧑 You: "${text}"`);
  listen(new EventSource(`/api/runs/${game.thread}/resume?answer=${encodeURIComponent(text)}`));
}

function listen(source) {
  game.source = source;
  const on = (name, handler) => source.addEventListener(name, (e) => handler(JSON.parse(e.data)));
  on("start", (e) => (game.thread = e.thread_id));
  on("token", onToken);
  on("think", onThink);
  on("act", onAct);
  on("ask_human", () => highlight("think"));
  on("interrupt", onInterrupt);
  on("reflect", onReflect);
  on("done", () => { source.close(); if (game.source === source) game.source = null; setPlaying(); });
  source.onerror = () => { source.close(); game.source = null; setPlaying(); };
  setPlaying();
}

function stop() {
  game.source?.close();
  game.source = null;
  showHelp(null);
  setPlaying();
}

function setPlaying() {
  $("play").textContent = game.source ? "■ Stop" : "▶ Play";
}

/** Live streaming: the LLM writes JSON token by token; show the "thought" field as it grows. */
function onToken(e) {
  highlight("think");
  let li = $("thoughts").querySelector("li.live");
  if (!li) {
    game.live = "";
    li = document.createElement("li");
    li.className = "live";
    li.innerHTML = `<div class="text"></div><div class="meta">step ${view.steps + 1} · thinking…</div>`;
    $("thoughts").prepend(li);
  }
  game.live += e.text;
  li.querySelector(".text").textContent = `💭 ${partialThought(game.live)}`;
}

function partialThought(json) {
  const match = json.match(/"thought"\s*:\s*"((?:[^"\\]|\\.)*)/);
  if (!match) return "";
  try { return JSON.parse(`"${match[1].replace(/\\$/, "")}"`); } catch { return match[1]; }
}

function onThink(e) {
  highlight("act");
  view.tokens += e.tokens;
  view.cost += e.cost;
  const li = $("thoughts").querySelector("li.live") || document.createElement("li");
  li.className = "";
  li.innerHTML = `<div><b>${ARROWS[e.direction]}</b> ${escape(e.thought)}</div>
    <div class="meta">step ${view.steps + 1} · ${e.tokens} tokens</div><pre>${escape(e.observation)}</pre>`;
  li.onclick = () => li.classList.toggle("open");
  if (!li.parentNode) $("thoughts").prepend(li);
  updateStats();
}

function onAct(e) {
  view.catFacing = facing(view.cat, e.pos, view.catFacing);
  view.dogFacing = e.world.dogs.map((dog, i) => facing(view.world.dogs[i].pos, dog.pos, view.dogFacing[i]));
  Object.assign(view, { cat: e.pos, world: e.world, lives: e.lives, steps: e.steps });
  view.trail.add(e.pos.join());
  const last = $("thoughts").firstElementChild;
  if (HURT[e.event] && last) {
    last.classList.add("hurt");
    last.querySelector(".meta").textContent += ` · ${HURT[e.event]}`;
  }
  updateStats();
  if (e.status === "running") return highlight("think");
  if (e.status === "won") {
    highlight("__end__");
    return finish();
  }
  highlight("reflect");
  setBanner(`${e.status === "dead" ? "Out of lives" : "Out of steps"}. The cat is writing a lesson…`, "lost");
}

/** Human in the loop: the graph paused in `ask_human` and waits for the player. */
function onInterrupt(e) {
  highlight("ask_human");
  showHelp(e.question);
  addNote(`🐈 "${e.question}"`);
}

function onReflect(e) {
  highlight("__end__");
  view.cost += e.cost;
  updateStats();
  addNote(`📝 Lesson learned: ${e.lesson}`);
  setBanner("Lost. Press Play to retry: the cat will remember its lesson.", "lost");
  renderLessons();
}

function finish() {
  game.cleared.add(game.current);
  renderLevels();
  const next = game.current + 1;
  if (next > game.levels.length) return setBanner("🏆 All levels cleared!", "won");
  setBanner(`Level cleared in ${view.steps} steps!`, "won");
  if ($("auto").checked) setTimeout(() => loadLevel(next).then(start), 2500);
  else game.current = next;
}

// ---------- side panels ----------

function showHelp(question) {
  $("help").hidden = !question;
  if (question) {
    $("help-question").textContent = question;
    $("help-text").value = "";
    $("help-text").focus({ preventScroll: true });
  }
}

async function renderLessons() {
  const model = encodeURIComponent($("model").value);
  const { lessons } = await getJson(`/api/lessons?level=${game.current}&model=${model}`);
  $("lessons").innerHTML = lessons.length ? "" : "<li class='empty'>No lesson yet: they are written after a failure.</li>";
  lessons.forEach((text) => $("lessons").append(Object.assign(document.createElement("li"), { textContent: text })));
  $("forget").hidden = !lessons.length;
}

async function forget() {
  const model = encodeURIComponent($("model").value);
  await getJson(`/api/lessons?level=${game.current}&model=${model}`, { method: "DELETE" });
  renderLessons();
}

function addNote(text) {
  const li = document.createElement("li");
  li.className = "note";
  li.textContent = text;
  $("thoughts").prepend(li);
}

function setBanner(text, kind) {
  $("banner").textContent = text;
  $("banner").className = kind;
}

function updateStats() {
  $("hearts").textContent = "❤️".repeat(view.lives) + "🖤".repeat(view.maxLives - view.lives);
  $("steps").textContent = view.steps;
  $("tokens").textContent = view.tokens.toLocaleString();
  $("cost").textContent = `$${view.cost.toFixed(4)}`;
}

function escape(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function animate() {
  if (view) {
    view.drawnCat = view.drawnCat.map((v, i) => v + (view.cat[i] - v) * 0.25);
    draw(view);
  }
  requestAnimationFrame(animate);
}

init();
