import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";

const $ = (id) => document.getElementById(id);
const canvas = $("maze");
const ctx = canvas.getContext("2d");
const ARROWS = { north: "⬆", south: "⬇", east: "➡", west: "⬅", wait: "⏸" };
const HURT = { spikes: "spikes!", dog: "bitten!" };

const game = { levels: [], lives: 3, current: 1, cleared: new Set(), source: null };
let view = null; // { maze, world, cat, drawnCat, trail, lives, steps, tokens }

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function init() {
  const info = await getJson("/api/levels");
  Object.assign(game, { levels: info.levels, lives: info.lives });
  $("model").textContent = info.model;
  $("play").onclick = () => (game.source ? stop() : start());
  await renderGraph();
  await loadLevel(1);
  requestAnimationFrame(animate);
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
    trail: new Set([start.join()]), lives: game.lives, steps: 0, tokens: 0,
    catFacing: -1, dogFacing: data.world.dogs.map(() => 1),
  };
  $("level-title").textContent = `Level ${number}: ${data.title}`;
  $("thoughts").innerHTML = "";
  setBanner("", "");
  updateStats();
  renderLevels();
  highlight("__start__");
}

function start() {
  loadLevel(game.current).then(() => {
    game.source = new EventSource(`/api/levels/${game.current}/play`);
    game.source.addEventListener("think", (e) => onThink(JSON.parse(e.data)));
    game.source.addEventListener("act", (e) => onAct(JSON.parse(e.data)));
    game.source.addEventListener("done", stop);
    game.source.onerror = () => { stop(); setBanner("Connection lost", "lost"); };
    $("play").textContent = "■ Stop";
    highlight("think");
  });
}

function stop() {
  game.source?.close();
  game.source = null;
  $("play").textContent = "▶ Play";
}

function onThink(e) {
  highlight("act");
  view.tokens += e.tokens;
  const li = document.createElement("li");
  li.innerHTML = `<div><b>${ARROWS[e.direction]}</b> ${escape(e.thought)}</div>
    <div class="meta">step ${view.steps + 1} · ${e.tokens} tokens</div><pre>${escape(e.observation)}</pre>`;
  li.onclick = () => li.classList.toggle("open");
  $("thoughts").prepend(li);
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
  highlight("__end__");
  stop();
  finish(e.status);
}

/** Emoji face left: flip them (-1) when moving east, keep the last facing on vertical moves. */
function facing(from, to, current) {
  if (to[0] > from[0]) return -1;
  if (to[0] < from[0]) return 1;
  return current;
}

function finish(status) {
  if (status !== "won") {
    const why = status === "dead" ? "Out of lives" : "Out of steps";
    return setBanner(`${why}. Press Play to retry.`, "lost");
  }
  game.cleared.add(game.current);
  renderLevels();
  const next = game.current + 1;
  if (next > game.levels.length) return setBanner("🏆 All levels cleared!", "won");
  setBanner(`Level cleared in ${view.steps} steps!`, "won");
  if ($("auto").checked) setTimeout(() => loadLevel(next).then(start), 2500);
  else game.current = next;
}

function setBanner(text, kind) {
  $("banner").textContent = text;
  $("banner").className = kind;
}

function updateStats() {
  $("hearts").textContent = "❤️".repeat(view.lives) + "🖤".repeat(game.lives - view.lives);
  $("steps").textContent = view.steps;
  $("tokens").textContent = view.tokens.toLocaleString();
}

function escape(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------- drawing ----------

function animate() {
  if (view) {
    view.drawnCat = view.drawnCat.map((v, i) => v + (view.cat[i] - v) * 0.25);
    draw();
  }
  requestAnimationFrame(animate);
}

function draw() {
  const { maze, world } = view;
  const s = canvas.width / maze.size;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = cssVar("--trail");
  for (const key of view.trail) {
    const [x, y] = key.split(",").map(Number);
    ctx.fillRect(x * s, y * s, s, s);
  }
  drawWalls(maze, s);
  world.spikes.forEach(([x, y]) => drawSpikes(x, y, s));
  emoji("🏁", maze.goal, s);
  if (world.key) emoji("🔑", world.key, s);
  if (world.door) emoji(world.has_key ? "🔓" : "🚪", world.door, s);
  world.dogs.forEach((dog, i) => emoji("🐕", dog.pos, s, 0.75, view.dogFacing[i]));
  const moving = Math.hypot(view.cat[0] - view.drawnCat[0], view.cat[1] - view.drawnCat[1]);
  emoji("🐈", view.drawnCat, s, 0.85, view.catFacing, Math.sin(Math.PI * Math.min(moving, 1)) * 0.25);
}

function drawWalls(maze, s) {
  ctx.strokeStyle = cssVar("--wall");
  ctx.lineWidth = Math.max(2, s / 14);
  ctx.lineCap = "round";
  ctx.beginPath();
  maze.cells.forEach((row, y) => row.forEach((open, x) => {
    const [l, t, r, b] = [x * s, y * s, (x + 1) * s, (y + 1) * s];
    if (!open.includes("north")) { ctx.moveTo(l, t); ctx.lineTo(r, t); }
    if (!open.includes("west")) { ctx.moveTo(l, t); ctx.lineTo(l, b); }
    if (x === maze.size - 1 && !open.includes("east")) { ctx.moveTo(r, t); ctx.lineTo(r, b); }
    if (y === maze.size - 1 && !open.includes("south")) { ctx.moveTo(l, b); ctx.lineTo(r, b); }
  }));
  ctx.stroke();
}

function drawSpikes(x, y, s) {
  ctx.fillStyle = "#9aa3b5";
  const base = (y + 0.75) * s;
  for (let i = 0; i < 3; i++) {
    const left = x * s + s * (0.2 + i * 0.2);
    ctx.beginPath();
    ctx.moveTo(left, base);
    ctx.lineTo(left + s * 0.1, (y + 0.3) * s);
    ctx.lineTo(left + s * 0.2, base);
    ctx.fill();
  }
}

/** Draw an emoji centred on a cell; `flip` -1 mirrors it, `hop` lifts it (in cells). */
function emoji(char, [x, y], s, scale = 0.65, flip = 1, hop = 0) {
  ctx.save();
  ctx.translate((x + 0.5) * s, (y + 0.5 - hop) * s);
  ctx.scale(flip, 1);
  ctx.fillStyle = "#fff";
  ctx.font = `${s * scale}px serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(char, 0, 0);
  ctx.restore();
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

init();
