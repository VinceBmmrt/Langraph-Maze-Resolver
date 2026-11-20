// Canvas drawing of the maze, its hazards, the dogs and the cat.

const canvas = document.getElementById("maze");
const ctx = canvas.getContext("2d");

/** Draw one frame of `view` ({ maze, world, trail, cat, drawnCat, catFacing, dogFacing }). */
export function draw(view) {
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
  world.dogs.forEach((dog, i) => emoji("🐩", dog.pos, s, 0.75, view.dogFacing[i]));
  glow(view.drawnCat, s);
  const moving = Math.hypot(view.cat[0] - view.drawnCat[0], view.cat[1] - view.drawnCat[1]);
  emoji("🐈", view.drawnCat, s, 0.85, view.catFacing, Math.sin(Math.PI * Math.min(moving, 1)) * 0.25);
}

/** Emoji face left: flip them (-1) when moving east, keep the last facing on vertical moves. */
export function facing(from, to, current) {
  if (to[0] > from[0]) return -1;
  if (to[0] < from[0]) return 1;
  return current;
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

/** Soft accent halo that marks the hero cat. */
function glow([x, y], s) {
  const [cx, cy] = [(x + 0.5) * s, (y + 0.5) * s];
  const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, s * 0.7);
  halo.addColorStop(0, "rgba(139, 124, 246, 0.75)");
  halo.addColorStop(1, "rgba(139, 124, 246, 0)");
  ctx.fillStyle = halo;
  ctx.fillRect(cx - s, cy - s, 2 * s, 2 * s);
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
