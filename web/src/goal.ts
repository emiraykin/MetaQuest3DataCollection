// Draws the mission's goal stack (from TaskSpec) as a small 3D-looking picture on a 2D canvas.
// Shared by the headset's goal panel and the dashboard. Nothing here knows the stack's order: it draws
// whatever cups and placements the server describes, so a new mission shows up without client changes.

import type { TaskSpec } from "./protocol";

export type LevelState = "placed" | "next" | "todo";

/** Per goal entry: placed (the base cup just stands on the table; the others counted by the server), the next
 * one to place, or still to do. */
export function levelStates(task: TaskSpec, progress: number): LevelState[] {
  const placed = task.goal.map((_, k) => k <= progress);
  const next = placed.indexOf(false);
  return placed.map((p, k) => (p ? "placed" : k === next ? "next" : "todo"));
}

type V3 = [number, number, number];
const EL = 0.6, AZ = -0.65; // view: from above (34°) and from the operator's right
const D: V3 = [Math.cos(EL) * Math.cos(AZ), Math.cos(EL) * Math.sin(AZ), -Math.sin(EL)]; // view direction
const R: V3 = norm([D[1], -D[0], 0]); // screen right = d x z
const U: V3 = cross(R, D); // screen up
const RING = 48; // points per drawn circle

function norm(v: V3): V3 { const n = Math.hypot(...v); return [v[0] / n, v[1] / n, v[2] / n]; }
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function shade(rgba: number[], k: number, alpha: number) {
  const c = (i: number) => Math.round(255 * Math.min(1, rgba[i] * k));
  return `rgba(${c(0)},${c(1)},${c(2)},${alpha})`;
}

function ring(c: V3, r: number): V3[] {
  return Array.from({ length: RING }, (_, i) => {
    const a = (2 * Math.PI * i) / RING;
    return [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]] as V3;
  });
}

/** Draw the goal into the box (x, y, w, h) of a canvas context. */
export function drawGoal(g: CanvasRenderingContext2D, task: TaskSpec, states: LevelState[],
  box: { x: number; y: number; w: number; h: number }, pulse = 1) {
  const { r_bottom: rb, r_top: rt, height: ch } = task.cup;
  const items = task.goal.map((e, k) => ({ k, rgba: task.cups[e.cup].rgba, pos: e.pos as V3 }));

  // Fit the stack (and a little table around its base) into the box.
  const base = items[0].pos;
  const floor = ring(base, 1.35 * rt);
  const pts = [...floor, ...items.flatMap((it) => ring([it.pos[0], it.pos[1], it.pos[2] + ch], rt))];
  const sx = pts.map((p) => dot(p, R)), sy = pts.map((p) => -dot(p, U));
  const minX = Math.min(...sx), maxX = Math.max(...sx), minY = Math.min(...sy), maxY = Math.max(...sy);
  const scale = Math.min(box.w / (maxX - minX), box.h / (maxY - minY)) * 0.92;
  const ox = box.x + box.w / 2 - ((minX + maxX) / 2) * scale, oy = box.y + box.h / 2 - ((minY + maxY) / 2) * scale;
  const P = (p: V3): [number, number] => [ox + dot(p, R) * scale, oy - dot(p, U) * scale];
  const poly = (ps: V3[]) => { g.beginPath(); ps.forEach((p, i) => (i ? g.lineTo(...P(p)) : g.moveTo(...P(p)))); g.closePath(); };

  // A soft shadow on the table under the base cup.
  poly(floor);
  g.fillStyle = "rgba(235,235,225,0.06)";
  g.fill();

  // Bottom cup first: each nested cup covers the inside of the one below.
  for (const it of items) {
    const st = states[it.k];
    const alpha = st === "todo" ? 0.25 : 1;
    const bot = ring(it.pos, rb), top = ring([it.pos[0], it.pos[1], it.pos[2] + ch], rt);
    // Side wall: the near half of the rim and of the bottom circle (points facing the viewer).
    const near = (i: number) => dot([Math.cos((2 * Math.PI * i) / RING), Math.sin((2 * Math.PI * i) / RING), 0], D) < 0;
    const idx = Array.from({ length: RING }, (_, i) => i);
    const start = idx.find((i) => near(i) && !near((i + RING - 1) % RING)) ?? 0;
    const front = idx.map((i) => (start + i) % RING).filter(near);
    const side = () => poly([...front.map((i) => bot[i]), ...front.slice().reverse().map((i) => top[i])]);
    side();
    g.fillStyle = shade(it.rgba, 0.72, alpha);
    g.fill();
    // Open top: the rim with the darker inside.
    poly(top);
    g.fillStyle = shade(it.rgba, 0.45, alpha);
    g.fill();
    g.lineWidth = 2;
    g.strokeStyle = shade(it.rgba, 1.15, alpha);
    g.stroke();
    // Outline: dashed for to-do cups, glowing for the next one.
    if (st === "todo") {
      g.setLineDash([6, 6]);
      g.strokeStyle = shade(it.rgba, 1.2, 0.6);
      g.lineWidth = 2;
      side();
      g.stroke();
      g.setLineDash([]);
    } else if (st === "next") {
      g.save();
      g.shadowColor = "#ffffff";
      g.shadowBlur = 10 + 10 * pulse;
      g.strokeStyle = `rgba(255,255,255,${0.6 + 0.4 * pulse})`;
      g.lineWidth = 3;
      poly(top);
      g.stroke();
      g.restore();
    }
  }
}
