# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QuestCollect: teleoperate one simulated Franka Panda arm (MuJoCo) from a Meta Quest 3 via a WebXR page, record
demonstrations of one task (`stack_cups`: nest 3 cups from `mujoco_toy-project` into one stack) to HDF5, export to
LeRobot. The right controller/hand drives the arm (`sim.ARMS = ("r",)`); the left controller only sends commands.
Python 3.12 backend in a flat package (`questcollect/`, one module per pipeline stage), TypeScript/three.js/React
frontend (`web/`). Two human docs, keep both in sync with behaviour changes: `README.md` (users: install, controls,
data format, config table, troubleshooting) and `docs/architecture.md` (developers: tick order, threads, endpoints,
protocol, frames, "Common changes" table of files that change together).

## Commands

```bash
./scripts/fetch_assets.sh               # Franka model from mujoco_menagerie -> assets/third_party/ (gitignored)
uv sync                                 # add --extra export for lerobot
(cd web && npm ci && npm run build)     # server serves web/dist; rebuild after web changes
uv run qc doctor                        # environment checklist

uv run qc serve [--mode hands] [--config my.yaml] [--no-https]
uv run qc view | replay <ep.hdf5> | inspect <ep.hdf5>
uv run qc export [--action ee] [--include-failures] [--out data/lerobot]

uv run --group dev pytest                                   # all tests
uv run --group dev pytest tests/test_teleop.py::test_protocol_roundtrip   # single test
MUJOCO_GL=egl uv run --group dev pytest                     # also runs the LeRobot export test (needs --extra export)

cd web && npm run typecheck             # tsc --noEmit (also part of npm run build)
cd web && npm run dev                   # vite on :5173, proxies /api and /ws to :8000

MUJOCO_GL=egl uv run python scripts/make_media.py   # docs/media: demo.gif, cups.png, cameras.png
```

`scripts/make_media.py` stacks all cups with the real IK + gripper stack and prints `demo progress: 2/2 ...
success: True`; it's a good end-to-end check after changing cup physics, grasping or IK. The GIF palette must
include cup-colour swatches (`gif_palette`), or the small cups quantize to grey.

Headset-free end-to-end test (emulated Quest 3 via Meta's IWER + Playwright): setup is at the top of
`web/e2e/emulated-quest.mjs` (`npm install && npx playwright install chromium && npm run build`, then
`uv run qc serve --no-https --config <yaml with a scratch record.out_dir>`, then `npm test`). It writes episodes to
the server's out_dir. `npm run screenshots` overwrites `dashboard.png`, `xr_landing.png`, `xr_headset.png` and
`xr_menu.png` in `docs/media/`; under software GL the emulated input runs at ~1 Hz, so arms may show as idle/lost.

## Architecture

Data flow: Quest Browser → `/ws/xr` binary input frames → `Runtime` sim thread (teleop → IK → physics → recorder) →
binary pose frames back to headset + JSON status/events to the dashboard (`/ws/dash`).

- **`runtime.py`** is the hub. A single real-time thread at `sim.control_hz` (50 Hz) owns the env, IK, teleop and
  recorder. `server.py` only talks to it through thread-safe methods (`submit_input`, `command`,
  `latest_pose_frame`, `status`, `previews`) and event listeners. Dashboard/REST actions go through
  `POST /api/command` → `Runtime.command` queue; don't mutate sim state from the server thread.
- **Wire protocol**: `protocol.py` and `web/src/protocol.ts` are hand-mirrored binary layouts (little endian,
  fixed float counts, 16-byte headers). Change both together and keep `test_protocol_roundtrip` passing.
- **Scene + task**: `sim.py` builds the scene programmatically with `MjSpec` (one arm, prefix `r_`, at the middle of
  the near table edge) and holds the one task (`CUPS` constants, `_cup_parts`, `_add_task`, `_randomize_task`,
  `stack_progress`, `held_cups`, `_task_success`). A cup is a thin cylinder base plus a ring of thin tilted boxes
  (MuJoCo only collides convex shapes); body frame = centre of the cup bottom. `cup_stacked` mirrors the toy
  project's `_check_cups_stacked` (upright, xy within `TOL_XY`, height above the lower cup in `NEST_DZ_RANGE`).
  `NEST_DZ` (measured rest height of a nested cup) is only used by the goal picture.
  Build area (`BUILD_AREA_*`, tape geoms in group 1, so cameras see it): `stack_progress` is 0 unless
  `in_build_area` (the base cup's rim inside); `_randomize_task` never spawns cups in it. Geom group 5
  (`OPERATOR_GROUP`) = headset/dashboard-only visuals (MuJoCo cameras render groups 0–2).
- **Goal UI is data-driven**: `sim.task_spec()` (cup shape, cups, goal placements relative to the base, area) is served at
  `GET /api/task`; `web/src/goal.ts` draws any such goal (isometric canvas) for the headset `xr/GoalPanel.ts` and the
  dashboard Goal card, using `status.progress`/`status.in_area`. Don't hard-code the stack in the client; there is
  no reference stack in the scene.
- **Gripper**: stiff tendon servo with `forcerange` = 2 × `sim.gripper_force` (each finger gets half the tendon force)
  and the target rate-limited to 50 mm/s per finger in `apply_action`, like the real Franka Hand. Elliptic friction
  cones + `impratio` 10.
  `ArmIK` compiles the same spec without the task (`build_spec(..., with_task=False)`). It is open-loop but
  `follow()` clamps it to `ik.max_lead` of the measured joints each tick, so a physically blocked arm doesn't wind
  up; the physics handles collisions.
  `scene_export.py` serializes visual geometry served at `/api/scene` + `/api/scene.bin`;
  `web/src/MujocoScene.ts` rebuilds it in three.js for both pages, which (re)load it on every websocket connect.
  Pose frames carry only per-body poses (no target ghosts). The hand has an operator-only `tcp_marker` geom
  (group 5) that the client colours per arm, red while `status.out_of_reach`, with a drop line to the table.
- **Task haptics + sounds**: `Runtime._task_feedback` emits distinct patterns (`HAPTIC_*` constants: grasp, hold tick,
  release, placed, complete); haptic events carry `repeat`/`gap_ms`, scheduled client-side in `web/src/xr/main.ts`.
  `{"type": "sound", name, arm}` events (engage/disengage, those task events, `area`, table/out_of_reach
  warnings, rate-limited by `WARN_SOUND_GAP_S`) are synthesized with WebAudio in `web/src/xr/sounds.ts` (cue names must
  match); audio is unlocked by the Enter VR click, muted via the wrist menu (localStorage).
- **Controller visuals** (`web/src/xr/visuals.ts` `HandViz`): real Touch Plus models via three's
  `XRControllerModelFactory`, assets vendored in `web/public/xr-profiles` (no CDN; `profilesList.json` trimmed to
  `meta-quest-touch-plus`, the capsule is the fallback). Per-button callouts (`HINTS`, labels mirror
  `CONTROLLER_COMMANDS` in `teleop.py`; keep them in sync) are laid out in view space each frame, plus a status chip
  and an engaged/out-of-reach emissive glow.
- **Status per arm**: `status.arms` lists the arms; per-arm maps (`engaged`, `tracked`, `held`, …) only have those
  keys and `grippers` is in `arms` order. `sources` covers both hands. The client iterates `status.arms`.
- **Teleop** (`teleop.py`): One-Euro filters, `ArmClutch` (relative motion, no jump on re-engage), controller (Grip
  press toggles the clutch) and hand (thumb-middle pinch toggles) mappers, and `Teleop` (stale-input disengage, jump rejection, workspace clamp, mode-mismatch detection; controllers without
  an arm go through `command_only` for their buttons).
  `frames.py` converts XR local-floor ↔ MuJoCo world and owns table placement/recenter.
- **Recording** (`recorder.py`): state machine (idle → countdown → recording → review → saved/discarded, auto-stop
  after success) plus HDF5 read/write; the file layout is documented in its module docstring. Images are never
  recorded: full `sim/state` per step lets the exporter re-render cameras at any resolution. `action/joint_pos` is
  `[7 joints, grip]`, grip 0 = closed, 1 = open.
- **Export**: keep all LeRobot API usage in `lerobot_export.py` (lerobot is an optional extra).
  `--randomize-visuals` applies `AppearanceRandomizer` (lights, colours, camera mounts) per episode at render time.
- **Config**: runtime tunables live in `config.yaml`, loaded into pydantic models in `config.py`. Exceptions: the task
  definition (cup shape/colors, tolerances, build area) and `ARMS` are constants at the top of `sim.py`, and
  the haptic patterns are `HAPTIC_*` constants in `runtime.py`.

## Design constraints

- Simulation-first; real robots (FR3) come later. Keep robot-specific details in `sim.py` and keep the HDF5 lossless
  (sim state, raw XR poses/hand joints) so data can be re-retargeted or re-rendered.
- The project was deliberately reduced to a single task; don't reintroduce a task registry or runtime task switching
  unless asked.
- `cli.py` defaults `MUJOCO_GL=egl`; a harmless EGLError can print at interpreter exit.
- `pyproject.toml` overrides away `evdev` (lerobot → pynput → evdev needs a C compiler); don't remove it.
