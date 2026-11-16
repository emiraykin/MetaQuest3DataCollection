"""The MuJoCo side: scene (one Franka arm + table + cups), the stack_cups task, and the environment.

World frame: z up, +x points away from the operator (towards the far table edge),
+y is the operator's left. The arm base sits on the near table edge facing +x.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import mujoco
import numpy as np
from PIL import Image

from questcollect.config import ROOT, Config, SimConfig

PANDA_XML = ROOT / "assets" / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

# One arm, driven by the operator's right hand/controller (the left controller only sends commands).
ARMS = ("r",)
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
TCP_OFFSET = 0.1034  # flange-to-fingertip-center along the hand z axis
GRIPPER_CTRL_MAX = 255.0  # Menagerie's ctrl range: 0 = closed, 255 = fully open
# Gripper like the real Franka Hand (70 N continuous, 140 N peak, 50 mm/s per finger, 80 mm stroke):
# a stiff position servo on the finger tendon whose force is capped at the configured grasp force, so
# closing on any cup squeezes with that force (Menagerie's spring squeezed 2-4 N depending on width).
FINGER_TRAVEL = 0.04      # per finger
FINGER_SPEED = 0.05       # m/s per finger
GRIP_KP = 28000.0         # N/m on the tendon (= mean finger travel); saturates within ~5 mm of contact
GRIP_KD = 200.0           # N s/m, about critical damping for the fingers
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
TABLE_HALF = (0.55, 0.75)  # half extents x, y
TABLE_CENTER_X = 0.4
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
OPERATOR_GROUP = 5       # geom group MuJoCo cameras don't render (they show 0-2): headset/dashboard only

# -- task: stack_cups ------------------------------------------------------------------------------
# Three tapered cups (from mujoco_toy-project's generate_cup.py; same colors every episode, start positions
# randomized) that the operator nests into one stack. Cup body frames sit at the centre of the cup bottom.
# MuJoCo only collides convex shapes, so a cup is a thin cylinder base plus a ring of thin tilted boxes.
TASK = "stack_cups"
CUP_R_BOTTOM = 0.028     # outer radius at the bottom; the Panda opens to 8 cm
CUP_R_TOP = 0.036
CUP_H = 0.07
CUP_T = 0.003            # wall and base thickness
CUP_N_WALL = 16          # boxes forming the wall
CUP_DENSITY = 600        # plastic, about 35 g per cup
CUP_FRICTION = 1.0


class Cup(NamedTuple):
    name: str
    rgba: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        return f"{self.name.split('_')[0]} cup"


# Bottom to top.
CUPS = (
    Cup("red_cup", (0.86, 0.15, 0.12, 1)),
    Cup("green_cup", (0.15, 0.65, 0.25, 1)),
    Cup("yellow_cup", (0.98, 0.78, 0.1, 1)),
)
LEVELS = len(CUPS) - 1   # cups stacked on the base cup
OBJECTS = tuple(c.name for c in CUPS)  # free-floating bodies whose poses are recorded
INSTRUCTION = (f"Stack the cups in the taped area: {CUPS[0].label} at the bottom, then "
               + ", ".join(c.label for c in CUPS[1:-1]) + f", and {CUPS[-1].label} on top.")
# Height of a nested cup above the one below, measured in the sim (the flat wall boxes sit a bit higher than
# a round cup would: CUP_T * CUP_H / (CUP_R_TOP - CUP_R_BOTTOM) = 26 mm). Only the goal picture uses it.
NEST_DZ = 0.033
# Stacked: upright, centred on the cup below and resting inside it (as in mujoco_toy-project).
TOL_TILT = np.deg2rad(15)
TOL_XY = 0.01
NEST_DZ_RANGE = (0.005, 0.05)
STACK_STILL_MPS = 0.02
# Build area: a taped square on the table (seen by the cameras, like tape on a real table). The stack only
# counts when its base cup lies completely inside; cups never start in it.
BUILD_AREA_CENTER = (0.45, 0.0)
BUILD_AREA_HALF = (0.07, 0.07)
TAPE_W = 0.006
TAPE_RGBA = (0.92, 0.92, 0.88, 1)

# Rigid contacts (cups, arm links, table top): timeconst 0.004 s (>= 2 * physics_dt for stability) and
# impedance 0.99-0.999, so an arm pushing at full torque stays on the surface instead of sinking in.
RIGID_SOLREF = [0.004, 1]
RIGID_SOLIMP = [0.99, 0.999, 0.0005, 0.5, 2]


def _cup_parts() -> list[tuple]:
    """(type, size, pos, quat) of the base and the wall boxes; the top leans outwards."""
    tilt = np.arctan2(CUP_R_TOP - CUP_R_BOTTOM, CUP_H)
    slant = np.hypot(CUP_R_TOP - CUP_R_BOTTOM, CUP_H)
    mid_r = (CUP_R_BOTTOM + CUP_R_TOP) / 2 - CUP_T / 2
    half_w = CUP_R_TOP * np.tan(np.pi / CUP_N_WALL) * 1.05  # a bit wider than the chord so neighbours overlap
    parts = [(CYL, [CUP_R_BOTTOM, CUP_T / 2, 0], [0, 0, CUP_T / 2], [1, 0, 0, 0])]
    for i in range(CUP_N_WALL):
        a = 2 * np.pi * i / CUP_N_WALL
        q = np.empty(4)
        mujoco.mju_mulQuat(q, [np.cos(a / 2), 0, 0, np.sin(a / 2)], [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0])
        parts.append((BOX, [CUP_T / 2, half_w, slant / 2], [mid_r * np.cos(a), mid_r * np.sin(a), CUP_H / 2], q))
    return parts


def _add_task(spec: mujoco.MjSpec, th: float) -> None:
    for i, c in enumerate(CUPS):
        spec.add_material(name=f"{c.name}_mat", rgba=list(c.rgba), specular=0.4, shininess=0.6)
        body = spec.worldbody.add_body(name=c.name, pos=[0.3 + 0.1 * i, 0.0, th])
        body.add_freejoint(name=f"{c.name}_free")
        # priority=1: cup contact params win against the table; finger pads (priority 2) win against cups.
        for gtype, size, pos, quat in _cup_parts():
            body.add_geom(type=gtype, size=size, pos=pos, quat=list(quat), material=f"{c.name}_mat",
                          density=CUP_DENSITY, condim=4, friction=[CUP_FRICTION, 0.01, 0.001],
                          solref=RIGID_SOLREF, solimp=RIGID_SOLIMP, priority=1)
    _add_build_area(spec, th)


def _add_build_area(spec: mujoco.MjSpec, th: float) -> None:
    """Four tape strips (visual only) just above the table top, outlining the build area from the inside."""
    (cx, cy), (hx, hy), t = BUILD_AREA_CENTER, BUILD_AREA_HALF, TAPE_W / 2
    strips = [([cx, cy + s * (hy - t)], [hx, t]) for s in (-1, 1)] + \
             [([cx + s * (hx - t), cy], [t, hy - 2 * t]) for s in (-1, 1)]
    for i, (xy, half) in enumerate(strips):
        spec.worldbody.add_geom(name=f"build_area_{i}", type=BOX, size=[*half, 0.0002], pos=[*xy, th + 0.0002],
                                rgba=list(TAPE_RGBA), group=1, contype=0, conaffinity=0)


def in_build_area(data: mujoco.MjData) -> bool:
    """Whether the base cup's whole footprint (its rim, seen from above) lies inside the build area."""
    xy = data.body(CUPS[0].name).xpos[:2]
    return bool(np.all(np.abs(xy - BUILD_AREA_CENTER) + CUP_R_TOP <= BUILD_AREA_HALF))


def task_spec() -> dict:
    """The mission as data, for the operator UI: cups, where each goes (relative to the base cup's bottom
    centre, in stacking order), and the build area. Everything the UI shows of the goal comes from here."""
    return {
        "task": TASK,
        "instruction": INSTRUCTION,
        "cup": {"r_bottom": CUP_R_BOTTOM, "r_top": CUP_R_TOP, "height": CUP_H},
        "cups": [{"name": c.name, "label": c.label, "rgba": list(c.rgba)} for c in CUPS],
        "goal": [{"cup": i, "pos": [0.0, 0.0, i * NEST_DZ]} for i in range(len(CUPS))],
        "area": {"center": list(BUILD_AREA_CENTER), "half": list(BUILD_AREA_HALF)},
    }


def set_cup_pose(data: mujoco.MjData, name: str, pos) -> None:
    j = data.joint(f"{name}_free")
    j.qpos[:3] = pos
    j.qpos[3:] = [1, 0, 0, 0]
    j.qvel[:] = 0


def _randomize_task(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, th: float) -> None:
    """Same cups every episode; only where they start on the table changes (non-overlapping, outside the
    build area, within reach of the arm)."""
    placed: list[np.ndarray] = []
    r = CUP_R_TOP
    for c in CUPS:
        for _ in range(1000):
            xy = np.array([rng.uniform(0.30, 0.60), rng.uniform(-0.30, 0.30)])
            to_area = np.maximum(np.abs(xy - BUILD_AREA_CENTER) - BUILD_AREA_HALF, 0.0)
            if np.linalg.norm(to_area) > r + 0.02 and all(np.linalg.norm(xy - p) > 2 * r + 0.03 for p in placed):
                break
        placed.append(xy)
        set_cup_pose(data, c.name, [*xy, th])


def cup_stacked(data: mujoco.MjData, lower: int, upper: int) -> bool:
    """Whether cup `upper` rests upright inside cup `lower`."""
    bl, bu = data.body(CUPS[lower].name), data.body(CUPS[upper].name)
    if bl.xmat[8] < np.cos(TOL_TILT) or bu.xmat[8] < np.cos(TOL_TILT):  # z axis . world z
        return False
    d = bu.xpos - bl.xpos
    return bool(np.linalg.norm(d[:2]) < TOL_XY and NEST_DZ_RANGE[0] < d[2] < NEST_DZ_RANGE[1])


def stack_progress(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Number of cups (0..LEVELS) correctly stacked on the base, counted bottom-up. 0 unless the base cup
    stands inside the build area."""
    if not in_build_area(data):
        return 0
    n = 0
    for k in range(1, len(CUPS)):
        if not cup_stacked(data, k - 1, k):
            break
        n = k
    return n


def held_cups(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int | None]:
    """Per arm, the index of the cup both of its fingers touch (None if none)."""
    cup_of = {model.body(c.name).id: i for i, c in enumerate(CUPS)}
    finger_of = {model.body(f"{a}_{f}").id: (a, f) for a in ARMS for f in ("left_finger", "right_finger")}
    touch: dict[tuple[str, str], set[int]] = {k: set() for k in finger_of.values()}
    gb = model.geom_bodyid
    for g1, g2 in zip(data.contact.geom1, data.contact.geom2):
        b1, b2 = gb[g1], gb[g2]
        for f, o in ((b1, b2), (b2, b1)):
            if f in finger_of and o in cup_of:
                touch[finger_of[f]].add(cup_of[o])
    out: dict[str, int | None] = {}
    for a in ARMS:
        both = touch[(a, "left_finger")] & touch[(a, "right_finger")]
        out[a] = min(both) if both else None
    return out


def _task_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Stack complete and standing on its own (released, at rest)."""
    if stack_progress(model, data) < LEVELS or any(v is not None for v in held_cups(model, data).values()):
        return False
    return all(np.linalg.norm(data.joint(f"{o}_free").qvel[:3]) < STACK_STILL_MPS for o in OBJECTS)


# -- scene -----------------------------------------------------------------------------------------
def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera xyaxes so that the camera at `pos` looks at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, f)
    return [*right, *upv]


def _add_environment(spec: mujoco.MjSpec, th: float) -> None:
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Elliptic friction cones are exact (the default pyramid approximates them and lets grasps creep);
    # impratio > 1 makes friction stiffer than normal contact, which MuJoCo recommends against slip.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.45, 0.45, 0.48]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]

    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.11, 0.12, 0.15], rgb2=[0.15, 0.16, 0.2], width=512, height=512,
    )
    spec.add_material(name="floor_mat", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.15)
    spec.add_material(name="table_mat", rgba=[0.32, 0.27, 0.23, 1], specular=0.2)
    spec.add_material(name="leg_mat", rgba=[0.2, 0.2, 0.22, 1])

    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.4, 0.0, 3.0], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45], castshadow=True)
    wb.add_light(name="fill", pos=[-1.0, 1.0, 2.5], dir=[0.5, -0.3, -1], diffuse=[0.3, 0.3, 0.35], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor_mat")

    hx, hy = TABLE_HALF
    table = wb.add_body(name="table", pos=[TABLE_CENTER_X, 0, 0])
    table.add_geom(name="table_top", type=BOX, size=[hx, hy, 0.02],
                   pos=[0, 0, th - 0.02], material="table_mat", friction=[1.0, 0.005, 0.0001],
                   solref=RIGID_SOLREF, solimp=RIGID_SOLIMP)
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(type=BOX, size=[0.025, 0.025, (th - 0.04) / 2],
                           pos=[sx * (hx - 0.05), sy * (hy - 0.05), (th - 0.04) / 2],
                           material="leg_mat", contype=0, conaffinity=0)

    # Scene cameras (ALOHA naming). cam_high looks down on the table from the operator's right: straight above
    # the workspace the single arm (based at the origin) would block the view.
    hi_pos = [0.42, -0.6, th + 0.8]
    wb.add_camera(name="cam_high", pos=hi_pos, xyaxes=lookat_xyaxes(hi_pos, [0.42, 0, th]), fovy=62)
    lo_pos = [1.15, 0.0, th + 0.35]
    wb.add_camera(name="cam_low", pos=lo_pos, xyaxes=lookat_xyaxes(lo_pos, [0.35, 0, th + 0.1]), fovy=60)


def _add_arm(spec: mujoco.MjSpec, arm: str, cfg: SimConfig) -> None:
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for body in panda.bodies:
        body.gravcomp = 1.0  # the real Franka controller compensates gravity internally
    # Gripper: force = kp * (target - travel) - kd * speed on the "split" tendon (mean finger travel). Each
    # finger gets half the tendon force, so the cap is 2x the grasp force (squeeze between the fingers).
    grip = panda.actuator("actuator8")
    grip.gainprm = [GRIP_KP * FINGER_TRAVEL / GRIPPER_CTRL_MAX, *grip.gainprm[1:]]
    grip.biasprm = [0.0, -GRIP_KP, -GRIP_KD, *grip.biasprm[3:]]
    grip.forcerange = [-2 * cfg.gripper_force, 2 * cfg.gripper_force]
    for g in panda.geoms:
        if g.contype:
            g.solref, g.solimp = RIGID_SOLREF, RIGID_SOLIMP
    # Grippy finger pads; priority 2 so their friction/stiffness win over the cups.
    # Menagerie's fingertip is a big pad plus four small boxes standing 1.5 mm proud of it with 4-5 mm slots in
    # between; a thin cup wall pressed into such a slot gets clamped by opposing contacts and the cup
    # stays on the open finger. One flat box over the same area (same inner face, so grasp width is unchanged).
    for f in ("left_finger", "right_finger"):
        for g in panda.body(f).geoms:
            if g.classname.name == "fingertip_pad_collision_1":
                g.size, g.pos = [0.0085, 0.00475, 0.0085], [0, 0.00475, 0.0445]
            elif g.classname.name.startswith("fingertip_pad_collision"):
                panda.delete(g)
                continue
            if g.contype:
                g.friction = [1.5, 0.01, 0.001]
                g.condim, g.priority, g.solref = 4, 2, [0.004, 1]
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0.006, 0.006],
                  rgba=[0, 1, 1, 0.0], group=4)
    # Grasp-point marker for the operator (headset + dashboard only, not the recorded cameras): the
    # point midway between the finger pads that the arm is steered by. The web client colours it per arm.
    hand.add_geom(name="tcp_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], pos=[0, 0, TCP_OFFSET],
                  rgba=[1, 1, 1, 1], group=OPERATOR_GROUP, contype=0, conaffinity=0, mass=0)
    cam_pos = [0.1, 0.0, 0.0]
    hand.add_camera(name="wrist_cam", pos=cam_pos,
                    xyaxes=lookat_xyaxes(cam_pos, [0, 0, 0.16], up=[1, 0, 0]), fovy=80)
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, cfg.table_height])
    spec.attach(panda, prefix=f"{arm}_", frame=frame)


def build_spec(cfg: SimConfig, with_task: bool = True) -> mujoco.MjSpec:
    """Scene spec with the arm and (optionally) the task objects."""
    spec = mujoco.MjSpec()
    spec.modelname = "panda_cups"
    _add_environment(spec, cfg.table_height)
    for arm in ARMS:
        _add_arm(spec, arm, cfg)
        spec.add_exclude(bodyname1="table", bodyname2=f"{arm}_link0")
    spec.camera(f"{ARMS[0]}_wrist_cam").name = "cam_wrist"
    spec.option.timestep = cfg.physics_dt
    if with_task:
        _add_task(spec, cfg.table_height)
    return spec


# -- environment -----------------------------------------------------------------------------------
class PandaEnv:
    """Joint-position-controlled single Franka scene with the stack_cups task."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.table_height = cfg.sim.table_height
        self.spec = build_spec(cfg.sim)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.xml = self.spec.to_xml()
        self.control_dt = 1.0 / cfg.sim.control_hz
        self.n_substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        m = self.model
        self.arm_qadr = np.array([m.joint(f"{a}_{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])
        self.arm_vadr = np.array([m.joint(f"{a}_{j}").dofadr[0] for a in ARMS for j in ARM_JOINTS])
        self.finger_qadr = np.array(
            [[m.joint(f"{a}_finger_joint{i}").qposadr[0] for i in (1, 2)] for a in ARMS]
        )
        self.arm_act = np.array([m.actuator(f"{a}_actuator{i}").id for a in ARMS for i in range(1, 8)])
        self.grip_act = np.array([m.actuator(f"{a}_actuator8").id for a in ARMS])
        self.tcp_sites = [m.site(f"{a}_tcp").id for a in ARMS]
        self.grip_step = GRIPPER_CTRL_MAX * FINGER_SPEED * self.control_dt / FINGER_TRAVEL  # ctrl per tick
        self.state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> dict:
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        for arm in ARMS:
            for j, q in zip(ARM_JOINTS, HOME_Q):
                self.data.joint(f"{arm}_{j}").qpos[0] = q
            for f in ("finger_joint1", "finger_joint2"):
                self.data.joint(f"{arm}_{f}").qpos[0] = 0.04
        _randomize_task(self.model, self.data, rng, self.table_height)
        self.data.ctrl[self.grip_act] = GRIPPER_CTRL_MAX  # fingers start open
        self.apply_action(np.tile(HOME_Q, len(ARMS)), np.ones(len(ARMS)))
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def apply_action(self, joint_targets: np.ndarray, grippers: np.ndarray) -> None:
        """joint_targets: (7,) radians; grippers: (1,) in [0=closed, 1=open].

        The finger target moves towards the command at the real hand's 50 mm/s per finger.
        """
        self.data.ctrl[self.arm_act] = joint_targets
        cur = self.data.ctrl[self.grip_act]
        target = np.clip(grippers, 0.0, 1.0) * GRIPPER_CTRL_MAX
        self.data.ctrl[self.grip_act] = cur + np.clip(target - cur, -self.grip_step, self.grip_step)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def observe(self) -> dict:
        d = self.data
        sid = self.tcp_sites[0]
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[sid])
        return {
            "joint_pos": d.qpos[self.arm_qadr].copy(),
            "joint_vel": d.qvel[self.arm_vadr].copy(),
            "gripper_width": d.qpos[self.finger_qadr].sum(axis=1),
            "ee_pose": np.concatenate([d.site_xpos[sid], q]),
            # (k, 7) [x y z qw qx qy qz] per tracked object
            "objects": np.stack([d.joint(f"{o}_free").qpos.copy() for o in OBJECTS]),
        }

    def success(self) -> bool:
        return _task_success(self.model, self.data)

    def progress(self) -> int:
        return stack_progress(self.model, self.data)

    def in_area(self) -> bool:
        return in_build_area(self.data)

    def held(self) -> dict[str, int | None]:
        return held_cups(self.model, self.data)

    # Full state, for exact replay and offline re-rendering.
    def get_state(self) -> np.ndarray:
        s = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return s

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, np.asarray(state, dtype=np.float64), STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)


class AppearanceRandomizer:
    """Per-episode visual variation for exported camera images (domain randomization).

    Policies trained on images from one fixed look tend to fail on a real camera. This varies lighting,
    table/floor colour, cup brightness (not hue: the instruction names the colours) and the camera
    mounts a little. Rendering only: physics and the recorded data are untouched.
    """

    FIELDS = ("light_diffuse", "light_pos", "mat_rgba", "cam_pos")

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.base = {k: getattr(model, k).copy() for k in self.FIELDS}
        self.scene_cams = [model.camera(n).id for n in ("cam_high", "cam_low")]
        self.wrist_cams = [model.camera("cam_wrist").id]
        self.surface_mats = [model.material(n).id for n in ("table_mat", "floor_mat")]
        self.cup_mats = [model.material(f"{c.name}_mat").id for c in CUPS]

    def apply(self, rng: np.random.Generator) -> None:
        m = self.model
        self.restore()
        m.light_diffuse[:] *= rng.uniform(0.6, 1.4, (m.nlight, 1))
        m.light_pos[:, :2] += rng.uniform(-0.4, 0.4, (m.nlight, 2))
        m.mat_rgba[self.surface_mats, :3] *= rng.uniform(0.8, 1.2, (len(self.surface_mats), 3))
        m.mat_rgba[self.cup_mats, :3] *= rng.uniform(0.85, 1.15, (len(self.cup_mats), 1))
        np.clip(m.mat_rgba, 0.0, 1.0, out=m.mat_rgba)
        m.cam_pos[self.scene_cams] += rng.uniform(-0.015, 0.015, (len(self.scene_cams), 3))
        m.cam_pos[self.wrist_cams] += rng.uniform(-0.003, 0.003, (len(self.wrist_cams), 3))

    def restore(self) -> None:
        for k, v in self.base.items():
            getattr(self.model, k)[:] = v


class CameraRenderer:
    """Offscreen rendering of the named cameras (dashboard previews and LeRobot export)."""

    def __init__(self, model: mujoco.MjModel, height: int, width: int, cameras: list[str]):
        self.cameras = list(cameras)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def render_jpeg(self, data: mujoco.MjData, quality: int = 70) -> dict[str, bytes]:
        out = {}
        for cam, img in self.render(data).items():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            out[cam] = buf.getvalue()
        return out

    def close(self) -> None:
        self.renderer.close()
