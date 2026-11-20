"""The MuJoCo side: scene (one Franka arm + table + cups), the stack_cups task, and the environment.

Built on mujoco_toy-project (the `toy_project` package, a uv path dependency): its models/panda_scene.xml (robot,
cups, cameras) is imported as it is and set on a table, and its names, reset randomization and cup stacking check
are used as they are. This module adds the table and the settings teleoperation needs (gripper servo, contacts,
grasp marker).

World frame: z up, +x points away from the operator (towards the far table edge),
+y is the operator's left. The arm base sits on the near table edge facing +x.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import toy_project
from PIL import Image
from toy_project import manipulation_environment as toy

from questcollect.config import Config, SimConfig

# The mujoco_toy-project checkout the package was installed from (src/toy_project -> repository root).
TOY_PROJECT = Path(toy_project.__file__).resolve().parents[2]
SCENE_XML = TOY_PROJECT / "models" / "panda_scene.xml"

# One arm, driven by the operator's right hand/controller (the left controller only sends commands).
ARMS = ("r",)
ARM_PREFIX = {"r": toy.ROBOT_PREFIX}  # name prefix of that arm's bodies/joints/actuators in the toy scene
ARM_JOINTS = [n.removeprefix(toy.ROBOT_PREFIX) for n in toy.JNT_NAMES]
HOME_Q = toy.JNT_RESET_QPOS  # the toy's start pose; its reset adds random noise to it
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
# The toy scene's cups (toy.OBJECT_NAMES, bottom to top; models/cup_<colour>.xml; start positions randomized by the
# toy's reset): the operator puts each one on top of (into) the one before it, anywhere on the table. Cup body
# frames sit at the centre of the cup bottom. MuJoCo only collides convex shapes, so the toy cup is a thin
# cylinder base plus a ring of thin tilted boxes.
TASK = "stack_cups"


def _cup_xml(name: str) -> Path:
    return TOY_PROJECT / "models" / f"cup_{name.split('_')[0]}.xml"


def _cup_dims() -> tuple[float, float, float]:
    """(outer radius at the bottom, at the rim, height) of the toy cup, measured from its geoms."""
    m = mujoco.MjSpec.from_file(str(_cup_xml(toy.OBJECT_NAMES[0]))).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    corners = []
    for g in range(m.ngeom):
        if m.geom_type[g] != BOX:
            continue
        half = m.geom_size[g]
        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            corners.append(d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (s * half))
    corners = np.array(corners)
    base = next(g for g in range(m.ngeom) if m.geom_type[g] == CYL)
    return float(m.geom_size[base, 0]), float(np.hypot(*corners[:, :2].T).max()), float(corners[:, 2].max())


CUP_R_BOTTOM, CUP_R_TOP, CUP_H = _cup_dims()


class Cup(NamedTuple):
    name: str
    rgba: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        return f"{self.name.split('_')[0]} cup"


def _cup_rgba(name: str) -> tuple[float, float, float, float]:
    m = mujoco.MjSpec.from_file(str(_cup_xml(name))).compile()
    return tuple(float(v) for v in m.geom_rgba[0])


# Bottom to top; body names and order from the toy project.
CUPS = tuple(Cup(name, _cup_rgba(name)) for name in toy.OBJECT_NAMES)
LEVELS = len(CUPS) - 1   # cups stacked on the base cup
OBJECTS = tuple(c.name for c in CUPS)  # free-floating bodies whose poses are recorded
INSTRUCTION = f"Put the {CUPS[1].label} on top of the {CUPS[0].label}."
# Height of a nested cup above the one below, measured in the sim with the toy cup. Only the goal picture uses it.
NEST_DZ = 0.033
STACK_STILL_MPS = 0.02

# Rigid contacts (cups, arm links, table top): timeconst 0.004 s (>= 2 * physics_dt for stability) and
# impedance 0.99-0.999, so an arm pushing at full torque stays on the surface instead of sinking in.
RIGID_SOLREF = [0.004, 1]
RIGID_SOLIMP = [0.99, 0.999, 0.0005, 0.5, 2]


def _adapt_cups(spec: mujoco.MjSpec) -> None:
    """The toy cups keep their shape, mass, friction and colour. Joint names get a `_free` suffix; priority=1: cup
    contact params win against the table, finger pads (priority 2) win against cups."""
    for c in CUPS:
        body = spec.body(c.name)
        spec.joint(f"{c.name.split('_')[0]}_joint").name = f"{c.name}_free"
        for g in body.geoms:
            g.condim, g.priority, g.solref, g.solimp = 4, 1, RIGID_SOLREF, RIGID_SOLIMP


def task_spec() -> dict:
    """The mission as data, for the operator UI: cups and where each goes (relative to the base cup's bottom
    centre, in stacking order). Everything the UI shows of the goal comes from here."""
    return {
        "task": TASK,
        "instruction": INSTRUCTION,
        "cup": {"r_bottom": CUP_R_BOTTOM, "r_top": CUP_R_TOP, "height": CUP_H},
        "cups": [{"name": c.name, "label": c.label, "rgba": list(c.rgba)} for c in CUPS],
        "goal": [{"cup": i, "pos": [0.0, 0.0, i * NEST_DZ]} for i in range(len(CUPS))],
    }


def set_cup_pose(data: mujoco.MjData, name: str, pos) -> None:
    j = data.joint(f"{name}_free")
    j.qpos[:3] = pos
    j.qpos[3:] = [1, 0, 0, 0]
    j.qvel[:] = 0


def cup_stacked(model: mujoco.MjModel, data: mujoco.MjData, lower: int, upper: int) -> bool:
    """Whether cup `upper` rests upright inside cup `lower`: the toy project's check."""
    return toy.check_cups_stacked(model, data, CUPS[lower].name, CUPS[upper].name)


def stack_progress(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Number of cups (0..LEVELS) correctly stacked on the base, counted bottom-up."""
    n = 0
    for k in range(1, len(CUPS)):
        if not cup_stacked(model, data, k - 1, k):
            break
        n = k
    return n


def held_cups(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int | None]:
    """Per arm, the index of the cup both of its fingers touch (None if none)."""
    cup_of = {model.body(c.name).id: i for i, c in enumerate(CUPS)}
    finger_of = {model.body(f"{ARM_PREFIX[a]}{f}").id: (a, f) for a in ARMS for f in ("left_finger", "right_finger")}
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
    # The cameras come with the toy scene (toy.CAMERA_NAMES).


def _adapt_arm(spec: mujoco.MjSpec, arm: str, cfg: SimConfig) -> None:
    """Teleop settings on the toy scene's Panda (Menagerie's, with the toy project's tcp site and wrist camera)."""
    pre = ARM_PREFIX[arm]
    for body in spec.bodies:
        if body.name.startswith(pre):
            body.gravcomp = 1.0  # the real Franka controller compensates gravity internally
    # Gripper: force = kp * (target - travel) - kd * speed on the "split" tendon (mean finger travel). Each
    # finger gets half the tendon force, so the cap is 2x the grasp force (squeeze between the fingers).
    grip = spec.actuator(f"{pre}actuator8")
    grip.gainprm = [GRIP_KP * FINGER_TRAVEL / GRIPPER_CTRL_MAX, *grip.gainprm[1:]]
    grip.biasprm = [0.0, -GRIP_KP, -GRIP_KD, *grip.biasprm[3:]]
    grip.forcerange = [-2 * cfg.gripper_force, 2 * cfg.gripper_force]
    robot_geoms = [g for b in spec.bodies if b.name.startswith(pre) for g in b.geoms]
    for g in robot_geoms:
        if g.contype:
            g.solref, g.solimp = RIGID_SOLREF, RIGID_SOLIMP
    # Grippy finger pads; priority 2 so their friction/stiffness win over the cups.
    # Menagerie's fingertip is a big pad plus four small boxes standing 1.5 mm proud of it with 4-5 mm slots in
    # between; a thin cup wall pressed into such a slot gets clamped by opposing contacts and the cup
    # stays on the open finger. One flat box over the same area (same inner face, so grasp width is unchanged).
    for f in ("left_finger", "right_finger"):
        for g in list(spec.body(f"{pre}{f}").geoms):
            cls = g.classname.name
            if cls.endswith("fingertip_pad_collision_1"):
                g.size, g.pos = [0.0085, 0.00475, 0.0085], [0, 0.00475, 0.0445]
            elif "fingertip_pad_collision" in cls:
                spec.delete(g)
                continue
            if g.contype:
                g.friction = [1.5, 0.01, 0.001]
                g.condim, g.priority, g.solref = 4, 2, [0.004, 1]
    tcp = spec.site(f"{pre}tcp")  # from the toy model: midway between the fingertips
    tcp.group = 4  # it is red and in group 1, which the recorded cameras would show
    # Grasp-point marker for the operator (headset + dashboard only, not the recorded cameras): the
    # point midway between the finger pads that the arm is steered by. The web client colours it per arm.
    spec.body(f"{pre}hand").add_geom(name=f"{pre}tcp_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
                                     pos=tcp.pos, rgba=[1, 1, 1, 1], group=OPERATOR_GROUP, contype=0, conaffinity=0,
                                     mass=0)


def _raise_scene(spec: mujoco.MjSpec, dz: float) -> None:
    """Move everything placed directly in the world up by dz (the toy scene stands on z=0, here on the table)."""
    wb = spec.worldbody
    for el in (*wb.frames, *[b for b in wb.bodies if b.frame is None], *wb.cameras, *wb.lights, *wb.geoms, *wb.sites):
        el.pos = [el.pos[0], el.pos[1], el.pos[2] + dz]


def build_spec(cfg: SimConfig, with_task: bool = True) -> mujoco.MjSpec:
    """The toy scene on the table, with (optionally) the cups.

    The toy scene is the root spec (not attached into another one), so every name stays exactly as in the toy
    project and the scene saves to XML that loads again (episodes store it as model_xml).
    """
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.modelname = "panda_cups"
    spec.delete(spec.body("gym_gym"))  # the toy's floor plane and light: here the table and its room take their place
    if not with_task:
        for c in CUPS:
            spec.delete(spec.body(c.name))
    _raise_scene(spec, cfg.table_height)
    _add_environment(spec, cfg.table_height)
    spec.option.timestep = cfg.physics_dt
    for arm in ARMS:
        _adapt_arm(spec, arm, cfg)
        spec.add_exclude(bodyname1="table", bodyname2=f"{ARM_PREFIX[arm]}link0")
    if with_task:
        _adapt_cups(spec)
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
        self.arm_qadr = np.array([m.joint(f"{ARM_PREFIX[a]}{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])
        self.arm_vadr = np.array([m.joint(f"{ARM_PREFIX[a]}{j}").dofadr[0] for a in ARMS for j in ARM_JOINTS])
        self.finger_qadr = np.array(
            [[m.joint(f"{ARM_PREFIX[a]}finger_joint{i}").qposadr[0] for i in (1, 2)] for a in ARMS]
        )
        self.arm_act = np.array([m.actuator(f"{ARM_PREFIX[a]}actuator{i}").id for a in ARMS for i in range(1, 8)])
        self.grip_act = np.array([m.actuator(f"{ARM_PREFIX[a]}actuator8").id for a in ARMS])
        self.tcp_sites = [m.site(f"{ARM_PREFIX[a]}tcp").id for a in ARMS]
        self.grip_step = GRIPPER_CTRL_MAX * FINGER_SPEED * self.control_dt / FINGER_TRAVEL  # ctrl per tick
        self.state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> dict:
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        # The toy project's reset randomization: arm = its start pose plus noise, cups at random spots in front of
        # the base. The toy scene sits on the table, so its origin is the table top under the arm base.
        m = self.model
        toy.reset_arm(m, self.data, rng, [m.joint(f"{ARM_PREFIX[a]}{j}").id for a in ARMS for j in ARM_JOINTS])
        toy.reset_object_positions(m, self.data, rng, [m.body(c.name).id for c in CUPS],
                                   origin=np.array([0.0, 0.0, self.table_height]))
        for arm in ARMS:
            for f in ("finger_joint1", "finger_joint2"):
                self.data.joint(f"{ARM_PREFIX[arm]}{f}").qpos[0] = 0.04
        self.data.ctrl[self.grip_act] = GRIPPER_CTRL_MAX  # fingers start open
        self.apply_action(self.data.qpos[self.arm_qadr].copy(), np.ones(len(ARMS)))  # hold the start pose
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

    FIELDS = ("light_diffuse", "light_pos", "mat_rgba", "geom_rgba", "cam_pos")

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.base = {k: getattr(model, k).copy() for k in self.FIELDS}
        on_robot = [n.startswith(toy.ROBOT_PREFIX) for n in toy.CAMERA_NAMES]
        self.scene_cams = [model.camera(n).id for n, r in zip(toy.CAMERA_NAMES, on_robot) if not r]
        self.wrist_cams = [model.camera(n).id for n, r in zip(toy.CAMERA_NAMES, on_robot) if r]
        self.surface_mats = [model.material(n).id for n in ("table_mat", "floor_mat")]
        self.cup_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in {model.body(c.name).id for c in CUPS}]

    def apply(self, rng: np.random.Generator) -> None:
        m = self.model
        self.restore()
        m.light_diffuse[:] *= rng.uniform(0.6, 1.4, (m.nlight, 1))
        m.light_pos[:, :2] += rng.uniform(-0.4, 0.4, (m.nlight, 2))
        m.mat_rgba[self.surface_mats, :3] *= rng.uniform(0.8, 1.2, (len(self.surface_mats), 3))
        for c in CUPS:  # one brightness per cup (the toy cups are coloured per geom, not by material)
            m.geom_rgba[[g for g in self.cup_geoms if m.geom_bodyid[g] == m.body(c.name).id], :3] *= rng.uniform(0.85, 1.15)
        np.clip(m.geom_rgba, 0.0, 1.0, out=m.geom_rgba)
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
