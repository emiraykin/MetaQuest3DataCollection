import os
import time

import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.ik import ArmIK
from questcollect.recorder import Recorder, RecState, list_episodes, read_episode
from questcollect.runtime import HAPTIC_COMPLETE, HAPTIC_PLACED, Runtime
from questcollect.scene_export import export_scene
from questcollect.sim import CUP_H, CUPS, HOME_Q, LEVELS, NEST_DZ, PandaEnv, set_cup_pose, task_spec


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(cfg):
    return PandaEnv(cfg)


def test_task_builds_resets_and_settles(env):
    obs = env.reset(seed=3)
    assert obs["joint_pos"].shape == (7,) and obs["objects"].shape == (len(CUPS), 7)
    for _ in range(50):
        env.step()
    o = env.observe()["objects"]
    assert np.all(np.abs(o[:, :3] - obs["objects"][:, :3]) < 0.002), "cups should rest on the table"
    assert np.allclose(o[:, 2], env.table_height, atol=0.001)
    assert env.progress() == 0 and not env.success()


def _stack(env, base_xy, order=None, offset=(0.0, 0.0)):
    """Teleport the cups into one stack (optionally in another order / with the top cup shifted), settle."""
    env.reset(seed=1)
    order = order or list(range(len(CUPS)))
    for level, i in enumerate(order):
        xy = np.add(base_xy, offset) if level == len(order) - 1 and level > 0 else base_xy
        set_cup_pose(env.data, CUPS[i].name, [*xy, env.table_height + level * (NEST_DZ + 0.002)])
    for _ in range(50):
        env.step()


def test_stack_complete_is_success(env):
    _stack(env, (0.45, 0.0))
    assert env.progress() == LEVELS and env.success()
    dz = np.diff(env.observe()["objects"][:, 2])
    assert np.allclose(dz, NEST_DZ, atol=0.003), "cups must nest, not stand on each other"


def test_stack_counts_anywhere_on_the_table(env):
    """There is no build area: the stack counts wherever the red cup stands."""
    for xy in ((0.35, 0.25), (0.6, -0.2)):
        _stack(env, xy)
        assert env.progress() == LEVELS and env.success()


def test_cup_beside_the_base_does_not_count(env):
    """A cup standing next to the base (not inside it) isn't stacked."""
    _stack(env, (0.45, 0.0), offset=(0.0, 0.1))
    assert env.progress() == 0 and not env.success()


def test_task_spec_describes_goal():
    """The UI draws the goal from task_spec alone, so it must match the stack the success check uses."""
    spec = task_spec()
    assert [g["cup"] for g in spec["goal"]] == list(range(len(CUPS)))
    assert [g["pos"][2] for g in spec["goal"]] == pytest.approx([k * NEST_DZ for k in range(len(CUPS))])
    assert "area" not in spec
    assert len(spec["cups"]) == len(CUPS) and spec["cup"]["height"] == CUP_H


def test_wrong_order_is_not_success(env):
    _stack(env, (0.45, 0.0), order=[1, 0])  # red in green
    assert env.progress() == 0 and not env.success()


def test_state_roundtrip(env):
    _stack(env, (0.45, 0.0))
    state, pos = env.get_state(), env.observe()["objects"]
    env.reset(seed=3)
    env.set_state(state)
    assert env.progress() == LEVELS
    for _ in range(25):
        env.step()
    assert np.allclose(env.observe()["objects"][:, :3], pos[:, :3], atol=5e-4)


def test_grasp_detection_and_force(cfg, env):
    """Close the gripper around a cup: held() reports it, the squeeze is the configured grasp force,
    and the cup lifts with the hand."""
    env.reset(seed=1)
    th = env.table_height
    for i, c in enumerate(CUPS):  # green cup under the hand, the others out of the way
        set_cup_pose(env.data, c.name, [0.45, 0.0, th] if i == 1 else [0.35 + 0.15 * i, -0.3, th])
    ik = ArmIK(cfg.sim, cfg.ik, env.control_dt)
    down = np.array([0.0, 1.0, 0.0, 0.0])
    z_grasp = CUP_H / 2
    for z, grip, n in ((0.15, 1.0, 60), (z_grasp, 1.0, 50), (z_grasp, 0.0, 40), (0.15, 0.0, 50)):
        ik.set_target("r", [0.45, 0.0, th + z], down)
        for _ in range(n):
            env.apply_action(ik.solve(), np.array([grip]))
            env.step()
        if grip == 0.0 and z == z_grasp:
            assert env.held() == {"r": 1}
    assert env.held()["r"] == 1
    squeeze = -env.data.actuator_force[env.grip_act[0]] / 2  # each finger gets half the tendon force
    assert abs(squeeze - cfg.sim.gripper_force) < 1.0
    assert env.data.body(CUPS[1].name).xpos[2] > th + 0.05, "cup should be lifted"


def test_fingers_move_at_franka_speed(env):
    env.reset(seed=1)
    width = lambda: env.data.qpos[env.finger_qadr[0]].sum()  # noqa: E731
    w0 = width()
    for _ in range(10):  # 0.2 s
        env.apply_action(HOME_Q, np.array([0.0]))
        env.step()
    per_finger = (w0 - width()) / 2 / 0.2
    assert 0.04 < per_finger < 0.055, per_finger


def test_reset_is_seeded(env):
    a = env.reset(seed=7)["objects"]
    b = env.reset(seed=7)["objects"]
    c = env.reset(seed=8)["objects"]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_ik_reaches_targets(cfg):
    ik = ArmIK(cfg.sim, cfg.ik, 0.02)
    rng = np.random.default_rng(0)
    p0, q0 = ik.tcp_pose("r")
    for _ in range(5):
        ik.set_target("r", p0 + rng.uniform([-0.1, -0.1, -0.15], [0.1, 0.1, 0.05]), q0)
        for _ in range(100):
            ik.solve()
        for a, (pos_err, rot_err) in ik.errors().items():
            assert pos_err < 0.005 and rot_err < np.deg2rad(3), (a, pos_err, rot_err)


def test_set_state_replay_determinism(env):
    env.reset(seed=1)
    init = env.get_state()
    ctrls = []
    rng = np.random.default_rng(0)
    q = env.data.ctrl[env.arm_act].copy()
    for _ in range(60):
        q = q + rng.normal(scale=0.01, size=7)
        env.apply_action(q, rng.uniform(0, 1, 1))
        ctrls.append(env.data.ctrl.copy())
        env.step()
    final = env.data.qpos.copy()
    env.set_state(init)
    for c in ctrls:
        env.data.ctrl[:] = c
        env.step()
    assert np.allclose(env.data.qpos, final, atol=1e-9)


def test_scene_export(env):
    scene, blob = export_scene(env.model)
    assert scene["nbody"] == env.model.nbody
    assert {g["type"] for g in scene["geoms"]} >= {"mesh", "box", "plane"}
    for m in scene["meshes"]:
        assert m["faceOffset"] + m["nface"] * 12 <= len(blob)
        faces = np.frombuffer(blob, "<u4", m["nface"] * 3, m["faceOffset"])
        assert faces.max() < m["nvert"]
    assert len(blob) == scene["blobSize"]


def _fake_step(env, i):
    return {
        "time": float(i), "wall_time": time.monotonic(),
        "obs": env.observe(),
        "action": {"joint_pos": np.zeros(8), "ee_pose": np.zeros(8)},
        "sim": {"state": env.get_state()},
        "success": False,
        "teleop": {"head": np.zeros(7), "engaged": np.zeros(1, bool), "tracked": np.ones(1, bool), "rtt_ms": 12.0,
                   "left": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)},
                   "right": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)}},
    }


def _no_countdown(cfg):
    cfg = cfg.model_copy(deep=True)
    cfg.record.countdown_s = 0.0
    return cfg


def test_recorder_state_machine_and_hdf5(cfg, env, tmp_path):
    env.reset(seed=5)
    rec = Recorder(_no_countdown(cfg), tmp_path)
    assert rec.start(0.0) and rec.state == RecState.COUNTDOWN
    rec.tick(0.01)
    assert rec.state == RecState.RECORDING
    for i in range(10):
        rec.append(i * 0.02, _fake_step(env, i), success=False)
        env.step()
    assert rec.stop() and rec.state == RecState.REVIEW
    path = rec.save(env, "controllers", success=True)
    rec.flush()
    assert rec.state == RecState.IDLE
    meta, data = read_episode(path)
    assert meta["success"] is True and meta["num_steps"] == 10 and meta["task"] == "stack_cups"
    assert data["obs/joint_pos"].shape == (10, 7) and data["action/joint_pos"].shape == (10, 8)
    assert data["teleop/left/joints"].shape == (10, 25, 7)
    assert data["sim/state"].shape == (10, env.state_size)
    assert "<mujoco" in meta["model_xml"]
    rows = list_episodes(tmp_path)
    assert len(rows) == 1 and rows[0]["success"]
    # Discarding drops the take.
    rec.start(0.0); rec.tick(0.0); rec.append(0.0, _fake_step(env, 0), False)
    assert rec.discard() and rec.state == RecState.IDLE and rec.steps == []
    rec.close()


def test_recorder_auto_stop_on_success(cfg, env, tmp_path):
    env.reset(seed=5)
    rec = Recorder(_no_countdown(cfg), tmp_path)
    rec.start(0.0)
    rec.tick(0.0)
    stopped = [rec.append(t, _fake_step(env, 0), success=t >= 0.5) for t in np.arange(0, 2.0, 0.02)]
    assert any(stopped) and rec.state == RecState.REVIEW and rec.success_seen
    rec.close()


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") is None, reason="needs headless GL (MUJOCO_GL=egl)")
def test_lerobot_export(cfg, env, tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from questcollect.lerobot_export import export

    rec = Recorder(_no_countdown(cfg), tmp_path / "raw")
    for ep in range(2):
        env.reset(seed=ep)
        rec.start(0.0); rec.tick(0.0)
        for i in range(12):
            rec.append(i * 0.02, _fake_step(env, i), False)
            env.step()
        rec.save(env, "controllers", success=True)
        rec.flush()
    n = export(tmp_path / "raw", tmp_path / "lerobot", "test/quest", height=96, width=128)
    assert n == 2
    ds = LeRobotDataset("test/quest", root=tmp_path / "lerobot")
    assert ds.num_episodes == 2 and ds.num_frames == 24 and ds.fps == cfg.sim.control_hz
    item = ds[0]
    assert item["observation.state"].shape == (8,) and item["action"].shape == (8,)
    assert item["observation.images.left"].shape == (3, 96, 128)
    assert item["observation.environment_state"].shape == (7 * len(CUPS),)


def test_stack_haptics(cfg):
    """Each stacked cup fires the double pulse once, the finished stack the triple pulse."""
    rt = Runtime(cfg)
    events = []
    rt.add_listener(events.append)
    env = rt.env
    for level, c in enumerate(CUPS):
        set_cup_pose(env.data, c.name, [0.45, 0.0, env.table_height + level * (NEST_DZ + 0.002)])
    for t in range(60):
        env.step()
        rt._task_feedback(t * 0.02, env.success())
    pulses = [(e["intensity"], e["ms"], e["repeat"], e["gap_ms"]) for e in events if e["type"] == "haptic"]
    assert pulses.count(HAPTIC_PLACED) == LEVELS  # no holder recorded -> the arm's hand
    assert pulses.count(HAPTIC_COMPLETE) == 1
    sounds = [e["name"] for e in events if e["type"] == "sound"]
    assert sounds.count("placed") == LEVELS and sounds.count("complete") == 1 and "area" not in sounds
