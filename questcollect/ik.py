"""Differential IK for the arm with mink (one QP with joint and velocity limits)."""

from __future__ import annotations

import mink
import numpy as np

from questcollect.config import IKConfig, SimConfig
from questcollect.sim import ARM_JOINTS, ARMS, HOME_Q, build_spec

class ArmIK:
    """Keeps its own kinematic configuration (robots only) that tracks EE targets smoothly."""

    def __init__(self, sim_cfg: SimConfig, cfg: IKConfig, dt: float):
        self.cfg = cfg
        self.dt = dt
        self.model = build_spec(sim_cfg, with_task=False).compile()
        self.configuration = mink.Configuration(self.model)
        m = self.model
        self.arm_qadr = np.array([m.joint(f"{a}_{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])

        self.frame_tasks = {
            a: mink.FrameTask(f"{a}_tcp", "site", position_cost=cfg.position_cost,
                              orientation_cost=cfg.orientation_cost, lm_damping=1.0)
            for a in ARMS
        }
        self.posture = mink.PostureTask(m, cost=cfg.posture_cost)
        self.tasks = [*self.frame_tasks.values(), self.posture]
        vel = {f"{a}_{j}": cfg.max_joint_vel for a in ARMS for j in ARM_JOINTS}
        self.limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel)]
        self.reset(np.tile(HOME_Q, len(ARMS)))

    def reset(self, arm_q: np.ndarray) -> None:
        q = self.model.qpos0.copy()
        q[self.arm_qadr] = arm_q
        self.configuration.update(q)
        self.posture.set_target(q)
        for a in ARMS:
            self.frame_tasks[a].set_target(self.configuration.get_transform_frame_to_world(f"{a}_tcp", "site"))

    def follow(self, arm_q: np.ndarray) -> None:
        """Keep the IK within max_lead rad of the measured joint angles (call before solve).

        The IK has no obstacles; physics does. When the arm is blocked (table, a cup), the IK would otherwise run on through the obstacle while the arm stays put,
        then yank the arm across when it comes free. In free motion the arm trails its target by about
        speed x kd/kp (~0.1 s), so a lead of 0.15 rad leaves moves up to ~1.5 rad/s untouched.
        """
        q = self.configuration.q.copy()
        lead = q[self.arm_qadr] - arm_q
        q[self.arm_qadr] = arm_q + np.clip(lead, -self.cfg.max_lead, self.cfg.max_lead)
        self.configuration.update(q)

    def tcp_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        T = self.configuration.get_transform_frame_to_world(f"{arm}_tcp", "site")
        return T.translation().copy(), T.rotation().wxyz.copy()

    def set_target(self, arm: str, pos: np.ndarray, wxyz: np.ndarray) -> None:
        T = mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(wxyz, float)), np.asarray(pos, float))
        self.frame_tasks[arm].set_target(T)

    def solve(self) -> np.ndarray:
        """Advance the IK configuration one control step; returns the (7,) joint targets."""
        sub_dt = self.dt / self.cfg.iterations
        for _ in range(self.cfg.iterations):
            v = mink.solve_ik(self.configuration, self.tasks, sub_dt, solver="daqp",
                              damping=1e-3, limits=self.limits)
            self.configuration.integrate_inplace(v, sub_dt)
        return self.configuration.q[self.arm_qadr].copy()

    def errors(self) -> dict[str, tuple[float, float]]:
        """Per-arm (position error m, orientation error rad) to the current target."""
        out = {}
        for a, t in self.frame_tasks.items():
            e = t.compute_error(self.configuration)
            out[a] = (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
        return out
