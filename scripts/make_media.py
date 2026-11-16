"""Renders the README media in docs/media/ from the simulator (no headset needed).

    MUJOCO_GL=egl uv run python scripts/make_media.py

- demo.gif      scripted stack_cups demo (green into red) (IK-driven, same control stack as teleop)
- cups.png      the two cups and the goal stack (close-ups)
- cameras.png   the three recorded camera views (what goes into the LeRobot dataset)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from questcollect.config import ROOT, load_config
from questcollect.ik import ArmIK
from questcollect.sim import CUP_H, CUPS, LEVELS, NEST_DZ, CameraRenderer, PandaEnv, set_cup_pose

OUT = ROOT / "docs" / "media"
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # TCP pointing at the table (wxyz)


def free_camera(distance=1.75, azimuth=180.0, elevation=-28.0, lookat=(0.35, 0.0, 0.92)) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.height // 18))
    pad = img.height // 40
    box = d.textbbox((0, 0), text, font=font)
    d.rounded_rectangle((pad, pad, pad * 3 + box[2], pad * 3 + box[3]), radius=pad, fill=(10, 14, 24))
    d.text((pad * 2, pad * 2), text, font=font, fill=(232, 238, 252))
    return img


# Deterministic demo layout: cup -> (x, y). The red cup stays where it is; the green one goes into it.
DEMO_LAYOUT = {"red_cup": (0.45, 0.05), "green_cup": (0.56, -0.2)}
GRASP_Z = CUP_H - 0.012  # just below the rim: lower, the fingers would hit the rim of the cup below when nesting


def stack_keys(env, ik) -> list[tuple]:
    """Keyframes (seconds, pose, grip) that put each cup into the one below it, on the red (base) cup."""
    th = env.table_height
    lift = 0.16
    home = (ik.tcp_pose("r")[0], DOWN)
    pose, grip, t = home, 1, 0.0
    keys = [(t, pose, grip)]

    def key(dt, p=None, q=None, g=None):
        nonlocal t, pose, grip
        t += dt
        if p is not None:
            pose = (np.asarray(p, float), pose[1] if q is None else q)
        if g is not None:
            grip = g
        keys.append((t, pose, grip))

    base = DEMO_LAYOUT[CUPS[0].name]
    for k, c in enumerate(CUPS[1:], start=1):
        x, y = DEMO_LAYOUT[c.name]
        gx, gy, gz = *base, th + k * NEST_DZ
        key(1.2, [x, y, th + lift], DOWN)
        key(0.8, [x, y, th + GRASP_Z])
        key(0.6, g=0)
        key(0.8, [x, y, th + lift])
        key(1.2, [gx, gy, gz + lift])
        key(0.9, [gx, gy, gz + GRASP_Z + 0.01])  # let it drop the last centimetre into the cup below
        key(0.6, g=1)
        key(0.8, [gx, gy, gz + lift])
    key(1.0, *home)
    key(1.0)
    return keys


def gif_palette(frames: list[Image.Image], colors: int = 160) -> Image.Image:
    """Palette from frames across the whole clip plus each cup colour from shadow to highlight: the cups
    cover a few hundred pixels, so a palette from the frames alone renders them grey."""
    w, h = frames[0].size
    picks = frames[:: max(1, len(frames) // 6)][:6]
    shades = np.linspace(0.25, 1.15, 24)
    swatch = np.array([[np.clip(np.array(c.rgba[:3]) * s, 0, 1) for s in shades] for c in CUPS])
    swatch = Image.fromarray((swatch * 255).astype(np.uint8)).resize((w, h // 2), Image.Resampling.NEAREST)
    src = Image.new("RGB", (w, h * len(picks) + swatch.height))
    for i, f in enumerate(picks):
        src.paste(f, (0, i * h))
    src.paste(swatch, (0, h * len(picks)))
    return src.quantize(colors=colors)


def demo_gif(cfg, path: Path | None, size=(640, 400), every=5) -> bool:
    env = PandaEnv(cfg)
    env.reset(0)
    th = env.table_height
    for name, (x, y) in DEMO_LAYOUT.items():
        set_cup_pose(env.data, name, [x, y, th])
    mujoco.mj_forward(env.model, env.data)
    ik = ArmIK(cfg.sim, cfg.ik, env.control_dt)
    keys = stack_keys(env, ik)
    times = np.array([k[0] for k in keys])

    def interp(t, i):
        j = min(np.searchsorted(times, t, side="right"), len(keys) - 1)
        k0, k1 = keys[j - 1], keys[j]
        s = np.clip((t - k0[0]) / max(k1[0] - k0[0], 1e-6), 0, 1)
        s = s * s * (3 - 2 * s)  # smoothstep
        if i == 2:
            return k0[i] + s * (k1[i] - k0[i])
        (p0, q0), (p1, q1) = k0[i], k1[i]
        q = (1 - s) * (q0 if np.dot(q0, q1) >= 0 else -q0) + s * q1  # nlerp; keyframes are close
        return p0 + s * (p1 - p0), q / np.linalg.norm(q)

    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0]) if path else None
    cam = free_camera(distance=1.0, elevation=-38.0, lookat=(0.4, 0.0, 0.85))
    frames = []
    best = 0
    for step in range(int(times[-1] / env.control_dt)):
        t = step * env.control_dt
        ik.set_target("r", *interp(t, 1))
        ik.follow(env.data.qpos[env.arm_qadr])
        env.apply_action(ik.solve(), np.array([interp(t, 2)]))
        env.step()
        best = max(best, env.progress())
        if renderer and step % every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
    print(f"demo progress: {env.progress()}/{LEVELS} (best {best}), success: {env.success()}")
    if renderer:
        renderer.close()
        # One shared palette + a static camera lets the GIF encoder store only what changed per frame.
        palette = gif_palette(frames)
        pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 * env.control_dt * every),
                    loop=0, optimize=True)
    return env.success()


def cups_png(cfg, path: Path, size=(480, 360)) -> None:
    """Two close-ups: the cups, and the finished stack."""
    env = PandaEnv(cfg)
    env.reset(0)
    th, d = env.table_height, env.data
    x0 = 0.75  # near the far table edge, away from the arm
    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])

    def shot(poses: dict, cam: mujoco.MjvCamera) -> Image.Image:
        for name, pos in poses.items():
            set_cup_pose(d, name, pos)
        mujoco.mj_forward(env.model, d)
        renderer.update_scene(d, camera=cam)
        return Image.fromarray(renderer.render())

    panels = [
        label(shot({c.name: [x0, 0.09 - 0.09 * i, th] for i, c in enumerate(CUPS)},
                   free_camera(0.4, 160, -30, (x0, 0.0, th + 0.03))), "The 2 cups"),
        label(shot({c.name: [x0, 0.0, th + i * NEST_DZ] for i, c in enumerate(CUPS)},
                   free_camera(0.3, 150, -25, (x0, 0.0, th + 0.05))), "Goal: green on top of red"),
    ]
    renderer.close()
    sheet = Image.new("RGB", (size[0] * len(panels), size[1]))
    for i, img in enumerate(panels):
        sheet.paste(img, (i * size[0], 0))
    sheet.save(path, optimize=True)


def cameras_png(cfg, path: Path, size=(320, 240)) -> None:
    env = PandaEnv(cfg)
    env.reset(2)
    imgs = CameraRenderer(env.model, size[1], size[0], cfg.record.cameras).render(env.data)
    sheet = Image.new("RGB", (size[0] * len(imgs), size[1]))
    for i, (name, img) in enumerate(imgs.items()):
        sheet.paste(label(Image.fromarray(img), name), (i * size[0], 0))
    sheet.save(path, optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    demo_gif(cfg, OUT / "demo.gif")
    cups_png(cfg, OUT / "cups.png")
    cameras_png(cfg, OUT / "cameras.png")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<16} {p.stat().st_size / 1e6:.2f} MB")
