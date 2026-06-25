"""
egodex_visualize_skeleton.py
-----------------------------
Visualize EgoDex hand skeleton motion as an animated 3D plot.

Produces:
  - An MP4 animation of both hands over time
  - A static PNG of the wrist trajectory

Usage:
  python scripts/egodex_visualize_skeleton.py \
      --hdf5 /home/grease/ego_dataset/work_bearlu/egodex/test/basic_pick_place/0.hdf5 \
      --out  /tmp/skeleton_basic_pick_place_ep0.mp4

  # Just preview every Nth frame as a static figure (no video):
  python scripts/egodex_visualize_skeleton.py \
      --hdf5 /home/grease/ego_dataset/work_bearlu/egodex/test/basic_pick_place/0.hdf5 \
      --static --frame 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — writes to file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------------
# Joint ordering — must match egodex_extract_egoscale.py
# ---------------------------------------------------------------------------
LEFT_HAND_JOINTS = [
    "leftHand",
    "leftThumbKnuckle", "leftThumbIntermediateBase",
    "leftThumbIntermediateTip", "leftThumbTip",
    "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase", "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    "leftMiddleFingerMetacarpal", "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase", "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    "leftRingFingerMetacarpal", "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase", "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    "leftLittleFingerMetacarpal", "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase", "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]
RIGHT_HAND_JOINTS = [j.replace("left", "right").replace("Left", "Right")
                     for j in LEFT_HAND_JOINTS]

# Bone connectivity: list of (parent_idx, child_idx) pairs
# Index layout: 0=wrist, 1-4=thumb, 5-9=index, 10-14=middle,
#               15-19=ring, 20-24=little
FINGER_BONES: list[tuple[int, int]] = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    # middle
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    # ring
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    # little
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
    # palm cross-bar (metacarpal bases)
    (5, 10), (10, 15), (15, 20),
]

# One color per finger
FINGER_COLORS = {
    "thumb":  "#FF6B6B",
    "index":  "#4ECDC4",
    "middle": "#45B7D1",
    "ring":   "#96CEB4",
    "little": "#FFEAA7",
    "palm":   "#DDA0DD",
}

def bone_color(parent: int, child: int) -> str:
    if parent <= 4 or child <= 4:
        return FINGER_COLORS["thumb"]
    if parent <= 9 or child <= 9:
        return FINGER_COLORS["index"]
    if parent <= 14 or child <= 14:
        return FINGER_COLORS["middle"]
    if parent <= 19 or child <= 19:
        return FINGER_COLORS["ring"]
    if parent <= 24 or child <= 24:
        return FINGER_COLORS["little"]
    return FINGER_COLORS["palm"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_keypoints(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load hand keypoints from HDF5.

    Returns:
        kp_left:  (T, 25, 3)  3D positions of left  hand joints in world frame
        kp_right: (T, 25, 3)  3D positions of right hand joints in world frame
        confs:    (T, 2)      wrist confidence [left, right]
        meta:     dict of episode metadata
    """
    with h5py.File(hdf5_path, "r") as f:
        # Extract 3D position = translation column of SE(3) matrix
        kp_left = np.stack(
            [f[f"transforms/{j}"][:, :3, 3] for j in LEFT_HAND_JOINTS], axis=1
        )   # (T, 25, 3)
        kp_right = np.stack(
            [f[f"transforms/{j}"][:, :3, 3] for j in RIGHT_HAND_JOINTS], axis=1
        )   # (T, 25, 3)

        conf_left  = f["confidences/leftHand"][:] if "confidences/leftHand" in f else np.ones(kp_left.shape[0], dtype=np.float32)
        conf_right = f["confidences/rightHand"][:] if "confidences/rightHand" in f else np.ones(kp_right.shape[0], dtype=np.float32)
        confs = np.stack([conf_left, conf_right], axis=1)  # (T, 2)

        meta = {
            "task":        str(f.attrs.get("task", "")),
            "episode":     int(hdf5_path.stem),
            "language":    str(f.attrs.get("llm_description") or f.attrs.get("description", "")),
            "environment": str(f.attrs.get("environment", "")),
            "n_frames":    kp_left.shape[0],
        }

    return kp_left, kp_right, confs, meta


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hand(ax: Axes3D, joints: np.ndarray,
              alpha: float = 1.0, zorder: int = 3) -> None:
    """Draw one hand skeleton on a 3D axes.

    Args:
        ax:     matplotlib 3D axes
        joints: (25, 3) joint positions
        alpha:  transparency
    """
    # Draw joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=18, c="white", edgecolors="gray",
               linewidths=0.5, zorder=zorder + 1, alpha=alpha)

    # Draw bones
    for p_idx, c_idx in FINGER_BONES:
        p = joints[p_idx]
        c = joints[c_idx]
        color = bone_color(p_idx, c_idx)
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                color=color, linewidth=2.0, alpha=alpha, zorder=zorder)

    # Highlight wrist
    ax.scatter(*joints[0], s=50, c="white", edgecolors="black",
               linewidths=1.5, zorder=zorder + 2, alpha=alpha)


def set_axes_equal(ax: Axes3D, kp_all: np.ndarray) -> None:
    """Set equal aspect ratio for 3D axes around the data extent."""
    mins = kp_all.reshape(-1, 3).min(axis=0)
    maxs = kp_all.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    half   = (maxs - mins).max() / 2 * 0.6

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def style_axes(ax: Axes3D) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(True, color="white", alpha=0.08, linewidth=0.5)
    ax.set_xlabel("X (m)", color="white", fontsize=8, labelpad=4)
    ax.set_ylabel("Y (m)", color="white", fontsize=8, labelpad=4)
    ax.set_zlabel("Z (m)", color="white", fontsize=8, labelpad=4)
    ax.tick_params(colors="white", labelsize=7)


# ---------------------------------------------------------------------------
# Static frame plot
# ---------------------------------------------------------------------------

def plot_static_frame(kp_left: np.ndarray, kp_right: np.ndarray,
                      frame: int, meta: dict,
                      confs: np.ndarray, out_path: Path) -> None:
    """Render a single frame of both hands as a static PNG."""
    fig = plt.figure(figsize=(12, 6), facecolor="#1a1a2e")

    # ── Left hand ────────────────────────────────────────────────────────────
    ax_l = fig.add_subplot(121, projection="3d")
    ax_l.set_facecolor("#1a1a2e")
    draw_hand(ax_l, kp_left[frame])
    style_axes(ax_l)
    all_kp = np.concatenate([kp_left, kp_right], axis=1)
    set_axes_equal(ax_l, all_kp)
    ax_l.set_title(f"Left Hand  conf={confs[frame, 0]:.2f}",
                   color="white", fontsize=10, pad=6)

    # ── Right hand ───────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(122, projection="3d")
    ax_r.set_facecolor("#1a1a2e")
    draw_hand(ax_r, kp_right[frame])
    style_axes(ax_r)
    set_axes_equal(ax_r, all_kp)
    ax_r.set_title(f"Right Hand  conf={confs[frame, 1]:.2f}",
                   color="white", fontsize=10, pad=6)

    # ── Super title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"[{meta['task']}  ep {meta['episode']}]  frame {frame}/{meta['n_frames']-1}\n"
        f"{meta['language']}",
        color="white", fontsize=9, y=1.02, wrap=True,
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=FINGER_COLORS["thumb"],  linewidth=2, label="Thumb"),
        Line2D([0], [0], color=FINGER_COLORS["index"],  linewidth=2, label="Index"),
        Line2D([0], [0], color=FINGER_COLORS["middle"], linewidth=2, label="Middle"),
        Line2D([0], [0], color=FINGER_COLORS["ring"],   linewidth=2, label="Ring"),
        Line2D([0], [0], color=FINGER_COLORS["little"], linewidth=2, label="Little"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Static frame saved → {out_path}")


# ---------------------------------------------------------------------------
# Wrist trajectory plot
# ---------------------------------------------------------------------------

def plot_wrist_trajectory(kp_left: np.ndarray, kp_right: np.ndarray,
                          meta: dict, out_path: Path) -> None:
    """Plot 3D wrist trajectories (paths over time) for both hands."""
    T = kp_left.shape[0]
    t_norm = np.linspace(0, 1, T)   # color maps time 0→1

    fig = plt.figure(figsize=(10, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")

    # Draw trajectory as colored line segments (early=dark, late=bright)
    for side, kp, cmap_name, label in [
        ("Left",  kp_left,  "Blues",  "Left wrist"),
        ("Right", kp_right, "Oranges","Right wrist"),
    ]:
        wrist = kp[:, 0, :]  # (T, 3) — joint 0 = wrist
        cmap  = plt.get_cmap(cmap_name)
        for i in range(T - 1):
            c = cmap(0.3 + 0.7 * t_norm[i])
            ax.plot(wrist[i:i+2, 0], wrist[i:i+2, 1], wrist[i:i+2, 2],
                    color=c, linewidth=2.0, alpha=0.9)
        # Start/end markers
        ax.scatter(*wrist[0],  s=80, color=cmap(0.4), marker="o",
                   edgecolors="white", linewidths=1, zorder=5, label=f"{side} start")
        ax.scatter(*wrist[-1], s=80, color=cmap(0.9), marker="*",
                   edgecolors="white", linewidths=1, zorder=5, label=f"{side} end")

    style_axes(ax)
    all_kp = np.concatenate([kp_left[:, 0], kp_right[:, 0]], axis=0)
    set_axes_equal(ax, all_kp[:, np.newaxis])

    ax.set_title(
        f"Wrist Trajectories  [{meta['task']}  ep {meta['episode']}]\n"
        f"{meta['language']}",
        color="white", fontsize=9, pad=8,
    )
    ax.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="none",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Trajectory plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def make_animation(kp_left: np.ndarray, kp_right: np.ndarray,
                   confs: np.ndarray, meta: dict,
                   out_path: Path, fps: int = 30,
                   stride: int = 1) -> None:
    """Render an animated MP4 of both hands side-by-side.

    Args:
        stride: render every Nth frame (use stride>1 to speed up rendering)
    """
    T       = kp_left.shape[0]
    frames  = list(range(0, T, stride))
    all_kp  = np.concatenate([kp_left, kp_right], axis=1)

    fig = plt.figure(figsize=(12, 6), facecolor="#1a1a2e")
    ax_l = fig.add_subplot(121, projection="3d")
    ax_r = fig.add_subplot(122, projection="3d")

    for ax in (ax_l, ax_r):
        style_axes(ax)
        set_axes_equal(ax, all_kp)

    # Title text objects for live update
    title = fig.suptitle("", color="white", fontsize=9)

    # ── Legend (static) ──────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=FINGER_COLORS["thumb"],  linewidth=2, label="Thumb"),
        Line2D([0], [0], color=FINGER_COLORS["index"],  linewidth=2, label="Index"),
        Line2D([0], [0], color=FINGER_COLORS["middle"], linewidth=2, label="Middle"),
        Line2D([0], [0], color=FINGER_COLORS["ring"],   linewidth=2, label="Ring"),
        Line2D([0], [0], color=FINGER_COLORS["little"], linewidth=2, label="Little"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    def update(frame_idx: int):
        t = frames[frame_idx]
        for ax, kp, side, c_idx in [
            (ax_l, kp_left,  "Left",  0),
            (ax_r, kp_right, "Right", 1),
        ]:
            ax.cla()
            style_axes(ax)
            set_axes_equal(ax, all_kp)

            # Wrist trail: draw last 15 frames as fading dots
            trail_start = max(0, t - 15)
            trail = kp[trail_start:t+1, 0, :]  # wrist positions
            if len(trail) > 1:
                alphas = np.linspace(0.15, 0.6, len(trail))
                for i, (pos, alpha) in enumerate(zip(trail, alphas)):
                    ax.scatter(*pos, s=8, c=[[0.6, 0.6, 1.0, alpha]], zorder=2)

            draw_hand(ax, kp[t])
            conf = confs[t, c_idx]
            conf_color = "lime" if conf >= 0.8 else "orange" if conf >= 0.5 else "red"
            ax.set_title(f"{side} Hand  conf=",
                         color="white", fontsize=9, pad=4)
            ax.set_title(f"{side} Hand  conf={conf:.2f}",
                         color=conf_color, fontsize=9, pad=4)

        title.set_text(
            f"[{meta['task']}  ep {meta['episode']}]  "
            f"frame {t}/{T-1}  ({t/30:.2f}s)\n"
            f"{meta['language']}"
        )
        return []

    n_frames = len(frames)
    print(f"Rendering {n_frames} frames → {out_path} …")
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Animation saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize EgoDex hand skeleton motion."
    )
    parser.add_argument("--hdf5", type=Path,
                        default=Path("/home/grease/ego_dataset/work_bearlu/egodex/test"
                                     "/basic_pick_place/0.hdf5"),
                        help="Path to one EgoDex .hdf5 episode file")
    parser.add_argument("--out",  type=Path,
                        default=Path("/home/grease/ego_dataset/work_bearlu/egodata_tools/outputs/skeleton_anim.mp4"),
                        help="Output MP4 path")
    parser.add_argument("--fps",  type=int, default=30,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Render every Nth frame (default: 1 = all frames)")
    parser.add_argument("--static", action="store_true",
                        help="Render a static PNG of one frame instead of animation")
    parser.add_argument("--frame",  type=int, default=None,
                        help="Frame index for --static (default: middle frame)")
    parser.add_argument("--trajectory", action="store_true",
                        help="Also save a wrist-trajectory plot PNG")
    args = parser.parse_args()

    print(f"Loading {args.hdf5} …")
    kp_left, kp_right, confs, meta = load_keypoints(args.hdf5)
    T = meta["n_frames"]
    print(f"  Task: {meta['task']}  Episode: {meta['episode']}")
    print(f"  Language: {meta['language']}")
    print(f"  Frames: {T}  ({T/30:.1f}s at 30 FPS)")
    print(f"  Left  conf: mean={confs[:,0].mean():.3f}  min={confs[:,0].min():.3f}")
    print(f"  Right conf: mean={confs[:,1].mean():.3f}  min={confs[:,1].min():.3f}")

    if args.static:
        frame = args.frame if args.frame is not None else T // 2
        out_png = args.out.with_suffix(".png")
        plot_static_frame(kp_left, kp_right, frame, meta, confs, out_png)
    else:
        make_animation(kp_left, kp_right, confs, meta,
                       args.out, fps=args.fps, stride=args.stride)

    if args.trajectory:
        traj_path = args.out.with_name(args.out.stem + "_trajectory.png")
        plot_wrist_trajectory(kp_left, kp_right, meta, traj_path)


if __name__ == "__main__":
    main()
