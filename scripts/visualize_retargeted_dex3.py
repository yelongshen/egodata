"""
visualize_retargeted_dex3.py
-----------------------------
Visualize geometric retargeting results for Unitree Dex3-1 (7 DoF).

For each EgoDex HDF5 sample:
  1. Runs geometric retargeting → (T, 7) joint angles
  2. Evaluates FK via yourdfpy at each frame → 8 link positions
  3. Renders side-by-side: human hand skeleton  |  Dex3-1 robot skeleton
  4. Saves MP4 animation + static PNG

Usage:
  python scripts/visualize_retargeted_dex3.py
  python scripts/visualize_retargeted_dex3.py --hdf5 samples/egodex/sort_beads/9.hdf5
  python scripts/visualize_retargeted_dex3.py --static --frame 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

REPO_ROOT = Path(__file__).parent.parent

# ─── Dex3-1 joint spec ────────────────────────────────────────────────────────
_LIMITS = [
    (math.radians(-60.0), math.radians(60.0)),
    (math.radians(-60.0), math.radians(35.0)),
    (math.radians(-100.0), math.radians(0.0)),
    (math.radians(0.0),  math.radians(90.0)),
    (math.radians(0.0),  math.radians(100.0)),
    (math.radians(0.0),  math.radians(90.0)),
    (math.radians(0.0),  math.radians(100.0)),
]
_LO = np.array([l for l, _ in _LIMITS], np.float32)
_HI = np.array([h for _, h in _LIMITS], np.float32)
JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2",
               "index_0", "index_1", "middle_0", "middle_1"]
URDF_JOINT_NAMES = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
URDF_PATH = REPO_ROOT / "assets/dex3_1/dex3_1_r.urdf"
PALM_LINK = "right_hand_palm_link"
ALL_LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
    "right_hand_index_0_link", "right_hand_index_1_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
]
# Bone connectivity: (parent_link_idx, child_link_idx) in ALL_LINKS
BONES = [
    (0, 1), (1, 2), (2, 3),   # thumb
    (0, 4), (4, 5),            # index
    (0, 6), (6, 7),            # middle
    # palm cross bar
    (4, 6),
]
FINGER_COLORS = {
    "thumb":  "#FF6B6B",
    "index":  "#4ECDC4",
    "middle": "#FFD93D",   # yellow — clearly distinct from teal index
    "palm":   "#888888",
}

def bone_color(p: int, c: int) -> str:
    if p <= 3 or c <= 3: return FINGER_COLORS["thumb"]
    if p <= 5 or c <= 5: return FINGER_COLORS["index"]
    if p <= 7 or c <= 7: return FINGER_COLORS["middle"]
    return FINGER_COLORS["palm"]


def urdf_to_display(links: np.ndarray) -> np.ndarray:
    """Identity — raw URDF coords used directly; matplotlib Z is vertical.
    URDF frame:  X=extension(forward), Y=curl(up), Z=lateral(index+,middle-)
    Matplotlib:  X=horizontal, Y=depth, Z=vertical
    Best viewed at elev=30, azim=-60 (right-front-above).
    """
    return links  # no transform needed


# ─── EgoDex joints ───────────────────────────────────────────────────────────
RIGHT_HAND_JOINTS = [
    "rightHand",
    "rightThumbKnuckle", "rightThumbIntermediateBase",
    "rightThumbIntermediateTip", "rightThumbTip",
    "rightIndexFingerMetacarpal", "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase", "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",
    "rightMiddleFingerMetacarpal", "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase", "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",
    "rightRingFingerMetacarpal", "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase", "rightRingFingerIntermediateTip",
    "rightRingFingerTip",
    "rightLittleFingerMetacarpal", "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase", "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]
# Human bone connectivity for 25-joint hand
HUMAN_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),(8,9),
    (0,10),(10,11),(11,12),(12,13),(13,14),
    (0,15),(15,16),(16,17),(17,18),(18,19),
    (0,20),(20,21),(21,22),(22,23),(23,24),
    (5,10),(10,15),(15,20),
]


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_hdf5(path: Path):
    with h5py.File(path) as f:
        kp_se3 = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_HAND_JOINTS], axis=1)
        kp_pos = kp_se3[:, :, :3, 3]
        conf   = (f["confidences/rightHand"][:] if "confidences/rightHand" in f
                  else np.ones(kp_pos.shape[0]))
        meta = {
            "task":     str(f.attrs.get("task", path.parent.name)),
            "episode":  int(path.stem),
            "language": str(f.attrs.get("llm_description") or f.attrs.get("description", "")),
        }
    return kp_pos, kp_se3, conf, meta


# ─── Geometric retargeting ────────────────────────────────────────────────────
def _n(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, eps)

def _ang(v1, v2):
    return np.arccos(np.clip(np.sum(_n(v1)*_n(v2), axis=-1), -1., 1.))

def _proj_out(v, n):
    return v - np.sum(v*n, axis=-1, keepdims=True)*n

def _signed_ang(v1, v2, n):
    return np.arctan2(np.sum(np.cross(v1, v2)*n, axis=-1),
                      np.sum(v1*v2, axis=-1))

def retarget_geometric(kp_se3: np.ndarray, smoothing_alpha=0.3) -> np.ndarray:
    T = kp_se3.shape[0]
    wi = np.zeros_like(kp_se3[:, 0])
    wi[:, :3, :3] = kp_se3[:, 0, :3, :3].transpose(0, 2, 1)
    wi[:, :3, 3]  = -(kp_se3[:, 0, :3, :3].transpose(0, 2, 1) @
                      kp_se3[:, 0, :3, 3:]).squeeze(-1)
    wi[:, 3, 3]   = 1.0
    kp = np.einsum("tij,tkj->tki", wi[:, :3, :3],
                   kp_se3[:, :, :3, 3] - kp_se3[:, :1, :3, 3])

    palm_fwd  = _n(kp[:, 10])
    palm_lat  = _n(kp[:, 20] - kp[:, 5])
    palm_norm = _n(np.cross(palm_fwd, palm_lat))

    q = np.zeros((T, 7), np.float32)
    # Thumb
    b_cmc = _n(kp[:, 1])
    b_t1  = _n(kp[:, 2] - kp[:, 1])
    b_t2  = _n(kp[:, 4] - kp[:, 2])
    q[:,0] = _signed_ang(palm_fwd, _n(_proj_out(b_cmc, palm_norm)), palm_norm)
    q[:,1] = _ang(b_cmc, b_t1)
    q[:,2] = _ang(b_t1,  b_t2)
    # Index (kp: metacarpal=5, knuckle=6, tip=9)
    b_i0 = _n(kp[:,6]  - kp[:,5])
    b_i1 = _n(kp[:,9]  - kp[:,6])
    q[:,3] = _ang(b_i0, b_i1)
    q[:,4] = _ang(b_i0, b_i1) * 0.8
    # Middle (kp: metacarpal=10, knuckle=11, tip=14)
    b_m0 = _n(kp[:,11] - kp[:,10])
    b_m1 = _n(kp[:,14] - kp[:,11])
    q[:,5] = _ang(b_m0, b_m1)
    q[:,6] = _ang(b_m0, b_m1) * 0.8

    np.clip(q, _LO, _HI, out=q)
    for t in range(1, T):
        q[t] = smoothing_alpha * q[t-1] + (1 - smoothing_alpha) * q[t]
    return q


# ─── FK via yourdfpy ──────────────────────────────────────────────────────────
def build_fk():
    import yourdfpy
    robot = yourdfpy.URDF.load(str(URDF_PATH))
    def fk(q: np.ndarray) -> np.ndarray:
        """Returns (8, 3) link positions in palm frame."""
        cfg = {name: float(v) for name, v in zip(URDF_JOINT_NAMES, q)}
        robot.update_cfg(cfg)
        return np.array([robot.get_transform(l, PALM_LINK)[:3, 3] for l in ALL_LINKS])
    return fk

def compute_fk_sequence(q_seq: np.ndarray, fk_fn) -> np.ndarray:
    """(T, 7) → (T, 8, 3) link positions in URDF palm frame."""
    T = q_seq.shape[0]
    out = np.zeros((T, 8, 3), np.float32)
    for t in range(T):
        out[t] = fk_fn(q_seq[t])
    return out   # raw URDF coords; urdf_to_display() applied at draw time


# ─── Axes helpers ─────────────────────────────────────────────────────────────
def style_ax(ax, robot: bool = False):
    ax.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none"); ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(True, color="white", alpha=0.06, linewidth=0.4)
    ax.tick_params(colors="white", labelsize=6)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
        lbl.set_color("white"); lbl.set_fontsize(7)
    if robot:
        # elev=30, azim=-60: right-front-above view — fingers extend left in image,
        # index (teal,+Z) on top, middle (yellow,-Z) on bottom, curl (Y) visible
        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel("X ext", color="white", fontsize=6)
        ax.set_ylabel("Y curl", color="white", fontsize=6)
        ax.set_zlabel("Z lat",  color="white", fontsize=6)

def set_equal(ax, pts: np.ndarray):
    pts = pts.reshape(-1, 3)
    c = (pts.max(0) + pts.min(0)) / 2
    h = (pts.max(0) - pts.min(0)).max() / 2 * 0.65
    ax.set_xlim(c[0]-h, c[0]+h)
    ax.set_ylim(c[1]-h, c[1]+h)
    ax.set_zlim(c[2]-h, c[2]+h)

def draw_human(ax, kp: np.ndarray, alpha=0.5):
    """Draw 25-joint human hand (grey/faded) on ax."""
    for p, c in HUMAN_BONES:
        ax.plot([kp[p,0],kp[c,0]], [kp[p,1],kp[c,1]], [kp[p,2],kp[c,2]],
                color="#888888", lw=1.0, alpha=alpha)
    ax.scatter(kp[:,0], kp[:,1], kp[:,2], s=8, c="#888888", alpha=alpha, zorder=2)

def draw_robot(ax, links_urdf: np.ndarray, alpha=1.0):
    """Draw 8-link Dex3-1 skeleton in display coordinates."""
    links = urdf_to_display(links_urdf)
    for p, c in BONES:
        col   = bone_color(p, c)
        lw    = 1.2 if (p == 4 and c == 6) else 2.5   # palm cross-bar lighter
        a     = alpha * 0.4 if (p == 4 and c == 6) else alpha
        ax.plot([links[p,0],links[c,0]], [links[p,1],links[c,1]], [links[p,2],links[c,2]],
                color=col, lw=lw, alpha=a, zorder=4)
    ax.scatter(links[:,0], links[:,1], links[:,2],
               s=25, c="white", edgecolors="gray", linewidths=0.5, zorder=5, alpha=alpha)
    ax.scatter(*links[0], s=60, c="white", edgecolors="black", lw=1.5, zorder=6, alpha=alpha)


# ─── Joint angle chart ────────────────────────────────────────────────────────
def _joint_bar_ax(ax, q_t: np.ndarray, title: str):
    """Mini horizontal bar chart showing current joint angles vs limits."""
    ax.set_facecolor("#1a1a2e")
    colors = ["#FF6B6B"]*3 + ["#4ECDC4"]*2 + ["#45B7D1"]*2
    ranges = _HI - _LO
    pct    = (q_t - _LO) / np.where(ranges > 1e-4, ranges, 1.0)
    ax.barh(range(7), pct, color=colors, edgecolor="none", height=0.6, alpha=0.85)
    ax.barh(range(7), [1]*7, color="none", edgecolor="#555", height=0.6, linewidth=0.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels(
        [f"{n} {math.degrees(v):+.0f}°" for n, v in zip(JOINT_NAMES, q_t)],
        fontsize=6, color="white"
    )
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["lo", "50%", "hi"], fontsize=6, color="white")
    ax.set_title(title, color="white", fontsize=7, pad=3)
    for spine in ax.spines.values(): spine.set_edgecolor("#444")


# ─── Static frame ─────────────────────────────────────────────────────────────
def plot_static(kp_pos, kp_seq, q_seq, frame: int, meta: dict, out_path: Path):
    """Side-by-side: human kp | robot FK | joint bar chart."""
    kp_wrist_rel = kp_pos - kp_pos[:, 0:1, :]  # wrist-relative for human side

    fig = plt.figure(figsize=(18, 6), facecolor="#1a1a2e")
    fig.suptitle(
        f"Dex3-1 Retargeting  [{meta['task']}  ep {meta['episode']}]  frame {frame}\n"
        f"{meta['language']}",
        color="white", fontsize=9, y=1.01,
    )

    # ── Human hand ──────────────────────────────────────────────────────────
    ax_h = fig.add_subplot(131, projection="3d")
    ax_h.set_facecolor("#1a1a2e")
    draw_human(ax_h, kp_wrist_rel[frame], alpha=0.9)
    style_ax(ax_h)
    set_equal(ax_h, kp_wrist_rel)
    ax_h.set_title("Human keypoints", color="white", fontsize=9)

    # ── Robot FK ────────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(132, projection="3d")
    ax_r.set_facecolor("#1a1a2e")
    draw_robot(ax_r, kp_seq[frame])
    style_ax(ax_r, robot=True)
    set_equal(ax_r, urdf_to_display(kp_seq.reshape(-1, 3)).reshape(kp_seq.shape))
    ax_r.set_title("Dex3-1 FK  (geometric retargeting)", color="white", fontsize=9)

    # ── Joint bar chart ──────────────────────────────────────────────────────
    ax_j = fig.add_subplot(133)
    _joint_bar_ax(ax_j, q_seq[frame], f"Joint angles  (frame {frame})")

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], color=FINGER_COLORS["thumb"],  lw=2, label="Thumb (red)"),
           Line2D([0],[0], color=FINGER_COLORS["index"],  lw=2, label="Index (teal)"),
           Line2D([0],[0], color=FINGER_COLORS["middle"], lw=2, label="Middle (yellow)")]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static → {out_path}")


# ─── Animation ────────────────────────────────────────────────────────────────
def make_animation(kp_pos, kp_seq, q_seq, meta: dict, out_path: Path,
                   fps: int = 30, stride: int = 1):
    T      = kp_pos.shape[0]
    frames = list(range(0, T, stride))
    kp_wr  = kp_pos - kp_pos[:, 0:1, :]

    fig = plt.figure(figsize=(16, 6), facecolor="#1a1a2e")
    ax_h = fig.add_subplot(131, projection="3d")
    ax_r = fig.add_subplot(132, projection="3d")
    ax_j = fig.add_subplot(133)

    style_ax(ax_h)
    style_ax(ax_r, robot=True)

    kp_robot_disp = urdf_to_display(kp_seq.reshape(-1, 3)).reshape(kp_seq.shape)
    set_equal(ax_h, kp_wr)
    set_equal(ax_r, kp_robot_disp)
    ax_h.set_title("Human keypoints", color="white", fontsize=8)
    ax_r.set_title("Dex3-1 FK  (geometric)", color="white", fontsize=8)

    title = fig.suptitle("", color="white", fontsize=8)

    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], color=FINGER_COLORS["thumb"],  lw=2, label="Thumb (red)"),
           Line2D([0],[0], color=FINGER_COLORS["index"],  lw=2, label="Index (teal)"),
           Line2D([0],[0], color=FINGER_COLORS["middle"], lw=2, label="Middle (yellow)"),
           Line2D([0],[0], color="#888888", lw=1.5, alpha=0.6, label="Human (ref)")]
    fig.legend(handles=leg, loc="lower center", ncol=4,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=7, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()

    def update(fi: int):
        t = frames[fi]
        ax_h.cla(); ax_r.cla(); ax_j.cla()
        style_ax(ax_h); style_ax(ax_r, robot=True)
        set_equal(ax_h, kp_wr); set_equal(ax_r, kp_robot_disp)
        ax_h.set_title("Human keypoints", color="white", fontsize=8)
        ax_r.set_title("Dex3-1 FK  (geometric)", color="white", fontsize=8)

        draw_human(ax_h, kp_wr[t], alpha=0.9)
        draw_robot(ax_r, kp_seq[t])
        _joint_bar_ax(ax_j, q_seq[t], f"Joint angles  frame {t}/{T-1}")

        title.set_text(
            f"Dex3-1 Retargeting  [{meta['task']}  ep {meta['episode']}]  "
            f"frame {t}/{T-1}\n{meta['language']}"
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                    interval=1000//fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                     extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"  Animation → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_one(hdf5_path: Path, out_dir: Path, fk_fn, args):
    kp_pos, kp_se3, conf, meta = load_hdf5(hdf5_path)
    T = kp_pos.shape[0]
    print(f"\n  Task: {meta['task']}  ep {meta['episode']}  T={T}  conf={conf.mean():.3f}")

    print("  Retargeting (geometric) …", end=" ", flush=True)
    t0 = time.perf_counter()
    q_seq = retarget_geometric(kp_se3, smoothing_alpha=0.3)
    print(f"{(time.perf_counter()-t0)*1000:.0f}ms")

    print("  FK evaluation …", end=" ", flush=True)
    t0 = time.perf_counter()
    kp_robot = compute_fk_sequence(q_seq, fk_fn)
    print(f"{(time.perf_counter()-t0)*1000:.0f}ms")

    task = meta["task"].replace(" ", "_").replace("/", "_")
    ep   = meta["episode"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.static:
        frame = args.frame if args.frame is not None else T // 2
        plot_static(kp_pos, kp_robot, q_seq, frame, meta,
                    out_dir / f"{task}_ep{ep}_dex3_frame{frame}.png")
    else:
        make_animation(kp_pos, kp_robot, q_seq, meta,
                       out_dir / f"{task}_ep{ep}_dex3_retarget.mp4",
                       fps=args.fps, stride=args.stride)

    # Always save a mid-frame static too
    frame = T // 2
    plot_static(kp_pos, kp_robot, q_seq, frame, meta,
                out_dir / f"{task}_ep{ep}_dex3_frame{frame}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5",   type=Path, default=None)
    parser.add_argument("--out_dir",type=Path, default=Path("/tmp/dex3_retarget_vis"))
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--frame",  type=int, default=None)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found: {URDF_PATH}"); sys.exit(1)

    files = [args.hdf5] if args.hdf5 else sorted(
        (REPO_ROOT / "samples/egodex").rglob("*.hdf5")
    )
    print(f"Building FK function …")
    fk_fn = build_fk()
    print(f"Episodes: {len(files)}\n")

    for f in files:
        run_one(f, args.out_dir, fk_fn, args)

    print(f"\nAll outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()
