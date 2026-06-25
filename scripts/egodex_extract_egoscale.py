"""
egodex_extract_egoscale.py
--------------------------
Extract EgoScale-style pretraining signals from EgoDex HDF5 episodes.

EgoScale pretraining (Section 2.1-2.2 of arXiv:2602.16710        conf_left  = (f["confidences/leftHand"][:]
                      if "confidences/leftHand" in f
                      else np.ones(T, dtype=np.float32))
        conf_right = (f["confidences/rightHand"][:]
                      if "confidences/rightHand" in f
                      else np.ones(T, dtype=np.float32))requires per-frame:
  1. Relative wrist motion  ΔW_t  ∈ SE(3)   — arm-level action signal
  2. Hand keypoints         H_t   ∈ SE(3)^K  — finger-level action signal (pre-retargeting)
  3. Camera intrinsic       K                 — for any pixel-space computations
  4. Confidence mask                          — filter unreliable tracking frames
  5. Language instruction                     — text conditioning

Usage:
  python scripts/egodex_extract_egoscale.py \
      --data_root /home/grease/ego_dataset/work_bearlu/egodex/test \
      --task basic_pick_place \
      --episode 0 \
      --out /tmp/egoscale_sample.npz

  # Process all episodes of all tasks and write one .npz per episode:
  python scripts/egodex_extract_egoscale.py \
      --data_root /home/grease/ego_dataset/work_bearlu/egodex/test \
      --out_dir /tmp/egoscale_test \
      --conf_thresh 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Hand keypoint ordering (25 joints per hand as stored in EgoDex HDF5)
# The first entry is the wrist (index 0), matching EgoScale's H_t^{c,1}.
# ---------------------------------------------------------------------------
LEFT_HAND_JOINTS: list[str] = [
    # wrist
    "leftHand",
    # thumb  (4 joints, no metacarpal)
    "leftThumbKnuckle", "leftThumbIntermediateBase",
    "leftThumbIntermediateTip", "leftThumbTip",
    # index  (5 joints including metacarpal)
    "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase", "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    # middle
    "leftMiddleFingerMetacarpal", "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase", "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    # ring
    "leftRingFingerMetacarpal", "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase", "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    # little
    "leftLittleFingerMetacarpal", "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase", "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]

RIGHT_HAND_JOINTS: list[str] = [j.replace("left", "right").replace("Left", "Right")
                                  for j in LEFT_HAND_JOINTS]


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert a (4, 4) SE(3) matrix.  T^{-1} = [R^T | -R^T t ; 0 | 1]"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -(R.T @ t)
    return T_inv


def se3_inv_batch(T: np.ndarray) -> np.ndarray:
    """Batch version: (N, 4, 4) → (N, 4, 4)."""
    R = T[:, :3, :3]                      # (N, 3, 3)
    t = T[:, :3, 3:]                      # (N, 3, 1)
    T_inv = np.zeros_like(T)
    T_inv[:, :3, :3] = R.transpose(0, 2, 1)
    T_inv[:, :3, 3:] = -(R.transpose(0, 2, 1) @ t)
    T_inv[:, 3, 3]   = 1.0
    return T_inv


def relative_wrist_motion(W: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame relative wrist deltas.

    Args:
        W: (T, 4, 4) wrist poses in world frame W_t^w

    Returns:
        delta_W: (T-1, 4, 4) where delta_W[t] = W_t^{-1} @ W_{t+1}
                 Camera-motion-invariant relative arm displacement.
    """
    W_prev_inv = se3_inv_batch(W[:-1])   # (T-1, 4, 4)
    delta_W    = W_prev_inv @ W[1:]      # (T-1, 4, 4)  matrix multiply
    return delta_W


def se3_to_pos_rotvec(T: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) SE(3) → (N, 6) [tx, ty, tz, rx, ry, rz] using
    the rotation-vector (axis-angle) representation for the rotation part.
    This matches the compact 6-DoF action vector used in VLA action chunks.
    """
    from scipy.spatial.transform import Rotation
    N = T.shape[0]
    pos    = T[:, :3, 3]                                      # (N, 3)
    rotvec = Rotation.from_matrix(T[:, :3, :3]).as_rotvec()  # (N, 3)
    return np.concatenate([pos, rotvec], axis=1)              # (N, 6)


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class EgoScaleSample(NamedTuple):
    """One training episode in EgoScale pretraining format."""
    # ── arm-level action (primary supervision signal) ──────────────────────
    delta_wrist_left:  np.ndarray   # (T-1, 4, 4) relative SE(3) wrist deltas
    delta_wrist_right: np.ndarray   # (T-1, 4, 4)
    delta_wrist_left_6d:  np.ndarray  # (T-1, 6)  compact [pos | rotvec]
    delta_wrist_right_6d: np.ndarray  # (T-1, 6)

    # ── finger-level signal (input to retargeting pipeline) ─────────────────
    hand_keypoints_left:  np.ndarray  # (T, 25, 4, 4) world-frame SE(3)
    hand_keypoints_right: np.ndarray  # (T, 25, 4, 4)

    # ── camera ──────────────────────────────────────────────────────────────
    camera_poses:    np.ndarray   # (T, 4, 4)  T_t^{w←c} world←camera
    camera_intrinsic: np.ndarray  # (3, 3)

    # ── quality / masking ───────────────────────────────────────────────────
    conf_left:   np.ndarray   # (T,)  wrist-level tracking confidence
    conf_right:  np.ndarray   # (T,)
    valid_mask:  np.ndarray   # (T,)  bool, conf_left & conf_right ≥ threshold

    # ── language conditioning ────────────────────────────────────────────────
    language:     str
    task:         str
    episode_id:   int
    video_path:   str         # absolute path to the paired .mp4


def extract_episode(
    hdf5_path: Path,
    conf_thresh: float = 0.5,
) -> EgoScaleSample:
    """Extract one EgoDex episode into EgoScale pretraining format.

    Args:
        hdf5_path:   path to the .hdf5 file
        conf_thresh: confidence threshold below which frames are masked out

    Returns:
        EgoScaleSample
    """
    with h5py.File(hdf5_path, "r") as f:
        T = f["transforms/camera"].shape[0]

        # ── camera ──────────────────────────────────────────────────────────
        camera_poses     = f["transforms/camera"][:]         # (T, 4, 4)
        camera_intrinsic = f["camera/intrinsic"][:]          # (3, 3)

        # ── wrist poses in world frame ───────────────────────────────────────
        #  W_t^w = transforms/leftHand[t]  (Apple Vision Pro outputs world-frame)
        wrist_left_world  = f["transforms/leftHand"][:]      # (T, 4, 4)
        wrist_right_world = f["transforms/rightHand"][:]     # (T, 4, 4)

        # ── hand keypoints (all 25 joints per hand) ──────────────────────────
        kp_left  = np.stack([f[f"transforms/{j}"][:] for j in LEFT_HAND_JOINTS],
                            axis=1)   # (T, 25, 4, 4)
        kp_right = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_HAND_JOINTS],
                            axis=1)   # (T, 25, 4, 4)

        # ── confidence ───────────────────────────────────────────────────────
        conf_left  = (f["confidences/leftHand"][:]
                      if "confidences/leftHand" in f
                      else np.ones(T, dtype=np.float32))
        conf_right = (f["confidences/rightHand"][:]
                      if "confidences/rightHand" in f
                      else np.ones(T, dtype=np.float32))

        # ── metadata ─────────────────────────────────────────────────────────
        language   = str(f.attrs.get("llm_description") or f.attrs.get("description", ""))
        task       = str(f.attrs.get("task", ""))

    # ── compute camera-invariant relative wrist deltas ────────────────────────
    #  ΔW_t = (W_{t-1}^w)^{-1} · W_t^w   (Section 2.1, arXiv:2602.16710)
    delta_left  = relative_wrist_motion(wrist_left_world)    # (T-1, 4, 4)
    delta_right = relative_wrist_motion(wrist_right_world)   # (T-1, 4, 4)

    # ── compact 6-DoF representation ─────────────────────────────────────────
    delta_left_6d  = se3_to_pos_rotvec(delta_left)           # (T-1, 6)
    delta_right_6d = se3_to_pos_rotvec(delta_right)          # (T-1, 6)

    # ── per-frame validity mask ───────────────────────────────────────────────
    valid_mask = (conf_left >= conf_thresh) & (conf_right >= conf_thresh)  # (T,)

    # ── paired video path ─────────────────────────────────────────────────────
    video_path = str(hdf5_path.with_suffix(".mp4"))
    episode_id = int(hdf5_path.stem)

    return EgoScaleSample(
        delta_wrist_left=delta_left,
        delta_wrist_right=delta_right,
        delta_wrist_left_6d=delta_left_6d,
        delta_wrist_right_6d=delta_right_6d,
        hand_keypoints_left=kp_left,
        hand_keypoints_right=kp_right,
        camera_poses=camera_poses,
        camera_intrinsic=camera_intrinsic,
        conf_left=conf_left,
        conf_right=conf_right,
        valid_mask=valid_mask,
        language=language,
        task=task,
        episode_id=episode_id,
        video_path=video_path,
    )


def sample_to_dict(s: EgoScaleSample) -> dict:
    """Flatten EgoScaleSample to a saveable dict (for np.savez / inspection)."""
    return {
        "delta_wrist_left":     s.delta_wrist_left,
        "delta_wrist_right":    s.delta_wrist_right,
        "delta_wrist_left_6d":  s.delta_wrist_left_6d,
        "delta_wrist_right_6d": s.delta_wrist_right_6d,
        "hand_keypoints_left":  s.hand_keypoints_left,
        "hand_keypoints_right": s.hand_keypoints_right,
        "camera_poses":         s.camera_poses,
        "camera_intrinsic":     s.camera_intrinsic,
        "conf_left":            s.conf_left,
        "conf_right":           s.conf_right,
        "valid_mask":           s.valid_mask,
        "language":             np.bytes_(s.language),
        "task":                 np.bytes_(s.task),
        "episode_id":           np.int32(s.episode_id),
        "video_path":           np.bytes_(s.video_path),
    }


# ---------------------------------------------------------------------------
# Print a human-readable summary of one sample
# ---------------------------------------------------------------------------

def print_summary(s: EgoScaleSample) -> None:
    T = s.camera_poses.shape[0]
    valid_frames = int(s.valid_mask.sum())
    print(f"\n{'='*60}")
    print(f"Task:          {s.task}")
    print(f"Episode:       {s.episode_id}")
    print(f"Language:      {s.language}")
    print(f"Video:         {s.video_path}")
    print(f"Frames (T):    {T}  ({T/30:.1f}s at 30 FPS)")
    print(f"Valid frames:  {valid_frames}/{T}  ({100*valid_frames/T:.1f}%)")
    print()
    print("── Arm-level actions (EgoScale ΔW) ──────────────────────")
    print(f"  delta_wrist_left:     {s.delta_wrist_left.shape}  SE(3) 4×4 per step")
    print(f"  delta_wrist_left_6d:  {s.delta_wrist_left_6d.shape}  [tx ty tz rx ry rz]")
    print(f"  delta_wrist_right:    {s.delta_wrist_right.shape}")
    print()
    # Show magnitude of motion
    left_trans  = np.linalg.norm(s.delta_wrist_left_6d[:, :3],  axis=1)
    left_rot    = np.linalg.norm(s.delta_wrist_left_6d[:, 3:],  axis=1)
    right_trans = np.linalg.norm(s.delta_wrist_right_6d[:, :3], axis=1)
    right_rot   = np.linalg.norm(s.delta_wrist_right_6d[:, 3:], axis=1)
    print(f"  Left  translation Δ: mean={left_trans.mean():.4f}m  max={left_trans.max():.4f}m")
    print(f"  Left  rotation    Δ: mean={np.degrees(left_rot.mean()):.2f}°  max={np.degrees(left_rot.max()):.2f}°")
    print(f"  Right translation Δ: mean={right_trans.mean():.4f}m  max={right_trans.max():.4f}m")
    print(f"  Right rotation    Δ: mean={np.degrees(right_rot.mean()):.2f}°  max={np.degrees(right_rot.max()):.2f}°")
    print()
    print("── Finger-level keypoints ────────────────────────────────")
    print(f"  hand_keypoints_left:  {s.hand_keypoints_left.shape}  (T, 25 joints, 4×4)")
    print(f"  Joint[0] = wrist, matches ΔW input")
    print(f"  Joints[1:5]  = thumb (4),  [5:10]  = index (5)")
    print(f"  Joints[10:15]= middle (5), [15:20] = ring (5), [20:25]= little (5)")
    print()
    print("── Confidence ────────────────────────────────────────────")
    print(f"  Left  wrist conf: min={s.conf_left.min():.3f}  mean={s.conf_left.mean():.3f}  max={s.conf_left.max():.3f}")
    print(f"  Right wrist conf: min={s.conf_right.min():.3f}  mean={s.conf_right.mean():.3f}  max={s.conf_right.max():.3f}")
    print()
    print("── Camera ────────────────────────────────────────────────")
    print(f"  camera_poses:    {s.camera_poses.shape}")
    print(f"  intrinsic K:\n{s.camera_intrinsic}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract EgoScale-style pretraining signals from EgoDex HDF5 data."
    )
    parser.add_argument("--data_root", type=Path,
                        default=Path("/home/grease/ego_dataset/work_bearlu/egodex/test"),
                        help="Root of extracted EgoDex split (contains <task>/<id>.hdf5)")
    parser.add_argument("--task",    type=str, default=None,
                        help="Single task to process (default: all tasks)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Single episode id to process (default: all)")
    parser.add_argument("--out",     type=Path, default=None,
                        help="Save a single .npz (use with --task + --episode)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory for batch processing (.npz per episode)")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Wrist confidence threshold for valid-frame mask (default 0.5)")
    parser.add_argument("--summary_only", action="store_true",
                        help="Print summary statistics without saving")
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # Collect episodes to process
    if args.task:
        task_dirs = [data_root / args.task]
    else:
        task_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    episodes: list[Path] = []
    for task_dir in task_dirs:
        if args.episode is not None:
            ep = task_dir / f"{args.episode}.hdf5"
            if ep.exists():
                episodes.append(ep)
        else:
            episodes.extend(sorted(task_dir.glob("*.hdf5"),
                                   key=lambda p: int(p.stem)))

    print(f"Processing {len(episodes)} episode(s) from {data_root}")

    for i, hdf5_path in enumerate(episodes):
        sample = extract_episode(hdf5_path, conf_thresh=args.conf_thresh)

        print_summary(sample)

        if args.summary_only:
            continue

        if args.out and len(episodes) == 1:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out, **sample_to_dict(sample))
            print(f"Saved → {args.out}")

        elif args.out_dir:
            out_path = args.out_dir / sample.task / f"{sample.episode_id:06d}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **sample_to_dict(sample))
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[{i+1}/{len(episodes)}] Saved → {out_path}")


if __name__ == "__main__":
    main()
