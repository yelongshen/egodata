"""
egodex_build_pretraining_dataset.py
-------------------------------------
Steps 3-5 of the EgoScale pipeline:
  3. Pack (image chunk, language, ΔW, q_hand) into training samples
  4. Optionally extract video frames
  5. Save as a HDF5 dataset ready for a flow-matching VLA trainer

EgoScale training sample (one action chunk):
  obs_image    : (1, H, W, 3)  uint8   egocentric RGB frame
  language     : str            natural language instruction
  delta_wrist  : (chunk, 12)   float32  [ΔW_left (6) | ΔW_right (6)]
  q_hand       : (chunk, 44)   float32  [q_left (22) | q_right (22)]
  valid        : (chunk,)      bool     confidence mask

Usage:
  python scripts/egodex_build_pretraining_dataset.py \
      --data_root /home/grease/ego_dataset/work_bearlu/egodex/test \
      --out       /tmp/egoscale_dataset.h5 \
      --chunk_size 16 \
      --stride 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Joint name lists (same as other scripts)
# ---------------------------------------------------------------------------
def _hand_joints(side: str) -> list[str]:
    s = side  # "left" or "right"
    S = side.capitalize()
    return [
        f"{s}Hand",
        f"{s}ThumbKnuckle", f"{s}ThumbIntermediateBase",
        f"{s}ThumbIntermediateTip", f"{s}ThumbTip",
        f"{s}IndexFingerMetacarpal",  f"{s}IndexFingerKnuckle",
        f"{s}IndexFingerIntermediateBase", f"{s}IndexFingerIntermediateTip",
        f"{s}IndexFingerTip",
        f"{s}MiddleFingerMetacarpal", f"{s}MiddleFingerKnuckle",
        f"{s}MiddleFingerIntermediateBase", f"{s}MiddleFingerIntermediateTip",
        f"{s}MiddleFingerTip",
        f"{s}RingFingerMetacarpal",  f"{s}RingFingerKnuckle",
        f"{s}RingFingerIntermediateBase", f"{s}RingFingerIntermediateTip",
        f"{s}RingFingerTip",
        f"{s}LittleFingerMetacarpal", f"{s}LittleFingerKnuckle",
        f"{s}LittleFingerIntermediateBase", f"{s}LittleFingerIntermediateTip",
        f"{s}LittleFingerTip",
    ]


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def se3_inv_batch(T: np.ndarray) -> np.ndarray:
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    Ti = np.zeros_like(T)
    Ti[:, :3, :3] = R.transpose(0, 2, 1)
    Ti[:, :3, 3:] = -(R.transpose(0, 2, 1) @ t)
    Ti[:, 3, 3] = 1.0
    return Ti


def se3_to_6d(T: np.ndarray) -> np.ndarray:
    """(N, 4, 4) → (N, 6) as [tx, ty, tz, rx, ry, rz] rotation-vector."""
    from scipy.spatial.transform import Rotation
    pos    = T[:, :3, 3]
    rotvec = Rotation.from_matrix(T[:, :3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec], axis=1).astype(np.float32)


def relative_wrist_6d(W: np.ndarray) -> np.ndarray:
    """(T, 4, 4) wrist poses → (T-1, 6) relative deltas."""
    delta = se3_inv_batch(W[:-1]) @ W[1:]
    return se3_to_6d(delta)


# ---------------------------------------------------------------------------
# PCA hand approximation (replace with CasADi retargeting for production)
# ---------------------------------------------------------------------------

def pca_retarget(kp_seq: np.ndarray, n_joints: int = 22) -> np.ndarray:
    """(T, 25, 3) → (T, n_joints) in [-1, 1]. Fallback without URDF."""
    T = kp_seq.shape[0]
    wrist   = kp_seq[:, 0:1, :]
    rel     = (kp_seq - wrist).reshape(T, -1).astype(np.float32)
    mean    = rel.mean(0)
    centered = rel - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj    = centered @ Vt[:n_joints].T
    lo, hi  = proj.min(0), proj.max(0)
    denom   = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    return (2.0 * (proj - lo) / denom - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Video frame extractor
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frame_indices: list[int],
                   size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Extract specific frames from an MP4 and resize.

    Returns:
        frames: (N, H, W, 3) uint8 RGB
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    prev_idx = -1
    for idx in sorted(set(frame_indices)):
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Pad with zeros if frame missing
            frame = np.zeros((*size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
        frames.append(frame)
        prev_idx = idx
    cap.release()
    # Reorder to match original frame_indices order
    idx_order = np.argsort(np.argsort(frame_indices))
    return np.array(frames)[idx_order]   # (N, H, W, 3)


# ---------------------------------------------------------------------------
# Episode → training chunks
# ---------------------------------------------------------------------------

def episode_to_chunks(
    hdf5_path: Path,
    chunk_size: int = 16,
    stride: int = 1,
    image_size: tuple[int, int] = (224, 224),
    conf_thresh: float = 0.5,
    n_joints: int = 22,
) -> list[dict]:
    """Convert one EgoDex episode into a list of training chunks.

    Each chunk is a dict:
      obs_image    : (1, H, W, 3)   uint8  — observation frame
      language     : str
      task         : str
      delta_wrist  : (chunk_size, 12) float32  [ΔW_L(6) | ΔW_R(6)]
      q_hand       : (chunk_size, 44) float32  [q_L(22) | q_R(22)]
      valid        : (chunk_size,)    bool

    Args:
        hdf5_path:  path to one episode HDF5
        chunk_size: number of future timesteps per training sample
        stride:     step between consecutive chunk start frames
        image_size: (H, W) to resize frames to
        conf_thresh: confidence threshold for valid mask
        n_joints:   robot hand DoF
    """
    with h5py.File(hdf5_path, "r") as f:
        T = f["transforms/camera"].shape[0]

        # Wrist poses in world frame
        W_left  = f["transforms/leftHand"][:]    # (T, 4, 4)
        W_right = f["transforms/rightHand"][:]   # (T, 4, 4)

        # Hand keypoints (positions only)
        kp_left  = np.stack([f[f"transforms/{j}"][:, :3, 3]
                              for j in _hand_joints("left")],  axis=1)  # (T,25,3)
        kp_right = np.stack([f[f"transforms/{j}"][:, :3, 3]
                              for j in _hand_joints("right")], axis=1)  # (T,25,3)

        # Confidence
        conf_l = f["confidences/leftHand"][:] if "confidences/leftHand" in f \
                 else np.ones(T, dtype=np.float32)
        conf_r = f["confidences/rightHand"][:] if "confidences/rightHand" in f \
                 else np.ones(T, dtype=np.float32)

        language = str(f.attrs.get("llm_description") or f.attrs.get("description", ""))
        task     = str(f.attrs.get("task", ""))

    video_path = hdf5_path.with_suffix(".mp4")

    # Compute actions (T-1 deltas)
    dw_left  = relative_wrist_6d(W_left)   # (T-1, 6)
    dw_right = relative_wrist_6d(W_right)  # (T-1, 6)
    delta_wrist = np.concatenate([dw_left, dw_right], axis=1)  # (T-1, 12)

    # Retarget (PCA approximation — swap in CasADi version for production)
    q_left  = pca_retarget(kp_left,  n_joints)   # (T, n_joints)
    q_right = pca_retarget(kp_right, n_joints)   # (T, n_joints)
    q_hand  = np.concatenate([q_left, q_right], axis=1)  # (T, 2*n_joints)

    # Validity mask
    valid = (conf_l >= conf_thresh) & (conf_r >= conf_thresh)  # (T,)

    # Slice into chunks: obs at t=start, actions for t=start..start+chunk_size
    # (T-1 deltas limits max start to T-1-chunk_size)
    max_start = T - 1 - chunk_size
    if max_start <= 0:
        return []

    # Extract all needed observation frames at once
    obs_frame_indices = list(range(0, max_start + 1, stride))
    if video_path.exists():
        obs_frames = extract_frames(video_path, obs_frame_indices, image_size)
    else:
        obs_frames = np.zeros((len(obs_frame_indices), *image_size, 3), dtype=np.uint8)

    chunks = []
    for i, start in enumerate(obs_frame_indices):
        end = start + chunk_size
        chunk = {
            "obs_image":   obs_frames[i][np.newaxis],            # (1, H, W, 3)
            "language":    language,
            "task":        task,
            "delta_wrist": delta_wrist[start:end],               # (chunk, 12)
            "q_hand":      q_hand[start:end],                    # (chunk, 2*n_joints)
            "valid":       valid[start:end],                      # (chunk,)
            "start_frame": np.int32(start),
            "episode_id":  np.int32(int(hdf5_path.stem)),
        }
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Build full dataset
# ---------------------------------------------------------------------------

def build_dataset(
    data_root: Path,
    out_path: Path,
    chunk_size: int = 16,
    stride: int = 1,
    image_size: tuple[int, int] = (224, 224),
    conf_thresh: float = 0.5,
    n_joints: int = 22,
    max_episodes: int | None = None,
) -> None:
    """Process all episodes and write a flat HDF5 training dataset.

    Output HDF5 layout:
      /obs_image    (N, 1, H, W, 3)  uint8
      /delta_wrist  (N, chunk, 12)   float32
      /q_hand       (N, chunk, 44)   float32
      /valid        (N, chunk)       bool
      /language     (N,)             variable-length string
      /task         (N,)             variable-length string
    """
    # Collect all HDF5 paths
    all_eps = sorted(data_root.rglob("*.hdf5"),
                     key=lambda p: (p.parent.name, int(p.stem)))
    if max_episodes:
        all_eps = all_eps[:max_episodes]

    print(f"Found {len(all_eps)} episodes under {data_root}")

    # First pass: count total chunks
    all_chunks: list[dict] = []
    for ep in tqdm(all_eps, desc="Processing episodes"):
        chunks = episode_to_chunks(
            ep, chunk_size=chunk_size, stride=stride,
            image_size=image_size, conf_thresh=conf_thresh,
            n_joints=n_joints,
        )
        all_chunks.extend(chunks)

    N = len(all_chunks)
    H, W = image_size
    print(f"Total training chunks: {N}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as out:
        # Pre-allocate datasets
        out.create_dataset("obs_image",   shape=(N, 1, H, W, 3),
                           dtype=np.uint8,    chunks=(64, 1, H, W, 3),
                           compression="lzf")
        out.create_dataset("delta_wrist", shape=(N, chunk_size, 12),
                           dtype=np.float32,  chunks=(256, chunk_size, 12))
        out.create_dataset("q_hand",      shape=(N, chunk_size, 2 * n_joints),
                           dtype=np.float32,  chunks=(256, chunk_size, 2 * n_joints))
        out.create_dataset("valid",       shape=(N, chunk_size),
                           dtype=bool,        chunks=(256, chunk_size))

        # Variable-length strings
        str_dtype = h5py.string_dtype()
        out.create_dataset("language", shape=(N,), dtype=str_dtype)
        out.create_dataset("task",     shape=(N,), dtype=str_dtype)
        out.create_dataset("start_frame", shape=(N,), dtype=np.int32)
        out.create_dataset("episode_id",  shape=(N,), dtype=np.int32)

        # Fill
        for i, chunk in enumerate(tqdm(all_chunks, desc="Writing HDF5")):
            out["obs_image"][i]    = chunk["obs_image"]
            out["delta_wrist"][i]  = chunk["delta_wrist"]
            out["q_hand"][i]       = chunk["q_hand"]
            out["valid"][i]        = chunk["valid"]
            out["language"][i]     = chunk["language"]
            out["task"][i]         = chunk["task"]
            out["start_frame"][i]  = chunk["start_frame"]
            out["episode_id"][i]   = chunk["episode_id"]

        # Metadata
        out.attrs["chunk_size"]   = chunk_size
        out.attrs["n_joints"]     = n_joints
        out.attrs["image_height"] = H
        out.attrs["image_width"]  = W
        out.attrs["n_chunks"]     = N
        out.attrs["note"] = (
            "q_hand uses PCA approximation. "
            "Replace with CasADi retargeting (egodex_retarget.py) for "
            "paper-faithful EgoScale training."
        )

    size_gb = os.path.getsize(out_path) / 1024**3
    print(f"\nDataset written → {out_path}  ({size_gb:.2f} GB)")
    print(f"  Chunks:      {N}")
    print(f"  Chunk size:  {chunk_size} steps")
    print(f"  Image size:  {H}×{W}")
    print(f"  Action dim:  {12} (ΔW) + {2*n_joints} (q_hand) = {12 + 2*n_joints}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EgoScale-style pretraining dataset from EgoDex."
    )
    parser.add_argument("--data_root", type=Path,
                        default=Path("/home/grease/ego_dataset/work_bearlu/egodex/test"),
                        help="Root of extracted EgoDex split")
    parser.add_argument("--out",       type=Path,
                        default=Path("/tmp/egoscale_pretraining.h5"))
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Action chunk length (default: 16 steps)")
    parser.add_argument("--stride",     type=int, default=4,
                        help="Step between chunk start frames (default: 4)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Square image crop size (default: 224)")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--n_joints",    type=int,   default=22,
                        help="Robot hand DoF for PCA approximation (default: 22)")
    parser.add_argument("--max_episodes", type=int,  default=None,
                        help="Cap number of episodes (useful for dry runs)")
    args = parser.parse_args()

    build_dataset(
        data_root=args.data_root,
        out_path=args.out,
        chunk_size=args.chunk_size,
        stride=args.stride,
        image_size=(args.image_size, args.image_size),
        conf_thresh=args.conf_thresh,
        n_joints=args.n_joints,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
