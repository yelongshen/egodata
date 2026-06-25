"""
egodex_retarget.py
------------------
Step 2 of the EgoScale pipeline: retarget EgoDex 25-joint hand keypoints
into the Unitree Dex5-1 (20-DoF) joint space.

URDF source: unitreerobotics/unitree_ros  →  assets/dex5_1/
  Left  hand:  assets/dex5_1/Dex5-URDF-L/Dex5-URDF-L.urdf
  Right hand:  assets/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf

EgoScale (arXiv:2602.16710, Section 2.1) uses CasADi + IPOPT to solve a
per-frame nonlinear program:

    minimize   Σ_i w_i * || FK(q)[i] - H_t[i] ||²
    subject to q_lo ≤ q ≤ q_hi   (joint limits from URDF)

where:
  q        : target robot joint angles (22-DoF for Sharpa hand)
  FK(q)[i] : forward kinematics position of keypoint i
  H_t[i]   : observed human keypoint i in world/camera frame
  w_i      : per-joint weight (wrist and fingertips weighted higher)

This file provides:
  1. RetargetConfig     - robot hand spec (URDF path, joint names, weights)
  2. retarget_sequence  - retarget a full episode (T, 25, 3) → (T, N_joints)
  3. A fallback PCA approximation when CasADi is unavailable

Requirements for full retargeting:
  pip install casadi    (optimization solver)
  pip install yourdfpy  (URDF forward kinematics)
  + your robot hand URDF file

For EgoScale replication:  use Sharpa Wave 22-DoF hand URDF
For custom robots:         substitute any dexterous hand URDF
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import math as _math

# ---------------------------------------------------------------------------
# Dex5-1 joint specification (extracted from official Unitree URDF)
# Joint naming: {Type}_{Finger}{Joint}{Side}
#   Finger 1 = Thumb, 2 = Index, 3 = Middle, 4 = Ring, 5 = Little
#   Type: Yaw = abduction/adduction, Roll = MCP, Pitch = flexion
# ---------------------------------------------------------------------------

DEX5_1_JOINTS_R: list[str] = [
    # Thumb (4 DoF)
    "Yaw_11R",   # thumb abduction/adduction  [-33.6° .. +39.0°]
    "Roll_12R",  # thumb MCP roll             [  0.0° .. 104.0°]
    "Pitch_13R", # thumb IP1 flexion          [  0.0° .. 101.1°]
    "Pitch_14R", # thumb IP2 flexion (coupled)[  0.0° ..  94.0°]
    # Index (4 DoF)
    "Roll_21R",  # lateral swing              [-22.0° ..  22.0°]
    "Pitch_22R", # MCP flexion                [  0.0° ..  90.0°]
    "Pitch_23R", # PIP flexion                [  0.0° ..  96.5°]
    "Pitch_24R", # DIP flexion (coupled)      [  0.0° ..  80.0°]
    # Middle (4 DoF)
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    # Ring (4 DoF)
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    # Little (4 DoF)
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
]

DEX5_1_JOINTS_L: list[str] = [j.replace("R", "L") if j.endswith("R") else j
                                for j in DEX5_1_JOINTS_R]
# Fix the ring finger left-hand quirk in the URDF (Roll_41L is named Link_41L)
DEX5_1_JOINTS_L[12] = "Link_41L"

def _deg(d: float) -> float: return _math.radians(d)

# (lower, upper) in radians for each joint — same order as DEX5_1_JOINTS_R
DEX5_1_LIMITS: list[tuple[float, float]] = [
    (_deg(-33.6), _deg(39.0)),   # Yaw_11   thumb abd
    (_deg(0.0),   _deg(104.0)),  # Roll_12  thumb MCP
    (_deg(0.0),   _deg(101.1)),  # Pitch_13 thumb IP1
    (_deg(0.0),   _deg(94.0)),   # Pitch_14 thumb IP2
    (_deg(-22.0), _deg(22.0)),   # Roll_21  index lat
    (_deg(0.0),   _deg(90.0)),   # Pitch_22 index MCP
    (_deg(0.0),   _deg(96.5)),   # Pitch_23 index PIP
    (_deg(0.0),   _deg(80.0)),   # Pitch_24 index DIP
    (_deg(-22.0), _deg(22.0)),   # Roll_31  middle lat
    (_deg(0.0),   _deg(90.0)),   # Pitch_32 middle MCP
    (_deg(0.0),   _deg(96.5)),   # Pitch_33 middle PIP
    (_deg(0.0),   _deg(80.0)),   # Pitch_34 middle DIP
    (_deg(-22.0), _deg(22.0)),   # Roll_41  ring lat
    (_deg(0.0),   _deg(90.0)),   # Pitch_42 ring MCP
    (_deg(0.0),   _deg(96.5)),   # Pitch_43 ring PIP
    (_deg(0.0),   _deg(80.0)),   # Pitch_44 ring DIP
    (_deg(-22.0), _deg(22.0)),   # Roll_51  little lat
    (_deg(0.0),   _deg(90.0)),   # Pitch_52 little MCP
    (_deg(0.0),   _deg(96.5)),   # Pitch_53 little PIP
    (_deg(0.0),   _deg(80.0)),   # Pitch_54 little DIP
]

# EgoDex keypoint index → Dex5-1 URDF link name (right hand)
# EgoDex layout: 0=wrist, 1-4=thumb, 5-9=index, 10-14=middle,
#                15-19=ring, 20-24=little
DEX5_1_KEYPOINT_LINK_MAP_R: dict[int, str] = {
    0:  "base_link00",  # wrist
    1:  "Link_11R",     # thumb knuckle    (after Yaw_11R)
    2:  "Link_12R",     # thumb interBase  (after Roll_12R)
    3:  "Link_13R",     # thumb interTip   (after Pitch_13R)
    4:  "Link_14R",     # thumb tip        (after Pitch_14R)
    6:  "Link_21R",     # index knuckle    (after Roll_21R)
    7:  "Link_22R",     # index interBase  (after Pitch_22R)
    8:  "Link_23R",     # index interTip   (after Pitch_23R)
    9:  "Link_24R",     # index tip        (after Pitch_24R)
    11: "Link_31R",     # middle knuckle
    12: "Link_32R",     # middle interBase
    13: "Link_33R",     # middle interTip
    14: "Link_34R",     # middle tip
    16: "Link_41R",     # ring knuckle
    17: "Link_42R",     # ring interBase
    18: "Link_43R",     # ring interTip
    19: "Link_44R",     # ring tip
    21: "Link_51R",     # little knuckle
    22: "Link_52R",     # little interBase
    23: "Link_53R",     # little interTip
    24: "Link_54R",     # little tip
}
# Metacarpal keypoints (5, 10, 15, 20) are not directly represented
# in the Dex5-1 — they fall on the palm and are skipped.

DEX5_1_KEYPOINT_WEIGHTS_R: dict[int, float] = {
    0: 2.0,   # wrist
    4: 2.0, 9: 2.0, 14: 1.5, 19: 1.5, 24: 1.5,  # fingertips
    1: 1.0, 2: 1.0, 3: 1.0,                       # thumb chain
    6: 0.8, 7: 0.8, 8: 0.8,                        # index chain
    11: 0.6, 12: 0.6, 13: 0.6,                     # middle chain
    16: 0.6, 17: 0.6, 18: 0.6,                     # ring chain
    21: 0.5, 22: 0.5, 23: 0.5,                     # little chain
}

_REPO_ROOT = Path(__file__).parent.parent

@dataclass
class RetargetConfig:
    """Describes the target robot hand for retargeting."""
    urdf_path:    Path
    n_joints:     int
    joint_names:  list[str]
    joint_lo:     np.ndarray             # (n_joints,) lower limits in radians
    joint_hi:     np.ndarray             # (n_joints,) upper limits in radians
    keypoint_link_map: dict[int, str]    # {egodex_kp_idx: urdf_link_name}
    keypoint_weights:  dict[int, float]
    smoothing_alpha: float = 0.3

    @classmethod
    def dex5_1_right(cls, urdf_path: Path | None = None) -> "RetargetConfig":
        """Unitree Dex5-1 right hand — 20 DoF (default)."""
        if urdf_path is None:
            urdf_path = _REPO_ROOT / "assets/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf"
        lo = np.array([l for l, _ in DEX5_1_LIMITS], dtype=np.float32)
        hi = np.array([h for _, h in DEX5_1_LIMITS], dtype=np.float32)
        return cls(
            urdf_path=urdf_path, n_joints=20,
            joint_names=DEX5_1_JOINTS_R,
            joint_lo=lo, joint_hi=hi,
            keypoint_link_map=DEX5_1_KEYPOINT_LINK_MAP_R,
            keypoint_weights=DEX5_1_KEYPOINT_WEIGHTS_R,
        )

    @classmethod
    def dex5_1_left(cls, urdf_path: Path | None = None) -> "RetargetConfig":
        """Unitree Dex5-1 left hand — 20 DoF."""
        if urdf_path is None:
            urdf_path = _REPO_ROOT / "assets/dex5_1/Dex5-URDF-L/Dex5-URDF-L.urdf"
        lo = np.array([l for l, _ in DEX5_1_LIMITS], dtype=np.float32)
        hi = np.array([h for _, h in DEX5_1_LIMITS], dtype=np.float32)
        # Mirror: Roll_12L lower/upper are negated vs right
        lo[1], hi[1] = _deg(-104.0), _deg(0.0)
        link_map = {k: v.replace("R", "L") for k, v in DEX5_1_KEYPOINT_LINK_MAP_R.items()}
        link_map[0] = "base_link00L"
        return cls(
            urdf_path=urdf_path, n_joints=20,
            joint_names=DEX5_1_JOINTS_L,
            joint_lo=lo, joint_hi=hi,
            keypoint_link_map=link_map,
            keypoint_weights=DEX5_1_KEYPOINT_WEIGHTS_R,
        )


# ---------------------------------------------------------------------------
# CasADi-based retargeting (full, paper-faithful)
# ---------------------------------------------------------------------------

def _retarget_frame_casadi(
    keypoints_3d: np.ndarray,     # (25, 3) target positions
    config: RetargetConfig,
    fk_fn,                        # callable: q (n,) → dict{link: (3,)}
    q_init: np.ndarray,           # (n_joints,) warm-start
) -> np.ndarray:
    """Solve one-frame retargeting via CasADi + IPOPT.

    Args:
        keypoints_3d: target human keypoint positions (25, 3)
        config:       robot hand config
        fk_fn:        forward kinematics function (differentiable via CasADi)
        q_init:       previous frame solution for warm-starting

    Returns:
        q_opt: (n_joints,) optimal joint angles
    """
    try:
        import casadi as ca
    except ImportError:
        raise ImportError(
            "casadi is required for full retargeting. "
            "Install with: pip install casadi"
        )

    n = config.n_joints
    q = ca.MX.sym("q", n)

    # Build objective: weighted sum of squared distances to target keypoints
    obj = 0.0
    for kp_idx, link_name in config.keypoint_link_map.items():
        w = config.keypoint_weights.get(kp_idx, 1.0)
        target = keypoints_3d[kp_idx]              # (3,)
        fk_pos = fk_fn(q, link_name)               # (3,) as CasADi expression
        diff   = fk_pos - ca.DM(target)
        obj   += w * ca.dot(diff, diff)

    # Joint limit constraints
    nlp = {"x": q, "f": obj}
    opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter":    50,
        "print_time":        False,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    result = solver(
        x0=q_init,
        lbx=config.joint_lo,
        ubx=config.joint_hi,
    )
    return np.array(result["x"]).flatten()


# ---------------------------------------------------------------------------
# PCA-based approximation (fallback without URDF/CasADi)
# ---------------------------------------------------------------------------

def retarget_pca_approx(
    keypoints_seq: np.ndarray,    # (T, 25, 3) human keypoints in world frame
    n_joints: int = 22,
) -> np.ndarray:
    """Approximate retargeting via PCA — no URDF needed.

    This is NOT paper-faithful but useful for prototyping:
    - Center each hand around its wrist
    - Flatten to (T, 75) relative positions
    - Project to n_joints dimensions via PCA
    - Scale to [-1, 1] as a proxy for joint angles

    Args:
        keypoints_seq: (T, 25, 3) keypoint positions
        n_joints:      target DoF

    Returns:
        q_approx: (T, n_joints) pseudo joint angles in [-1, 1]
    """
    T = keypoints_seq.shape[0]
    wrist = keypoints_seq[:, 0:1, :]              # (T, 1, 3) wrist position
    relative = keypoints_seq - wrist              # (T, 25, 3) wrist-relative
    flat = relative.reshape(T, -1).astype(np.float32)   # (T, 75)

    # PCA
    mean = flat.mean(axis=0)
    centered = flat - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_joints]                    # (n_joints, 75)
    projected  = centered @ components.T          # (T, n_joints)

    # Normalize to [-1, 1]
    p_min = projected.min(axis=0, keepdims=True)
    p_max = projected.max(axis=0, keepdims=True)
    denom = np.where(p_max - p_min > 1e-6, p_max - p_min, 1.0)
    q_approx = 2.0 * (projected - p_min) / denom - 1.0

    return q_approx   # (T, n_joints)


# ---------------------------------------------------------------------------
# Main sequence retargeter
# ---------------------------------------------------------------------------

def retarget_sequence(
    keypoints_seq: np.ndarray,    # (T, 25, 3) human keypoints in world frame
    config: Optional[RetargetConfig] = None,
    fk_fn=None,
) -> np.ndarray:
    """Retarget a full episode of hand keypoints to robot joint angles.

    If config and fk_fn are provided: uses full CasADi optimization (paper method).
    Otherwise: falls back to PCA approximation.

    Args:
        keypoints_seq: (T, 25, 3) world-frame joint positions
        config:        robot hand RetargetConfig (None = use PCA fallback)
        fk_fn:         CasADi forward kinematics function (None = use PCA)

    Returns:
        q_seq: (T, n_joints) robot joint angles
    """
    T, K, _ = keypoints_seq.shape

    if config is None or fk_fn is None:
        warnings.warn(
            "No RetargetConfig or FK function provided. "
            "Using PCA approximation instead of paper-faithful CasADi retargeting.",
            stacklevel=2,
        )
        n_joints = config.n_joints if config else 22
        return retarget_pca_approx(keypoints_seq, n_joints=n_joints)

    # Full CasADi retargeting with temporal smoothing
    n = config.n_joints
    q_seq   = np.zeros((T, n), dtype=np.float32)
    q_prev  = np.zeros(n, dtype=np.float32)    # warm-start: neutral pose

    for t in range(T):
        q_opt  = _retarget_frame_casadi(
            keypoints_seq[t], config, fk_fn, q_init=q_prev
        )
        # Exponential smoothing to reduce temporal jitter (Appendix D)
        if t > 0:
            q_opt = (config.smoothing_alpha * q_prev
                     + (1 - config.smoothing_alpha) * q_opt)
        q_seq[t] = q_opt
        q_prev   = q_opt

    return q_seq   # (T, 22)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import h5py

    hdf5_path = Path(
        "/home/grease/ego_dataset/work_bearlu/egodex/test/basic_pick_place/0.hdf5"
    )
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

    with h5py.File(hdf5_path) as f:
        kp_right = np.stack(
            [f[f"transforms/{j}"][:, :3, 3] for j in RIGHT_HAND_JOINTS], axis=1
        )   # (T, 25, 3)

    config = RetargetConfig.dex5_1_right()

    print(f"Dex5-1 right hand: {config.n_joints} joints")
    print(f"URDF: {config.urdf_path}")
    print(f"Joints:\n  " + "\n  ".join(
        f"{name:30s}  [{_math.degrees(lo):7.1f}° .. {_math.degrees(hi):6.1f}°]"
        for name, lo, hi in zip(config.joint_names, config.joint_lo, config.joint_hi)
    ))
    print()

    print(f"Keypoints shape: {kp_right.shape}")
    q_approx = retarget_pca_approx(kp_right, n_joints=config.n_joints)
    print(f"PCA approx q_right shape: {q_approx.shape}")
    print(f"q range: [{q_approx.min():.3f}, {q_approx.max():.3f}]")
    print()
    print("NOTE: For paper-faithful retargeting, install casadi:")
    print("  pip install casadi")
    print("  Then pass fk_fn to retarget_sequence() using yourdfpy or pinocchio FK.")
