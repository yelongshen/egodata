"""
egodex_retarget_dex3.py
-----------------------
Retarget EgoDex Apple Vision Pro hand keypoints → Unitree Dex3-1 (7 DoF).

Dex3-1 joint layout (right hand):
  Thumb  : thumb_0 (abduction, Y-axis)  [-60°..+60°]
           thumb_1 (MCP flexion, Z-axis) [-60°..+35°]
           thumb_2 (IP  flexion, Z-axis) [-100°..0°]
  Index  : index_0 (MCP, Z-axis)         [0°..90°]
           index_1 (PIP, Z-axis)         [0°..100°]
  Middle : middle_0 (MCP, Z-axis)        [0°..90°]
           middle_1 (PIP, Z-axis)        [0°..100°]

Four retargeting algorithms implemented:
  1. Angle-based   — geometric inter-segment angles (fast, no FK)
  2. Position IK   — scipy.optimize minimise 3-fingertip position error
  3. PCA           — project hand pose to 7-D robot subspace
  4. Linear-map    — normalise total finger curl to robot range

Usage:
  python scripts/egodex_retarget_dex3.py \
      --hdf5 samples/egodex/lock_unlock_key/9.hdf5 \
      --algo angle
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dex3-1 joint metadata
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "thumb_0",   # abduction  Y-axis  [-60°,+60°]
    "thumb_1",   # MCP        Z-axis  [-60°,+35°]
    "thumb_2",   # IP         Z-axis  [-100°,0°]
    "index_0",   # MCP        Z-axis  [0°,90°]
    "index_1",   # PIP        Z-axis  [0°,100°]
    "middle_0",  # MCP        Z-axis  [0°,90°]
    "middle_1",  # PIP        Z-axis  [0°,100°]
]
N_JOINTS = 7

_LO = np.radians(np.array([-60., -60., -100.,   0.,   0.,   0.,   0.], np.float32))
_HI = np.radians(np.array([ 60.,  35.,    0.,  90., 100.,  90., 100.], np.float32))

# URDF joint origins and axes (from dex3_1_r.urdf)
# [xyz_in_parent_frame, rotation_axis]
_JOINT_XYZS = {
    "thumb_0":  (np.array([0.0255,  0.0,    0.0   ]), np.array([0,1,0])),
    "thumb_1":  (np.array([-0.0025, 0.0193, 0.0   ]), np.array([0,0,1])),
    "thumb_2":  (np.array([0.0,     0.0458, 0.0   ]), np.array([0,0,1])),
    "index_0":  (np.array([0.0777, -0.0016, 0.0285]), np.array([0,0,1])),
    "index_1":  (np.array([0.0458,  0.0,    0.0   ]), np.array([0,0,1])),
    "middle_0": (np.array([0.0777, -0.0016,-0.0285]), np.array([0,0,1])),
    "middle_1": (np.array([0.0458,  0.0,    0.0   ]), np.array([0,0,1])),
}

# EgoDex keypoint indices (right hand)
# 0=wrist, 1-4=thumb, 5-9=index, 10-14=middle, 15-24=ring/little
KP = dict(
    wrist=0,
    thumb_knuckle=1, thumb_base=2, thumb_ip=3, thumb_tip=4,
    idx_meta=5, idx_knuckle=6, idx_pip=7, idx_dip=8, idx_tip=9,
    mid_meta=10, mid_knuckle=11, mid_pip=12, mid_dip=13, mid_tip=14,
)


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def _se3_inv_batch(T: np.ndarray) -> np.ndarray:
    R, t = T[:, :3, :3], T[:, :3, 3:]
    Ti = np.zeros_like(T)
    Ti[:, :3, :3] = R.transpose(0, 2, 1)
    Ti[:, :3, 3:] = -(R.transpose(0, 2, 1) @ t)
    Ti[:, 3, 3]   = 1.0
    return Ti


def _to_wrist(kp_se3: np.ndarray) -> np.ndarray:
    """(T,25,4,4) → (T,25,3) positions in wrist-local frame."""
    T = kp_se3.shape[0]
    wi = _se3_inv_batch(kp_se3[:, 0])
    return np.einsum("tij,tkj->tki",
                     wi[:, :3, :3],
                     kp_se3[:, :, :3, 3] - kp_se3[:, :1, :3, 3])


def _norm(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, eps)


def _angle(v1, v2):
    c = np.clip(np.sum(_norm(v1) * _norm(v2), axis=-1), -1., 1.)
    return np.arccos(c)


def _signed_angle(v1, v2, n):
    cross = np.cross(v1, v2)
    return np.arctan2(np.sum(cross * n, axis=-1), np.sum(v1 * v2, axis=-1))


def _project_out(v, n):
    return v - np.sum(v * n, axis=-1, keepdims=True) * n


# ---------------------------------------------------------------------------
# Forward kinematics (for visualisation and IK)
# ---------------------------------------------------------------------------

def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about arbitrary unit axis (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c+ux*ux*(1-c),     ux*uy*(1-c)-uz*s,  ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s,  c+uy*uy*(1-c),     uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s,  uz*uy*(1-c)+ux*s,  c+uz*uz*(1-c)   ],
    ])


def fk_dex3(q: np.ndarray) -> dict[str, np.ndarray]:
    """Compute 3-D positions of all Dex3-1 links given joint angles.

    Args:
        q: (7,) joint angles in radians [thumb0, thumb1, thumb2,
                                          idx0, idx1, mid0, mid1]

    Returns:
        dict link_name → (3,) position in palm frame
    """
    def joint_tf(name, q_i):
        xyz, ax = _JOINT_XYZS[name]
        R = _rot_axis(ax, q_i)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = xyz
        return T

    T_palm = np.eye(4)

    # Thumb chain
    T_t0 = T_palm @ joint_tf("thumb_0", q[0])
    T_t1 = T_t0   @ joint_tf("thumb_1", q[1])
    T_t2 = T_t1   @ joint_tf("thumb_2", q[2])

    # Index chain
    T_i0 = T_palm @ joint_tf("index_0", q[3])
    T_i1 = T_i0   @ joint_tf("index_1", q[4])

    # Middle chain
    T_m0 = T_palm @ joint_tf("middle_0", q[5])
    T_m1 = T_m0   @ joint_tf("middle_1", q[6])

    return {
        "palm":      T_palm[:3, 3],
        "thumb_0":   T_t0[:3, 3],
        "thumb_1":   T_t1[:3, 3],
        "thumb_tip": T_t2[:3, 3],
        "index_0":   T_i0[:3, 3],
        "index_tip": T_i1[:3, 3],
        "middle_0":  T_m0[:3, 3],
        "middle_tip":T_m1[:3, 3],
    }


# ---------------------------------------------------------------------------
# Algorithm 1 — Angle-based (geometric, vectorised over T)
# ---------------------------------------------------------------------------

def retarget_angle_based(kp_se3: np.ndarray,
                          smoothing: float = 0.3) -> np.ndarray:
    """Map EgoDex inter-segment angles → Dex3-1 joints.

    Fully analytical: no optimisation, no URDF FK.
    ~0.1 ms/episode.
    """
    kp = _to_wrist(kp_se3)   # (T,25,3)
    T = kp.shape[0]
    q = np.zeros((T, N_JOINTS), np.float32)

    # Palm frame
    palm_fwd  = _norm(kp[:, KP["mid_meta"]])
    palm_lat  = _norm(kp[:, KP["idx_meta"]] - kp[:, KP["mid_meta"]])
    palm_norm = _norm(np.cross(palm_fwd, palm_lat))

    # ── Thumb ──────────────────────────────────────────────────────────────
    b_cmc  = _norm(kp[:, KP["thumb_knuckle"]])
    b_prox = _norm(kp[:, KP["thumb_base"]] - kp[:, KP["thumb_knuckle"]])
    b_ip   = _norm(kp[:, KP["thumb_ip"]]   - kp[:, KP["thumb_base"]])

    # thumb_0: abduction — thumb out-of-plane angle (Y-axis in Dex3-1 = lateral)
    cmc_flat   = _norm(_project_out(b_cmc, palm_fwd))
    q[:, 0]    = _signed_angle(palm_norm, cmc_flat, palm_fwd)  # lateral swing

    # thumb_1: MCP flexion — angle between CMC and proximal phalanx
    q[:, 1]    = _angle(b_cmc, b_prox) * -1   # Dex3-1 thumb_1 is negative flex

    # thumb_2: IP flexion — angle between proximal and intermediate phalanx
    q[:, 2]    = _angle(b_prox, b_ip) * -1    # negative convention

    # ── Index ──────────────────────────────────────────────────────────────
    b_idx_meta = _norm(kp[:, KP["idx_knuckle"]] - kp[:, KP["idx_meta"]])
    b_idx_prox = _norm(kp[:, KP["idx_pip"]]     - kp[:, KP["idx_knuckle"]])
    b_idx_mid  = _norm(kp[:, KP["idx_dip"]]     - kp[:, KP["idx_pip"]])
    q[:, 3]    = _angle(b_idx_meta, b_idx_prox)   # MCP
    q[:, 4]    = _angle(b_idx_prox, b_idx_mid)    # PIP

    # ── Middle ─────────────────────────────────────────────────────────────
    b_mid_meta = _norm(kp[:, KP["mid_knuckle"]] - kp[:, KP["mid_meta"]])
    b_mid_prox = _norm(kp[:, KP["mid_pip"]]     - kp[:, KP["mid_knuckle"]])
    b_mid_mid  = _norm(kp[:, KP["mid_dip"]]     - kp[:, KP["mid_pip"]])
    q[:, 5]    = _angle(b_mid_meta, b_mid_prox)   # MCP
    q[:, 6]    = _angle(b_mid_prox, b_mid_mid)    # PIP

    np.clip(q, _LO, _HI, out=q)
    for t in range(1, T):
        q[t] = smoothing * q[t-1] + (1-smoothing) * q[t]
    return q


# ---------------------------------------------------------------------------
# Algorithm 2 — Position-based IK (scipy L-BFGS-B)
# ---------------------------------------------------------------------------

def _fingertip_targets(kp_local: np.ndarray) -> dict[str, np.ndarray]:
    """Extract wrist-frame fingertip positions from one frame."""
    # Scale EgoDex coords to Dex3-1 scale (Dex3-1 links ≈ 45-80 mm)
    # EgoDex positions are in metres; Dex3-1 URDF also in metres → no scaling
    return {
        "thumb_tip": kp_local[KP["thumb_tip"]],
        "index_tip": kp_local[KP["idx_tip"]],
        "middle_tip":kp_local[KP["mid_tip"]],
    }


def _ik_cost(q: np.ndarray, targets: dict) -> float:
    fk  = fk_dex3(q)
    err = 0.0
    for k, v in targets.items():
        d    = fk[k] - v
        err += float(np.dot(d, d))
    return err


def retarget_position_ik(kp_se3: np.ndarray,
                          n_iter: int = 30,
                          smoothing: float = 0.3) -> np.ndarray:
    """Minimise 3-D fingertip position error via L-BFGS-B.

    ~5-15 ms/episode depending on convergence.
    """
    from scipy.optimize import minimize

    kp = _to_wrist(kp_se3)   # (T,25,3)
    T  = kp.shape[0]
    q_seq = np.zeros((T, N_JOINTS), np.float32)
    q_prev = np.zeros(N_JOINTS, np.float32)

    for t in range(T):
        tgt = _fingertip_targets(kp[t])
        res = minimize(_ik_cost, q_prev, args=(tgt,),
                       method="L-BFGS-B",
                       bounds=list(zip(_LO, _HI)),
                       options={"maxiter": n_iter, "ftol": 1e-5})
        q_seq[t] = res.x.astype(np.float32)
        if smoothing > 0 and t > 0:
            q_seq[t] = smoothing * q_seq[t-1] + (1-smoothing) * q_seq[t]
        q_prev = q_seq[t]

    return q_seq


# ---------------------------------------------------------------------------
# Algorithm 3 — PCA projection
# ---------------------------------------------------------------------------

def retarget_pca(kp_se3: np.ndarray,
                 smoothing: float = 0.3) -> np.ndarray:
    """Project wrist-relative keypoints to 7-D robot subspace via PCA.

    No physical grounding — mathematically minimises reconstruction error.
    ~0.2 ms/episode (after fit).
    """
    kp = _to_wrist(kp_se3)   # (T,25,3)
    T  = kp.shape[0]

    # Flatten to (T,75) wrist-relative positions
    flat = kp.reshape(T, -1).astype(np.float64)
    mean = flat.mean(0)
    cen  = flat - mean
    _, _, Vt = np.linalg.svd(cen, full_matrices=False)
    proj  = cen @ Vt[:N_JOINTS].T            # (T, 7)

    # Normalise each component to [-1,1] then scale to robot range midpoint
    p_lo, p_hi = proj.min(0), proj.max(0)
    denom = np.where(p_hi - p_lo > 1e-6, p_hi - p_lo, 1.)
    proj_n = 2*(proj - p_lo)/denom - 1       # (T,7) in [-1,1]

    # Map to robot joint range midpoint ± half-range
    mid   = (_LO + _HI) / 2
    half  = (_HI - _LO) / 2
    q_seq = (mid + half * proj_n).astype(np.float32)
    np.clip(q_seq, _LO, _HI, out=q_seq)

    for t in range(1, T):
        q_seq[t] = smoothing * q_seq[t-1] + (1-smoothing) * q_seq[t]
    return q_seq


# ---------------------------------------------------------------------------
# Algorithm 4 — Linear mapping (normalised total finger curl)
# ---------------------------------------------------------------------------

def retarget_linear(kp_se3: np.ndarray,
                    smoothing: float = 0.3) -> np.ndarray:
    """Map total per-finger curl to robot joint range linearly.

    Ignores inter-joint coupling.  Fastest approach.
    ~0.05 ms/episode.
    """
    kp = _to_wrist(kp_se3)
    T  = kp.shape[0]
    q  = np.zeros((T, N_JOINTS), np.float32)

    # Thumb abduction: angle of thumb base vs palm normal
    palm_fwd  = _norm(kp[:, KP["mid_meta"]])
    palm_norm = _norm(np.cross(palm_fwd,
                               _norm(kp[:, KP["idx_meta"]] - kp[:, KP["mid_meta"]])))
    b_cmc     = _norm(kp[:, KP["thumb_knuckle"]])
    cmc_flat  = _norm(_project_out(b_cmc, palm_fwd))
    q[:, 0]   = np.clip(
        _signed_angle(palm_norm, cmc_flat, palm_fwd),
        _LO[0], _HI[0])

    # Thumb: total curl = angle(wrist→tip vs palm_fwd)
    thumb_dir  = _norm(kp[:, KP["thumb_tip"]])
    thumb_curl = _angle(palm_fwd, thumb_dir)   # 0 = extended, π = fully curled
    q[:, 1]    = np.clip(-thumb_curl * 0.5, _LO[1], _HI[1])  # MCP half of curl
    q[:, 2]    = np.clip(-thumb_curl * 0.5, _LO[2], _HI[2])  # IP  half of curl

    # Index total curl
    idx_dir  = _norm(kp[:, KP["idx_tip"]] - kp[:, KP["idx_meta"]])
    meta_dir = _norm(kp[:, KP["idx_knuckle"]] - kp[:, KP["idx_meta"]])
    idx_curl = _angle(meta_dir, idx_dir)
    q[:, 3]  = np.clip(idx_curl * 0.55, _LO[3], _HI[3])   # MCP ~55% of curl
    q[:, 4]  = np.clip(idx_curl * 0.45, _LO[4], _HI[4])   # PIP ~45%

    # Middle total curl
    mid_dir  = _norm(kp[:, KP["mid_tip"]] - kp[:, KP["mid_meta"]])
    meta_dir = _norm(kp[:, KP["mid_knuckle"]] - kp[:, KP["mid_meta"]])
    mid_curl = _angle(meta_dir, mid_dir)
    q[:, 5]  = np.clip(mid_curl * 0.55, _LO[5], _HI[5])
    q[:, 6]  = np.clip(mid_curl * 0.45, _LO[6], _HI[6])

    np.clip(q, _LO, _HI, out=q)
    for t in range(1, T):
        q[t] = smoothing * q[t-1] + (1-smoothing) * q[t]
    return q


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "angle":    retarget_angle_based,
    "ik":       retarget_position_ik,
    "pca":      retarget_pca,
    "linear":   retarget_linear,
}


def retarget_dex3(kp_se3: np.ndarray, algo: str = "angle",
                   smoothing: float = 0.3) -> np.ndarray:
    """Retarget one episode to Dex3-1.

    Args:
        kp_se3:    (T,25,4,4) Apple Vision Pro SE3 keypoints (world frame)
        algo:      "angle" | "ik" | "pca" | "linear"
        smoothing: exponential filter alpha

    Returns:
        q: (T,7) float32 joint angles in radians, within Dex3-1 limits
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from {list(ALGORITHMS)}")
    return ALGORITHMS[algo](kp_se3, smoothing=smoothing)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import h5py, time

    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path,
                   default=Path("samples/egodex/lock_unlock_key/9.hdf5"))
    p.add_argument("--algo", default="angle",
                   choices=list(ALGORITHMS))
    args = p.parse_args()

    with h5py.File(args.hdf5) as f:
        kp = np.stack(
            [f[f"transforms/right{n}"][:] if f"transforms/right{n}" in f
             else f[f"transforms/{n}"][:] for n in [
                 "Hand","ThumbKnuckle","ThumbIntermediateBase",
                 "ThumbIntermediateTip","ThumbTip",
                 "IndexFingerMetacarpal","IndexFingerKnuckle",
                 "IndexFingerIntermediateBase","IndexFingerIntermediateTip",
                 "IndexFingerTip","MiddleFingerMetacarpal","MiddleFingerKnuckle",
                 "MiddleFingerIntermediateBase","MiddleFingerIntermediateTip",
                 "MiddleFingerTip","RingFingerMetacarpal","RingFingerKnuckle",
                 "RingFingerIntermediateBase","RingFingerIntermediateTip",
                 "RingFingerTip","LittleFingerMetacarpal","LittleFingerKnuckle",
                 "LittleFingerIntermediateBase","LittleFingerIntermediateTip",
                 "LittleFingerTip"]], axis=1) if False else None

    # Simpler load using our existing extractor
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.egodex_extract_egoscale import RIGHT_HAND_JOINTS
    with h5py.File(args.hdf5) as f:
        kp_se3 = np.stack(
            [f[f"transforms/{j}"][:] for j in RIGHT_HAND_JOINTS], axis=1)
        T = kp_se3.shape[0]
        desc = f.attrs.get("llm_description", "")

    print(f"Episode: {args.hdf5}  T={T}")
    print(f"Description: {desc}")
    print()

    for algo_name, fn in ALGORITHMS.items():
        t0 = time.perf_counter()
        q = fn(kp_se3)
        ms = (time.perf_counter()-t0)*1000
        print(f"{algo_name:8s} → shape={q.shape}  "
              f"range=[{math.degrees(q.min()):+.1f}°, {math.degrees(q.max()):+.1f}°]  "
              f"time={ms:.1f}ms")
        for j, name in enumerate(JOINT_NAMES):
            lo_d = math.degrees(_LO[j])
            hi_d = math.degrees(_HI[j])
            mean_d = math.degrees(float(q[:,j].mean()))
            pct = (mean_d - lo_d) / max(hi_d - lo_d, 1) * 100
            bar = "█"*int(max(0,min(1,pct/100))*12) + "░"*(12-int(max(0,min(1,pct/100))*12))
            print(f"  {name:12s}[{bar}] {mean_d:+7.1f}°  [{lo_d:.0f}°..{hi_d:.0f}°]")
        print()
