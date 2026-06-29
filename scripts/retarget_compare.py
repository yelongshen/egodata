"""
retarget_compare.py
--------------------
Compare three retargeting algorithms on EgoDex HDF5 sample episodes.
Supports Unitree Dex5-1 (20 DoF, 5-finger) and Dex3-1 (7 DoF, 3-finger).

Algorithms
----------
1. Geometric (angle-based)  — egodex_retarget_dex5.py approach
   Fast analytic method: decomposes bone vectors into joint angles directly.

2. PCA approximation        — egodex_retarget.py fallback
   Projects wrist-relative keypoints into a low-dim space; no kinematic model.

3. Scipy NLP (numeric FK)   — optimization-based (comparable to CasADi paper method)
   Minimizes weighted FK reprojection error using yourdfpy + scipy L-BFGS-B.

Metrics
-------
  fk_err_m   : mean FK reprojection error (m) — how well FK(q) matches human kp
  vel_mean    : mean joint velocity (rad/step) — smoothness proxy
  vel_max     : max joint velocity
  limit_util  : fraction of joint range actually used (0=no motion, 1=full range)
  ms_per_frame: wall-clock time per frame

Usage
-----
  # Dex5-1 (default)
  python scripts/retarget_compare.py --hdf5 samples/egodex/sort_beads/9.hdf5
  # Dex3-1
  python scripts/retarget_compare.py --robot dex3_1
  python scripts/retarget_compare.py --robot dex3_1 --hdf5 samples/egodex/sort_beads/9.hdf5
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
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# ── Add repo src to path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ── Robot config registry ────────────────────────────────────────────────────
import math as _math
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class RobotConfig:
    name: str
    n_joints: int
    joint_names: List[str]
    limits: List[Tuple[float, float]]   # (lower, upper) radians per joint
    # EgoDex keypoint index → URDF link name
    kp_link_map: Dict[int, str]
    # EgoDex keypoint index → weight
    kp_weights: Dict[int, float]
    urdf_path: Path
    # link used as FK scale reference (index fingertip)
    scale_ref_link: str
    # EgoDex keypoint index for scale reference (index fingertip = kp 9)
    scale_ref_kp: int = 9

    @property
    def lo(self) -> np.ndarray:
        return np.array([l for l, _ in self.limits], np.float32)

    @property
    def hi(self) -> np.ndarray:
        return np.array([h for _, h in self.limits], np.float32)


def _make_dex5_1() -> RobotConfig:
    limits = [
        (_math.radians(-33.6), _math.radians(39.0)),
        (_math.radians(0.0),   _math.radians(104.0)),
        (_math.radians(0.0),   _math.radians(101.1)),
        (_math.radians(0.0),   _math.radians(94.0)),
        (_math.radians(-22.0), _math.radians(22.0)),
        (_math.radians(0.0),   _math.radians(90.0)),
        (_math.radians(0.0),   _math.radians(96.5)),
        (_math.radians(0.0),   _math.radians(80.0)),
        (_math.radians(-22.0), _math.radians(22.0)),
        (_math.radians(0.0),   _math.radians(90.0)),
        (_math.radians(0.0),   _math.radians(96.5)),
        (_math.radians(0.0),   _math.radians(80.0)),
        (_math.radians(-22.0), _math.radians(22.0)),
        (_math.radians(0.0),   _math.radians(90.0)),
        (_math.radians(0.0),   _math.radians(96.5)),
        (_math.radians(0.0),   _math.radians(80.0)),
        (_math.radians(-22.0), _math.radians(22.0)),
        (_math.radians(0.0),   _math.radians(90.0)),
        (_math.radians(0.0),   _math.radians(96.5)),
        (_math.radians(0.0),   _math.radians(80.0)),
    ]
    return RobotConfig(
        name="Dex5-1",
        n_joints=20,
        joint_names=[
            "Yaw_11", "Roll_12", "Pitch_13", "Pitch_14",
            "Roll_21", "Pitch_22", "Pitch_23", "Pitch_24",
            "Roll_31", "Pitch_32", "Pitch_33", "Pitch_34",
            "Roll_41", "Pitch_42", "Pitch_43", "Pitch_44",
            "Roll_51", "Pitch_52", "Pitch_53", "Pitch_54",
        ],
        limits=limits,
        kp_link_map={
            0: "base_link00", 1: "Link_11R", 2: "Link_12R", 3: "Link_13R", 4: "Link_14R",
            6: "Link_21R",  7: "Link_22R",  8: "Link_23R",  9: "Link_24R",
            11: "Link_31R", 12: "Link_32R", 13: "Link_33R", 14: "Link_34R",
            16: "Link_41R", 17: "Link_42R", 18: "Link_43R", 19: "Link_44R",
            21: "Link_51R", 22: "Link_52R", 23: "Link_53R", 24: "Link_54R",
        },
        kp_weights={
            0: 2.0, 4: 2.0, 9: 2.0, 14: 1.5, 19: 1.5, 24: 1.5,
            1: 1.0, 2: 1.0, 3: 1.0,
            6: 0.8, 7: 0.8, 8: 0.8,
            11: 0.6, 12: 0.6, 13: 0.6,
            16: 0.6, 17: 0.6, 18: 0.6,
            21: 0.5, 22: 0.5, 23: 0.5,
        },
        urdf_path=REPO_ROOT / "assets/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf",
        scale_ref_link="Link_24R",
        scale_ref_kp=9,
    )


def _make_dex3_1() -> RobotConfig:
    """Unitree Dex3-1 right hand — 7 DoF (thumb×3, index×2, middle×2).

    EgoDex keypoint → Dex3-1 link mapping:
      Thumb:  kp 0 (wrist/palm), 1→thumb_0, 2→thumb_1, 4→thumb_2
      Index:  kp 6→index_0, 9→index_1
      Middle: kp 11→middle_0, 14→middle_1
      (ring/little have no counterpart on Dex3-1 — omitted from FK loss)
    """
    limits = [
        # thumb
        (_math.radians(-60.0), _math.radians(60.0)),    # thumb_0: abduction
        (_math.radians(-60.0), _math.radians(35.0)),    # thumb_1: proximal flex
        (_math.radians(-100.0), _math.radians(0.0)),    # thumb_2: distal flex
        # index
        (_math.radians(0.0),   _math.radians(90.0)),    # index_0: MCP
        (_math.radians(0.0),   _math.radians(100.0)),   # index_1: DIP
        # middle
        (_math.radians(0.0),   _math.radians(90.0)),    # middle_0: MCP
        (_math.radians(0.0),   _math.radians(100.0)),   # middle_1: DIP
    ]
    return RobotConfig(
        name="Dex3-1",
        n_joints=7,
        joint_names=[
            "thumb_0", "thumb_1", "thumb_2",
            "index_0", "index_1",
            "middle_0", "middle_1",
        ],
        limits=limits,
        kp_link_map={
            0:  "right_hand_palm_link",
            1:  "right_hand_thumb_0_link",   # thumb knuckle
            2:  "right_hand_thumb_1_link",   # thumb proximal
            4:  "right_hand_thumb_2_link",   # thumb tip
            6:  "right_hand_index_0_link",   # index proximal (kp_idx=6: knuckle)
            9:  "right_hand_index_1_link",   # index tip
            11: "right_hand_middle_0_link",  # middle proximal
            14: "right_hand_middle_1_link",  # middle tip
        },
        kp_weights={
            0: 2.0,
            4: 2.0, 9: 2.0, 14: 1.5,        # fingertips
            1: 1.0, 2: 1.0,                  # thumb chain
            6: 0.8,  11: 0.6,                # knuckles
        },
        urdf_path=REPO_ROOT / "assets/dex3_1/dex3_1_r.urdf",
        scale_ref_link="right_hand_index_1_link",
        scale_ref_kp=9,
    )


ROBOT_CONFIGS = {
    "dex5_1": _make_dex5_1,
    "dex3_1": _make_dex3_1,
}

# ── Active robot config (set by main()) ──────────────────────────────────────
_ROBOT: RobotConfig = None  # type: ignore[assignment]


def _robot() -> RobotConfig:
    """Return the active robot config (set via --robot CLI arg)."""
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = _make_dex5_1()
    return _ROBOT


# ── EgoDex joint ordering ────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_hdf5(path: Path):
    """Load right-hand keypoints: returns (T,25,3) positions and (T,25,4,4) SE3."""
    with h5py.File(path) as f:
        kp_se3  = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_HAND_JOINTS], axis=1)
        kp_pos  = kp_se3[:, :, :3, 3]   # (T, 25, 3)
        conf    = (f["confidences/rightHand"][:] if "confidences/rightHand" in f
                   else np.ones(kp_pos.shape[0]))
        meta = {
            "task":     str(f.attrs.get("task", path.parent.name)),
            "episode":  int(path.stem),
            "language": str(f.attrs.get("llm_description") or f.attrs.get("description", "")),
        }
    return kp_pos, kp_se3, conf, meta


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 1 — Geometric angle-based (fast, analytical)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, eps)

def _angle_between(v1, v2):
    c = np.clip(np.sum(_normalize(v1) * _normalize(v2), axis=-1), -1., 1.)
    return np.arccos(c)

def _project_out(v, n):
    return v - np.sum(v * n, axis=-1, keepdims=True) * n

def _signed_angle(v1, v2, n):
    cross = np.cross(v1, v2)
    return np.arctan2(np.sum(cross * n, axis=-1), np.sum(v1 * v2, axis=-1))

def _se3_inv_batch(T):
    R = T[:, :3, :3]; t = T[:, :3, 3:]
    Ti = np.zeros_like(T)
    Ti[:, :3, :3] = R.transpose(0, 2, 1)
    Ti[:, :3, 3:] = -(R.transpose(0, 2, 1) @ t)
    Ti[:, 3, 3]   = 1.0
    return Ti

def retarget_geometric(kp_se3: np.ndarray, smoothing_alpha=0.3) -> np.ndarray:
    """Angle-based geometric retargeting. (T,25,4,4) → (T,N_joints).

    For Dex5-1 (20 DoF): decomposes all 5 fingers.
    For Dex3-1 (7 DoF):  decomposes only thumb, index, middle.
    """
    robot = _robot()
    T = kp_se3.shape[0]
    wi = _se3_inv_batch(kp_se3[:, 0])
    kp = np.einsum("tij,tkj->tki",
                   wi[:, :3, :3],
                   kp_se3[:, :, :3, 3] - kp_se3[:, :1, :3, 3])

    palm_fwd  = _normalize(kp[:, 10])
    palm_lat  = _normalize(kp[:, 20] - kp[:, 5])
    palm_norm = _normalize(np.cross(palm_fwd, palm_lat))

    if robot.n_joints == 7:  # Dex3-1
        q = np.zeros((T, 7), np.float32)
        # Thumb (joints 0-2)
        b_cmc = _normalize(kp[:, 1])
        b_t1  = _normalize(kp[:, 2] - kp[:, 1])
        b_t2  = _normalize(kp[:, 4] - kp[:, 2])   # tip - proximal (2 segments→1)
        q[:,0] = _signed_angle(palm_fwd, _normalize(_project_out(b_cmc, palm_norm)), palm_norm)
        q[:,1] = _angle_between(b_cmc, b_t1)
        q[:,2] = _angle_between(b_t1,  b_t2)
        # Index (joints 3-4)  kp: metacarpal=5, knuckle=6, tip=9
        b_i0 = _normalize(kp[:,6]  - kp[:,5])
        b_i1 = _normalize(kp[:,9]  - kp[:,6])
        q[:,3] = _angle_between(b_i0, b_i1)
        q[:,4] = _angle_between(b_i0, b_i1) * 0.8  # DIP coupled to MCP
        # Middle (joints 5-6) kp: metacarpal=10, knuckle=11, tip=14
        b_m0 = _normalize(kp[:,11] - kp[:,10])
        b_m1 = _normalize(kp[:,14] - kp[:,11])
        q[:,5] = _angle_between(b_m0, b_m1)
        q[:,6] = _angle_between(b_m0, b_m1) * 0.8
    else:  # Dex5-1 (20 DoF)
        q = np.zeros((T, 20), np.float32)
        b_cmc = _normalize(kp[:, 1])
        b_t1  = _normalize(kp[:, 2] - kp[:, 1])
        b_t2  = _normalize(kp[:, 3] - kp[:, 2])
        b_t3  = _normalize(kp[:, 4] - kp[:, 3])
        q[:,0] = _signed_angle(palm_fwd, _normalize(_project_out(b_cmc, palm_norm)), palm_norm)
        q[:,1] = _angle_between(b_cmc, b_t1)
        q[:,2] = _angle_between(b_t1,  b_t2)
        q[:,3] = _angle_between(b_t2,  b_t3)
        for fi, (mi,ki,ii,iti,ti) in enumerate([
            (5,6,7,8,9),(10,11,12,13,14),(15,16,17,18,19),(20,21,22,23,24)
        ]):
            jb = 4 + fi*4
            b_meta = _normalize(kp[:,ki] - kp[:,mi])
            b_prox = _normalize(kp[:,ii] - kp[:,ki])
            b_mid  = _normalize(kp[:,iti]- kp[:,ii])
            b_dist = _normalize(kp[:,ti] - kp[:,iti])
            neutral = _normalize(kp[:,mi])
            q[:,jb]   = _signed_angle(_normalize(_project_out(neutral, palm_norm)),
                                       _normalize(_project_out(b_meta, palm_norm)), palm_norm)
            q[:,jb+1] = _angle_between(b_meta, b_prox)
            q[:,jb+2] = _angle_between(b_prox, b_mid)
            q[:,jb+3] = _angle_between(b_mid,  b_dist)

    np.clip(q, robot.lo, robot.hi, out=q)
    if smoothing_alpha > 0:
        for t in range(1, T):
            q[t] = smoothing_alpha * q[t-1] + (1 - smoothing_alpha) * q[t]
    return q


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 2 — PCA approximation (baseline, no kinematics)
# ═══════════════════════════════════════════════════════════════════════════

def retarget_pca(kp_pos: np.ndarray) -> np.ndarray:
    """PCA-based pseudo-retargeting. (T,25,3) → (T,N_joints)."""
    robot = _robot()
    n_joints = robot.n_joints
    T = kp_pos.shape[0]
    relative = (kp_pos - kp_pos[:, 0:1, :]).reshape(T, -1).astype(np.float32)
    mean = relative.mean(0)
    centered = relative - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_joints].T
    p_min = projected.min(0, keepdims=True)
    p_max = projected.max(0, keepdims=True)
    denom = np.where(p_max - p_min > 1e-6, p_max - p_min, 1.0)
    lo = robot.lo; hi = robot.hi
    q_pca = lo + (2.0 * (projected - p_min) / denom) * (hi - lo) / 2.0
    return np.clip(q_pca, lo, hi)


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 3 — Scipy NLP with numeric FK (optimization-based)
# ═══════════════════════════════════════════════════════════════════════════

def _build_fk(urdf_path: Path, kp_link_map: dict, palm_link: str):
    """Return a function fk(q) → dict{link_name: (3,)} using yourdfpy."""
    import yourdfpy
    urdf_robot = yourdfpy.URDF.load(str(urdf_path))
    joint_names = [j.name for j in urdf_robot.actuated_joints]

    def fk(q: np.ndarray) -> dict:
        cfg = {name: float(v) for name, v in zip(joint_names, q)}
        urdf_robot.update_cfg(cfg)
        return {link: urdf_robot.get_transform(link, palm_link)[:3, 3]
                for link in kp_link_map.values() if link != palm_link}

    return fk, urdf_robot

def retarget_nlp(
    kp_pos: np.ndarray,       # (T, 25, 3) world-frame positions
    smoothing_alpha: float = 0.3,
) -> np.ndarray:
    """Scipy L-BFGS-B NLP retargeting with numeric FK. (T,25,3) → (T,N_joints)."""
    robot = _robot()
    palm_link = list(robot.kp_link_map.values())[0]   # first entry = palm/base
    fk_fn, _ = _build_fk(robot.urdf_path, robot.kp_link_map, palm_link)
    T = kp_pos.shape[0]
    n = robot.n_joints
    q_seq  = np.zeros((T, n), np.float32)
    q_prev = np.zeros(n, np.float32)

    kp_rel = kp_pos - kp_pos[:, 0:1, :]   # wrist-relative (T, 25, 3)

    # Scale factor: robot index-fingertip distance at zero pose vs human mean
    robot_zero_fk = fk_fn(np.zeros(n))
    robot_ref = robot_zero_fk.get(robot.scale_ref_link, np.array([0.1, 0., 0.]))
    human_ref = kp_rel[:, robot.scale_ref_kp, :].mean(0)
    scale = np.linalg.norm(robot_ref) / max(np.linalg.norm(human_ref), 1e-4)

    for t in range(T):
        target = kp_rel[t] * scale

        def objective(q, _target=target):
            try:
                fk = fk_fn(q)
            except Exception:
                return 1e6
            cost = 0.0
            for kp_idx, link in robot.kp_link_map.items():
                if link == palm_link:
                    continue
                w = robot.kp_weights.get(kp_idx, 1.0)
                diff = fk[link] - _target[kp_idx]
                cost += w * float(np.dot(diff, diff))
            return cost

        bounds = list(zip(robot.lo.tolist(), robot.hi.tolist()))
        result = minimize(objective, q_prev, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 30, "ftol": 1e-5})
        q_opt = result.x.astype(np.float32)

        if t > 0:
            q_opt = smoothing_alpha * q_prev + (1 - smoothing_alpha) * q_opt
        q_seq[t] = q_opt
        q_prev = q_opt

    return q_seq


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

def fk_reprojection_error(q_seq: np.ndarray, kp_pos: np.ndarray) -> float:
    """Mean FK reprojection error in meters."""
    robot = _robot()
    palm_link = list(robot.kp_link_map.values())[0]
    fk_fn, _ = _build_fk(robot.urdf_path, robot.kp_link_map, palm_link)
    kp_rel = kp_pos - kp_pos[:, 0:1, :]
    robot_zero = fk_fn(np.zeros(robot.n_joints))
    robot_scale = np.linalg.norm(robot_zero.get(robot.scale_ref_link, np.array([0.1,0,0])))
    human_scale = np.linalg.norm(kp_rel[:, robot.scale_ref_kp, :].mean(0))
    scale = robot_scale / max(human_scale, 1e-4)
    errors = []
    for t in range(q_seq.shape[0]):
        fk = fk_fn(q_seq[t])
        for kp_idx, link in robot.kp_link_map.items():
            if link == palm_link:
                continue
            w = robot.kp_weights.get(kp_idx, 1.0)
            diff = fk[link] - kp_rel[t, kp_idx] * scale
            errors.append(w * float(np.linalg.norm(diff)))
    return float(np.mean(errors))


def compute_metrics(q_seq: np.ndarray, kp_pos: np.ndarray, elapsed_s: float) -> dict:
    robot = _robot()
    T = q_seq.shape[0]
    dq = np.diff(q_seq, axis=0)
    vel_mean = float(np.abs(dq).mean())
    vel_max  = float(np.abs(dq).max())
    ranges = (robot.hi - robot.lo)
    used   = q_seq.max(0) - q_seq.min(0)
    limit_util = float((used / np.where(ranges > 1e-4, ranges, 1.0)).mean())
    fk_err = fk_reprojection_error(q_seq, kp_pos)
    return {
        "fk_err_m":    fk_err,
        "vel_mean":    vel_mean,
        "vel_max":     vel_max,
        "limit_util":  limit_util,
        "ms_per_frame": elapsed_s / T * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

ALGO_COLORS = {"Geometric": "#4ECDC4", "PCA": "#FFEAA7", "Scipy-NLP": "#FF6B6B"}

def plot_comparison(results: dict, meta: dict, out_path: Path) -> None:
    """
    results: {algo_name: {"q": (T,20), "metrics": {...}}}
    """
    algos  = list(results.keys())
    colors = [ALGO_COLORS.get(a, "#AAAAAA") for a in algos]
    T      = next(iter(results.values()))["q"].shape[0]
    t_ax   = np.arange(T)

    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")
    fig.suptitle(
        f"Retargeting Comparison — [{meta['task']}  ep {meta['episode']}]\n"
        f"{meta['language']}",
        color="white", fontsize=10, y=0.99,
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: joint angle traces for thumb and index ────────────────────────
    finger_info = [
        ("Thumb",  [0,1,2,3],    0, 0),
        ("Index",  [4,5,6,7],    0, 1),
        ("Middle", [8,9,10,11],  0, 2),
        ("Limit utilization", None, 0, 3),
    ]
    robot = _robot()
    for label, jids, row, col in finger_info:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1a1a2e")
        # filter jids to valid range for this robot
        if jids is not None:
            jids = [j for j in jids if j < robot.n_joints]
        if jids is not None and jids:
            for algo, color in zip(algos, colors):
                q = results[algo]["q"]
                for ji, jid in enumerate(jids):
                    ls = ["-","--",":","-."][ji % 4]
                    ax.plot(t_ax, np.degrees(q[:, jid]),
                            color=color, ls=ls, lw=1.2,
                            label=f"{algo}/{robot.joint_names[jid]}" if ji == 0 else None)
            ax.set_title(f"{label} joints (°)", color="white", fontsize=8)
            lo_deg = np.degrees(robot.lo[jids].min()); hi_deg = np.degrees(robot.hi[jids].max())
            ax.axhline(lo_deg, color="gray", lw=0.5, ls="--", alpha=0.5)
            ax.axhline(hi_deg, color="gray", lw=0.5, ls="--", alpha=0.5)
        elif not jids:
            ax.set_visible(False)
        else:
            # Bar chart: limit utilization per algo
            utils = [results[a]["metrics"]["limit_util"] for a in algos]
            bars = ax.bar(algos, utils, color=colors, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, utils):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", color="white", fontsize=7)
            ax.set_ylim(0, 1.1)
            ax.set_title("Limit utilization\n(fraction of range used)", color="white", fontsize=8)

        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Row 1: per-joint mean angle (bar chart per algo) ─────────────────────
    robot = _robot()
    n = robot.n_joints
    for col, algo in enumerate(algos[:3]):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#1a1a2e")
        q    = results[algo]["q"]
        means = np.degrees(q.mean(0))
        lo_d  = np.degrees(robot.lo); hi_d = np.degrees(robot.hi)
        bars = ax.bar(range(n), means, color=ALGO_COLORS.get(algo, "#AAA"),
                      edgecolor="white", linewidth=0.3, alpha=0.8)
        ax.bar(range(n), lo_d, color="none", edgecolor="#555", linewidth=0.5, ls="--")
        ax.bar(range(n), hi_d, color="none", edgecolor="#555", linewidth=0.5, ls="--")
        ax.set_xticks(range(n))
        ax.set_xticklabels(robot.joint_names, rotation=90, fontsize=5, color="white")
        ax.set_title(f"{algo} — per-joint mean (°)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Row 1, col 3: metrics table ───────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[1, 3])
    ax_tab.set_facecolor("#1a1a2e")
    ax_tab.axis("off")
    col_labels = ["Algorithm", "FK err (m)", "vel_mean", "vel_max", "util", "ms/f"]
    rows = []
    for algo in algos:
        m = results[algo]["metrics"]
        rows.append([
            algo,
            f"{m['fk_err_m']:.4f}",
            f"{m['vel_mean']:.4f}",
            f"{m['vel_max']:.3f}",
            f"{m['limit_util']:.2f}",
            f"{m['ms_per_frame']:.1f}",
        ])
    tbl = ax_tab.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r > 0 else "#2a2a4e")
        cell.set_edgecolor("#555")
        cell.set_text_props(color="white")
    ax_tab.set_title("Summary metrics", color="white", fontsize=8, pad=4)

    # ── Row 2: velocity profiles ──────────────────────────────────────────────
    for col, algo in enumerate(algos[:3]):
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor("#1a1a2e")
        q = results[algo]["q"]
        dq = np.abs(np.diff(q, axis=0))  # (T-1, 20)
        ax.plot(dq.mean(1), color=ALGO_COLORS.get(algo, "#AAA"), lw=1.5, label="mean |Δq|")
        ax.fill_between(range(len(dq)), dq.min(1), dq.max(1),
                        color=ALGO_COLORS.get(algo, "#AAA"), alpha=0.2)
        ax.set_title(f"{algo} — joint velocity (rad/step)", color="white", fontsize=8)
        ax.set_xlabel("Frame", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Row 2, col 3: FK error bar ────────────────────────────────────────────
    ax_fk = fig.add_subplot(gs[2, 3])
    ax_fk.set_facecolor("#1a1a2e")
    fk_errs = [results[a]["metrics"]["fk_err_m"] for a in algos]
    bars = ax_fk.bar(algos, fk_errs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, fk_errs):
        ax_fk.text(bar.get_x() + bar.get_width()/2, v + 0.0002, f"{v:.4f}m",
                   ha="center", va="bottom", color="white", fontsize=7)
    ax_fk.set_title("FK reprojection error (↓ better)", color="white", fontsize=8)
    ax_fk.tick_params(colors="white", labelsize=7)
    for spine in ax_fk.spines.values():
        spine.set_edgecolor("#444")

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color=ALGO_COLORS.get(a,"#AAA"), lw=2, label=a) for a in algos
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_one(hdf5_path: Path, out_dir: Path) -> None:
    robot = _robot()
    print(f"\n{'='*60}")
    print(f"Episode: {hdf5_path}")
    kp_pos, kp_se3, conf, meta = load_hdf5(hdf5_path)
    T = kp_pos.shape[0]
    print(f"  Task: {meta['task']}  ep {meta['episode']}  T={T}  "
          f"conf={conf.mean():.3f}")

    results = {}

    # ── Algorithm 1: Geometric ─────────────────────────────────────────────
    print("  [1/3] Geometric angle-based …", end=" ", flush=True)
    t0 = time.perf_counter()
    q_geo = retarget_geometric(kp_se3, smoothing_alpha=0.3)
    elapsed_geo = time.perf_counter() - t0
    print(f"{elapsed_geo*1000:.0f}ms total")
    m_geo = compute_metrics(q_geo, kp_pos, elapsed_geo)
    results["Geometric"] = {"q": q_geo, "metrics": m_geo}

    # ── Algorithm 2: PCA ───────────────────────────────────────────────────
    print("  [2/3] PCA approximation …", end=" ", flush=True)
    t0 = time.perf_counter()
    q_pca = retarget_pca(kp_pos)
    elapsed_pca = time.perf_counter() - t0
    print(f"{elapsed_pca*1000:.0f}ms total")
    m_pca = compute_metrics(q_pca, kp_pos, elapsed_pca)
    results["PCA"] = {"q": q_pca, "metrics": m_pca}

    # ── Algorithm 3: Scipy NLP ─────────────────────────────────────────────
    print("  [3/3] Scipy NLP (numeric FK) …", end=" ", flush=True)
    t0 = time.perf_counter()
    q_nlp = retarget_nlp(kp_pos, smoothing_alpha=0.3)
    elapsed_nlp = time.perf_counter() - t0
    print(f"{elapsed_nlp*1000:.0f}ms total")
    m_nlp = compute_metrics(q_nlp, kp_pos, elapsed_nlp)
    results["Scipy-NLP"] = {"q": q_nlp, "metrics": m_nlp}

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n  {'Algorithm':<14} {'FK err(m)':>10} {'vel_mean':>10} "
          f"{'vel_max':>9} {'util':>7} {'ms/f':>8}")
    print(f"  {'-'*60}")
    for algo, res in results.items():
        m = res["metrics"]
        print(f"  {algo:<14} {m['fk_err_m']:>10.4f} {m['vel_mean']:>10.4f} "
              f"{m['vel_max']:>9.3f} {m['limit_util']:>7.2f} {m['ms_per_frame']:>8.1f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    task  = meta["task"]
    ep    = meta["episode"]
    out_p = out_dir / f"{task}_ep{ep}_{robot.name}_retarget_compare.png"
    plot_comparison(results, meta, out_p)


def main():
    global _ROBOT
    parser = argparse.ArgumentParser(description="Compare retargeting algorithms on EgoDex samples.")
    parser.add_argument("--hdf5",    type=Path, default=None,
                        help="Single HDF5 to process (default: all samples/egodex/**/*.hdf5)")
    parser.add_argument("--out_dir", type=Path, default=Path("/tmp/retarget_compare"),
                        help="Output directory for comparison PNGs")
    parser.add_argument("--robot",   choices=list(ROBOT_CONFIGS.keys()),
                        default="dex5_1",
                        help="Target robot hand (default: dex5_1)")
    args = parser.parse_args()

    _ROBOT = ROBOT_CONFIGS[args.robot]()

    if not _ROBOT.urdf_path.exists():
        print(f"ERROR: URDF not found at {_ROBOT.urdf_path}")
        sys.exit(1)

    files = [args.hdf5] if args.hdf5 else sorted(
        (REPO_ROOT / "samples/egodex").rglob("*.hdf5")
    )
    print(f"Target robot : Unitree {_ROBOT.name} ({_ROBOT.n_joints} DoF)")
    print(f"URDF         : {_ROBOT.urdf_path}")
    print(f"Episodes     : {len(files)}\n")

    for f in files:
        run_one(f, args.out_dir)

    print(f"\nAll outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()
