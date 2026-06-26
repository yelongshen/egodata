"""
sonic_hand_eval.py
------------------
Offline evaluation of HandFlowTransformer vs two baselines,
using EgoDex test_step2 data as a proxy for real Manus→Dex5-1 teleoperation.

Simulated pipeline (no Manus glove required):
  EgoDex keypoints → retarget_hand_frame() + noise  →  q_manus  (simulated Manus)
  q_manus → [baseline_v1 | baseline_v2 | model]    →  q_pred
  Compare q_pred vs q_hand_right (ground truth from Step 2)

Metrics reported per method:
  1. MAE           mean |q_pred - q_target| per joint (degrees)
  2. RMSE          sqrt mean squared error (degrees)
  3. Vel_RMSE      trajectory velocity smoothness  (deg/step std)
  4. DIP_err       |q_DIP - 0.8×q_PIP| coupling violation (degrees)
  5. Range_viol    fraction of frames with joint outside [lo, hi]
  6. Correlation   Pearson R² between predicted and target trajectory

Usage:
  # Without pretrained model (baseline comparison only):
  conda run -n env_isaaclab python scripts/sonic_hand_eval.py \
      --data_dir /home/grease/ego_dataset/work_bearlu/egodex/test_step2 \
      --n_episodes 200

  # With pretrained model:
  conda run -n env_isaaclab python scripts/sonic_hand_eval.py \
      --data_dir /home/grease/ego_dataset/work_bearlu/egodex/test_step2 \
      --ckpt checkpoints/sonic_hand/best.pt \
      --n_episodes 200 --plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Joint metadata
# ---------------------------------------------------------------------------
_JOINT_NAMES = [
    "Yaw_11","Roll_12","Pitch_13","Pitch_14",          # thumb
    "Roll_21","Pitch_22","Pitch_23","Pitch_24",          # index
    "Roll_31","Pitch_32","Pitch_33","Pitch_34",          # middle
    "Roll_41","Pitch_42","Pitch_43","Pitch_44",          # ring
    "Roll_51","Pitch_52","Pitch_53","Pitch_54",          # little
]
_DIP_IDX = [3,  7,  11, 15, 19]   # Pitch_14/24/34/44/54
_PIP_IDX = [2,  6,  10, 14, 18]   # Pitch_13/23/33/43/53
COUPLING_RATIO = 0.8

_LO_DEG = np.array([
    -33.6, 0, 0, 0,
    -22, 0, 0, 0,
    -22, 0, 0, 0,
    -22, 0, 0, 0,
    -22, 0, 0, 0,
], dtype=np.float32)
_HI_DEG = np.array([
    39, 104, 101.1, 94,
    22, 90, 96.5, 80,
    22, 90, 96.5, 80,
    22, 90, 96.5, 80,
    22, 90, 96.5, 80,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Simulate Manus input from EgoDex keypoints
# ---------------------------------------------------------------------------

def simulate_manus_from_keypoints(
    kp_se3: np.ndarray,         # (T, 25, 4, 4) world-frame SE3
    noise_std: float = 0.05,    # radians — Manus tracking noise
) -> np.ndarray:
    """Simulate what a Manus glove would report for these EgoDex keypoints.

    Uses the same angle-based retarget as egodex_retarget_dex5.py,
    then adds Gaussian noise to represent:
      - Glove sensor quantization
      - Kinematic model mismatch
      - Tremor at high frequencies
    """
    from scripts.egodex_retarget_dex5 import retarget_hand  # batch version
    T = kp_se3.shape[0]
    # Unsmoothed retarget (smoothing_alpha=0 to get raw noisy signal)
    q_raw = retarget_hand(kp_se3, side="right", smoothing_alpha=0.0)   # (T, 20)
    noise = np.random.randn(T, 20).astype(np.float32) * noise_std
    q_manus = np.clip(q_raw + noise,
                      np.radians(_LO_DEG), np.radians(_HI_DEG))
    return q_manus   # (T, 20)


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------

def baseline_direct(q_manus: np.ndarray) -> np.ndarray:
    """Baseline v1: pass q_manus through unchanged (no smoothing, no model)."""
    return q_manus.copy()


def baseline_filtered(q_manus: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Baseline v2: exponential filter — what dex5_teleop_manus.py does."""
    T = q_manus.shape[0]
    q = np.zeros_like(q_manus)
    q[0] = q_manus[0]
    for t in range(1, T):
        q[t] = alpha * q[t-1] + (1 - alpha) * q_manus[t]
    return q


def model_predict(
    model,
    q_manus: np.ndarray,        # (T, 20)
    q_robot_init: np.ndarray,   # (20,)  initial robot state
    chunk_size: int = 16,
    n_denoise_steps: int = 10,
    device=None,
) -> np.ndarray:
    """Run pretrained model on a full episode, re-planning every stride steps."""
    import torch
    from scripts.sonic_hand_flow import flow_sample, N_JOINTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T    = q_manus.shape[0]
    q_out = np.zeros((T, N_JOINTS), dtype=np.float32)
    q_robot = q_robot_init.copy()
    stride  = chunk_size // 4   # re-plan every 4 steps (same as training)

    for start in range(0, T, stride):
        end = min(start + chunk_size, T)
        qm  = torch.tensor(q_manus[start], dtype=torch.float32, device=device)
        qr  = torch.tensor(q_robot,         dtype=torch.float32, device=device)
        chunk = flow_sample(model, qm, qr, n_steps=n_denoise_steps, device=device)
        chunk_np = chunk.cpu().numpy()   # (chunk_size, 20)
        n_fill   = end - start
        q_out[start:end] = chunk_np[:n_fill]
        # Update robot state for next planning step
        q_robot = chunk_np[min(stride, n_fill) - 1]

    return q_out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(q_pred: np.ndarray, q_target: np.ndarray,
                    valid_mask: np.ndarray) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        q_pred:     (T, 20) predicted joint angles [rad]
        q_target:   (T, 20) ground-truth joint angles [rad]
        valid_mask: (T,) boolean mask

    Returns:
        dict of metric_name → scalar
    """
    q_pred   = q_pred[valid_mask]
    q_target = q_target[valid_mask]
    if len(q_pred) == 0:
        return {}

    err_deg = np.degrees(np.abs(q_pred - q_target))  # (T, 20)

    # 1. Mean absolute error (degrees, per joint + overall)
    mae_per_joint = err_deg.mean(axis=0)   # (20,)
    mae_overall   = float(err_deg.mean())

    # 2. RMSE (degrees)
    rmse = float(np.sqrt(((q_pred - q_target)**2).mean())) * 180 / math.pi

    # 3. Velocity smoothness — std of per-step velocity
    vel = np.diff(q_pred, axis=0) * 180 / math.pi   # (T-1, 20) deg/step
    vel_rmse = float(np.abs(vel).mean())

    # 4. DIP coupling violation
    dip_err = np.abs(
        q_pred[:, _DIP_IDX] - COUPLING_RATIO * q_pred[:, _PIP_IDX]
    ) * 180 / math.pi
    dip_coupling = float(dip_err.mean())

    # 5. Joint range violations
    lo = np.radians(_LO_DEG)
    hi = np.radians(_HI_DEG)
    below = (q_pred < lo[None]).sum()
    above = (q_pred > hi[None]).sum()
    range_viol_pct = float(below + above) / q_pred.size * 100

    # 6. Pearson R² averaged across joints
    r2_vals = []
    for j in range(20):
        yp = q_pred[:, j]; yt = q_target[:, j]
        if yt.std() < 1e-6:
            continue
        corr = np.corrcoef(yp, yt)[0, 1]
        r2_vals.append(corr ** 2)
    r2 = float(np.mean(r2_vals)) if r2_vals else 0.0

    return {
        "MAE_deg":      mae_overall,
        "RMSE_deg":     rmse,
        "Vel_deg_step": vel_rmse,
        "DIP_err_deg":  dip_coupling,
        "RangeViol_%":  range_viol_pct,
        "R2":           r2,
        "mae_per_joint": mae_per_joint,  # (20,) array
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    import os

    # Collect episode files
    npz_files = sorted(
        Path(args.data_dir).rglob("*.npz"),
        key=lambda p: (p.parent.name, p.stem)
    )
    if args.n_episodes:
        rng = np.random.default_rng(42)
        npz_files = list(rng.choice(npz_files,
                                    min(args.n_episodes, len(npz_files)),
                                    replace=False))
    print(f"Evaluating on {len(npz_files)} episodes from {args.data_dir}\n")

    # Load model if provided
    model = None
    if args.ckpt and Path(args.ckpt).exists():
        import torch
        from scripts.sonic_hand_flow import HandFlowTransformer, N_JOINTS, CHUNK_SIZE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
        saved  = ckpt["args"]
        model  = HandFlowTransformer(
            n_joints=N_JOINTS,
            chunk_size=saved.get("chunk", CHUNK_SIZE),
            d_model=saved.get("d_model", 256),
            n_heads=saved.get("n_heads", 4),
            n_layers=saved.get("n_layers", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Model loaded: {args.ckpt}  (step={ckpt['step']}, "
              f"val_loss={ckpt['val_loss']:.5f})\n")
    else:
        print("No model checkpoint provided — comparing baselines only.\n")
        device = None

    # Accumulators per method
    methods = ["Baseline_direct", "Baseline_filtered"]
    if model is not None:
        methods.append("Model_flow")

    all_metrics: dict[str, list[dict]] = {m: [] for m in methods}
    all_mae_per_joint: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for ep_idx, path in enumerate(npz_files):
        try:
            d       = np.load(path, allow_pickle=True)
            kp_se3  = d["hand_keypoints_right"]   # (T, 25, 4, 4)
            q_gt    = d["q_hand_right"]            # (T, 20) ground truth
            valid   = d["valid_mask"]              # (T,) bool

            # Simulate Manus input
            q_manus = simulate_manus_from_keypoints(kp_se3, noise_std=args.noise_std)
            T = q_gt.shape[0]

            # Evaluate each method
            preds = {
                "Baseline_direct":   baseline_direct(q_manus),
                "Baseline_filtered": baseline_filtered(q_manus, alpha=0.3),
            }
            if model is not None:
                preds["Model_flow"] = model_predict(
                    model, q_manus, q_manus[0],
                    chunk_size=16, n_denoise_steps=args.n_denoise_steps,
                    device=device,
                )

            for method, q_pred in preds.items():
                m = compute_metrics(q_pred, q_gt, valid)
                if m:
                    all_metrics[method].append(m)
                    all_mae_per_joint[method].append(m["mae_per_joint"])

        except Exception as exc:
            print(f"  SKIP {path.name}: {exc}")
            continue

        if (ep_idx + 1) % 50 == 0:
            print(f"  Processed {ep_idx+1}/{len(npz_files)} episodes…")

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Metric':<20}", end="")
    for m in methods:
        print(f"  {m:>22}", end="")
    print()
    print("-"*70)

    scalar_keys = ["MAE_deg", "RMSE_deg", "Vel_deg_step",
                   "DIP_err_deg", "RangeViol_%", "R2"]
    results_summary: dict[str, dict] = {}

    for key in scalar_keys:
        print(f"{key:<20}", end="")
        for m in methods:
            vals  = [ep[key] for ep in all_metrics[m] if key in ep]
            mean  = float(np.mean(vals)) if vals else float("nan")
            results_summary.setdefault(m, {})[key] = mean
            fmt   = f"{mean:.2f}" if key != "R2" else f"{mean:.4f}"
            print(f"  {fmt:>22}", end="")
        print()

    print("="*70)

    # ── Per-joint MAE comparison ───────────────────────────────────────────
    print(f"\n{'Joint':15s}", end="")
    for m in methods:
        short = m.replace("Baseline_", "BL_").replace("Model_", "M_")
        print(f"  {short:>14}", end="")
    print()
    print("-"*65)

    for j, jname in enumerate(_JOINT_NAMES):
        lo_d = _LO_DEG[j]; hi_d = _HI_DEG[j]
        print(f"{jname:15s}", end="")
        for m in methods:
            maes = [ep[j] for ep in all_mae_per_joint[m]]
            mean = float(np.mean(maes)) if maes else float("nan")
            pct  = mean / max(hi_d - lo_d, 1) * 100
            bar  = "█" * int(pct / 5) + "░" * max(0, 10 - int(pct / 5))
            print(f"  [{bar}]{mean:5.1f}°", end="")
        print()

    # ── Summary verdict ───────────────────────────────────────────────────
    if model is not None and "Baseline_filtered" in results_summary \
            and "Model_flow" in results_summary:
        bl_mae   = results_summary["Baseline_filtered"]["MAE_deg"]
        m_mae    = results_summary["Model_flow"]["MAE_deg"]
        bl_dip   = results_summary["Baseline_filtered"]["DIP_err_deg"]
        m_dip    = results_summary["Model_flow"]["DIP_err_deg"]
        mae_improv = (bl_mae - m_mae) / bl_mae * 100
        dip_improv = (bl_dip - m_dip) / bl_dip * 100

        print(f"\n{'='*70}")
        print(f"Model vs Baseline_filtered:")
        print(f"  MAE improvement:          {mae_improv:+.1f}%  "
              f"({bl_mae:.2f}° → {m_mae:.2f}°)")
        print(f"  DIP coupling improvement: {dip_improv:+.1f}%  "
              f"({bl_dip:.2f}° → {m_dip:.2f}°)")
        verdict = "✅ Model improves over baseline" if mae_improv > 5 \
                  else "⚠️  Model improvement marginal (<5%)"
        print(f"\n  {verdict}")
        print(f"{'='*70}")

    # ── Optional trajectory plots ──────────────────────────────────────────
    if args.plot and len(npz_files) > 0:
        _plot_sample(npz_files[0], model, methods, device, args)


def _plot_sample(path, model, methods, device, args) -> None:
    """Plot one episode's predicted vs ground-truth trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    d       = np.load(path, allow_pickle=True)
    kp_se3  = d["hand_keypoints_right"]
    q_gt    = d["q_hand_right"]
    valid   = d["valid_mask"]
    q_manus = simulate_manus_from_keypoints(kp_se3, noise_std=args.noise_std)
    T       = q_gt.shape[0]
    t_axis  = np.arange(T) / 30.0   # seconds at 30 FPS

    preds = {
        "Ground truth": q_gt,
        "Baseline_direct": baseline_direct(q_manus),
        "Baseline_filtered": baseline_filtered(q_manus),
    }
    if model is not None:
        preds["Model_flow"] = model_predict(model, q_manus, q_manus[0],
                                             device=device)

    colors = {"Ground truth": "#2ecc71", "Baseline_direct": "#e74c3c",
              "Baseline_filtered": "#e67e22", "Model_flow": "#3498db"}

    # Plot 8 representative joints (thumb + index + middle)
    plot_joints = [0, 1, 2, 3, 4, 5, 6, 7]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), facecolor="#1a1a2e")
    fig.suptitle(f"Trajectory comparison — {path.parent.name} ep {path.stem}",
                 color="white", fontsize=11)

    for ax_idx, j in enumerate(plot_joints):
        ax = axes[ax_idx // 2][ax_idx % 2]
        ax.set_facecolor("#1a1a2e")
        for name, q in preds.items():
            ax.plot(t_axis, np.degrees(q[:, j]),
                    color=colors.get(name, "white"),
                    linewidth=1.5 if name == "Ground truth" else 1.0,
                    linestyle="-" if name == "Ground truth" else "--",
                    alpha=0.9, label=name)
        ax.set_title(_JOINT_NAMES[j], color="white", fontsize=9)
        ax.set_ylabel("degrees", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.set_ylim(_LO_DEG[j] - 5, _HI_DEG[j] + 5)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7,
                      facecolor="#1a1a2e", labelcolor="white",
                      edgecolor="none")

    plt.tight_layout()
    out = Path(args.out_dir) / "trajectory_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nTrajectory plot saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path("/home/grease/ego_dataset/work_bearlu/egodex")
    p = argparse.ArgumentParser(
        description="Offline evaluation: HandFlowModel vs baselines on EgoDex data."
    )
    p.add_argument("--data_dir",       type=Path, default=base/"test_step2",
                   help="Step 2 output directory")
    p.add_argument("--ckpt",           type=Path, default=None,
                   help="Pretrained model checkpoint (optional)")
    p.add_argument("--n_episodes",     type=int,  default=200,
                   help="Number of episodes to evaluate (default 200)")
    p.add_argument("--noise_std",      type=float, default=0.05,
                   help="Simulated Manus noise std in radians (default 0.05)")
    p.add_argument("--n_denoise_steps",type=int,  default=10,
                   help="Denoising steps for model inference (default 10)")
    p.add_argument("--plot",           action="store_true",
                   help="Generate trajectory comparison PNG")
    p.add_argument("--out_dir",        type=Path, default=Path("outputs"),
                   help="Output directory for plots")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
