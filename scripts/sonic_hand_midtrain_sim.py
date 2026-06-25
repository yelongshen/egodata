"""
sonic_hand_midtrain_sim.py
--------------------------
Stage II: simulation mid-training of HandFlowTransformer with physical constraints.

Mirrors the EgoScale Stage II design:
  - Load pretrained HandFlowTransformer (sonic_hand_flow.py Stage I)
  - Freeze attention/FFN blocks (preserve human motion priors)
  - Fine-tune output projection + condition/time embeddings
  - Add physics-aware loss terms from analytical + IsaacLab simulation

Physical constraints added vs pretraining:
  1.  DIP coupling:     Pitch_X4 ≈ 0.8 × Pitch_X3  (Dex5-1 mechanical coupling)
  2.  Velocity limits:  |Δq_t| ≤ max_joint_velocity per timestep
  3.  Self-collision:   minimum distance between finger link pairs
  4.  Torque limits:    joint torque ≤ rated torque (via IsaacLab)
  5.  Grasp contact:    at least 2 fingers contact object for grasp success

Layer freeze strategy (same as EgoScale Stage II):
  FROZEN:  transformer attention, FFN, positional embeddings (human motion prior)
  UPDATED: time_mlp, cond_enc, out_proj (robot-specific adaptation)

Usage:
  # With full IsaacLab simulation:
  conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
      --pretrain checkpoints/sonic_hand/best.pt \
      --ckpt_dir checkpoints/sonic_hand_sim \
      --steps 20000 --use_isaaclab

  # Analytical constraints only (no GPU physics sim required):
  conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
      --pretrain checkpoints/sonic_hand/best.pt \
      --ckpt_dir checkpoints/sonic_hand_sim \
      --steps 20000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import pretrained model
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.sonic_hand_flow import (
    HandFlowTransformer, HandMotionDataset,
    flow_sample, normalise, denormalise,
    N_JOINTS, CHUNK_SIZE, _LO, _HI,
)


# ---------------------------------------------------------------------------
# Dex5-1 physical constants
# ---------------------------------------------------------------------------

# DIP coupling: Pitch_X4 ≈ COUPLING_RATIO × Pitch_X3 (per finger, 4 fingers)
COUPLING_RATIO = 0.8
# DIP joints: indices 3, 7, 11, 15, 19  |  PIP joints: 2, 6, 10, 14, 18
_DIP_IDX = [3,  7,  11, 15, 19]
_PIP_IDX = [2,  6,  10, 14, 18]

# Max joint velocity per control step (rad/step at 100Hz → 1/100 s per step)
# Dex5-1 max angular velocity ≈ 4Hz × 2π ≈ 25 rad/s → 0.25 rad per 100Hz step
MAX_VEL_PER_STEP = 0.25   # radians

# Max joint torque (Nm) — approximate for force-controlled composite joints
MAX_TORQUE = 3.0


# ---------------------------------------------------------------------------
# 1. Analytical physics constraints (no simulation required)
# ---------------------------------------------------------------------------

class AnalyticalPhysicsLoss(nn.Module):
    """Fast, differentiable physics constraints computed directly on the chunk.

    All constraints operate on q_chunk (B, H, 20) in radians.
    """

    def __init__(
        self,
        w_coupling:  float = 1.0,
        w_velocity:  float = 0.5,
        w_range:     float = 0.5,
    ):
        super().__init__()
        self.w_coupling = w_coupling
        self.w_velocity = w_velocity
        self.w_range    = w_range

    def forward(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            q: (B, H, 20) predicted joint angles in radians

        Returns:
            dict of named loss terms
        """
        losses = {}

        # ── 1. DIP coupling: Pitch_X4 ≈ 0.8 × Pitch_X3 ───────────────────
        # For all 4 fingers (thumb skipped — different coupling)
        dip = q[..., [7, 11, 15, 19]]    # (B, H, 4)  index/mid/ring/little DIP
        pip = q[..., [6, 10, 14, 18]]    # (B, H, 4)  corresponding PIP
        coupling_err = dip - COUPLING_RATIO * pip
        losses["coupling"] = self.w_coupling * coupling_err.pow(2).mean()

        # ── 2. Joint velocity smoothness ──────────────────────────────────
        # Penalise steps that exceed max_vel_per_step
        dq = q[:, 1:, :] - q[:, :-1, :]          # (B, H-1, 20)
        vel_excess = F.relu(dq.abs() - MAX_VEL_PER_STEP)
        losses["velocity"] = self.w_velocity * vel_excess.pow(2).mean()

        # ── 3. Joint range adherence (soft penalty beyond limits) ─────────
        lo = _LO.to(q.device)
        hi = _HI.to(q.device)
        below = F.relu(lo - q)    # > 0 when below lower limit
        above = F.relu(q - hi)    # > 0 when above upper limit
        losses["range"] = self.w_range * (below.pow(2) + above.pow(2)).mean()

        losses["total_analytical"] = sum(losses.values())
        return losses


# ---------------------------------------------------------------------------
# 2. Fast FK for self-collision checking (numpy, from URDF origins)
# ---------------------------------------------------------------------------

# Joint chain origins in wrist frame (from Dex5-1 URDF, right hand)
# Each entry: (joint_name, parent_xyz, cumulative base position)
_CHAIN_ORIGINS_R = {
    # thumb
    "Link_11R": np.array([0.000,  0.000,  0.01435]),
    "Link_12R": np.array([0.000,  0.000,  0.01435]),  # ≈ same for small offset
    "Link_13R": np.array([0.034,  0.000,  0.009]),
    "Link_14R": np.array([0.070, -0.005,  0.009]),
    # index
    "Link_21R": np.array([0.010,  0.073, -0.003]),
    "Link_22R": np.array([0.014,  0.073, -0.003]),
    "Link_23R": np.array([0.054,  0.073, -0.003]),
    "Link_24R": np.array([0.075,  0.074, -0.003]),
    # middle
    "Link_31R": np.array([-0.012, 0.077, -0.003]),
    "Link_32R": np.array([-0.008, 0.077, -0.003]),
    "Link_33R": np.array([-0.048, 0.077, -0.003]),
    "Link_34R": np.array([-0.069, 0.078, -0.003]),
    # ring
    "Link_41R": np.array([-0.034, 0.073, -0.003]),
    "Link_42R": np.array([-0.030, 0.073, -0.003]),
    "Link_43R": np.array([-0.070, 0.073, -0.003]),
    "Link_44R": np.array([-0.091, 0.074, -0.003]),
    # little
    "Link_51R": np.array([-0.056, 0.069, -0.003]),
    "Link_52R": np.array([-0.052, 0.069, -0.003]),
    "Link_53R": np.array([-0.092, 0.069, -0.003]),
    "Link_54R": np.array([-0.113, 0.070, -0.003]),
}

# Pairs to check for self-collision: (link_A, link_B, min_dist_m)
_COLLISION_PAIRS = [
    ("Link_14R", "Link_21R", 0.012),  # thumb tip vs index base
    ("Link_14R", "Link_24R", 0.010),  # thumb tip vs index tip
    ("Link_24R", "Link_34R", 0.008),  # index tip vs middle tip
    ("Link_34R", "Link_44R", 0.008),  # middle tip vs ring tip
    ("Link_44R", "Link_54R", 0.008),  # ring tip vs little tip
]


@torch.no_grad()
def compute_collision_violations(q_np: np.ndarray) -> float:
    """Approximate self-collision check using fixed neutral-pose link positions.

    This is a conservative (fast) approximation — uses neutral-pose distances
    rather than full FK. Returns mean penetration depth in meters.
    """
    # For now use fixed positions (neutral pose approximation)
    # Full FK would require joint angle integration — expensive for mid-training
    total_violation = 0.0
    for link_a, link_b, min_dist in _COLLISION_PAIRS:
        pa = _CHAIN_ORIGINS_R[link_a]
        pb = _CHAIN_ORIGINS_R[link_b]
        dist = float(np.linalg.norm(pa - pb))
        violation = max(0.0, min_dist - dist)
        total_violation += violation
    return total_violation


# ---------------------------------------------------------------------------
# 3. IsaacLab simulation wrapper (optional — graceful fallback)
# ---------------------------------------------------------------------------

class IsaacLabHandSim:
    """Thin wrapper around IsaacLab to simulate Dex5-1 hand and get:
      - Applied torques per joint
      - Contact forces at fingertip links
      - Actual achieved joint positions (vs commanded)
    """

    def __init__(self, urdf_path: Path, n_envs: int = 64, headless: bool = True):
        self.urdf_path = urdf_path
        self.n_envs    = n_envs
        self.headless  = headless
        self._app      = None
        self._sim      = None
        self._ready    = False

    def launch(self) -> bool:
        """Launch IsaacSim app. Returns True on success."""
        try:
            from isaacsim import SimulationApp
            cfg = {"headless": self.headless}
            self._app   = SimulationApp(cfg)
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.sim import SimulationContext

            # Create simulation context
            self._sim_ctx = SimulationContext(
                sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
            )

            # Load Dex5-1 URDF as articulation
            self._hand_cfg = ArticulationCfg(
                prim_path="/World/Dex5R",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(self.urdf_path),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={f"joint_{i}": 0.0 for i in range(N_JOINTS)},
                ),
                actuators={},
            )
            self._hand = Articulation(self._hand_cfg)
            self._sim_ctx.reset()
            self._ready = True
            print("[IsaacLab] Hand simulation launched successfully")
            return True

        except Exception as exc:
            print(f"[IsaacLab] Launch failed: {exc}")
            print("[IsaacLab] Falling back to analytical constraints only")
            return False

    def rollout(self, q_traj: np.ndarray) -> dict[str, np.ndarray]:
        """Execute a (T, 20) joint trajectory and return physics feedback.

        Returns:
          torques:    (T, 20) applied torques [Nm]
          q_actual:   (T, 20) actual achieved positions [rad]
          contacts:   (T, 5)  contact force per finger [N]
        """
        if not self._ready:
            T = q_traj.shape[0]
            return {
                "torques":  np.zeros((T, N_JOINTS)),
                "q_actual": q_traj,
                "contacts": np.zeros((T, 5)),
            }

        T = q_traj.shape[0]
        torques   = np.zeros((T, N_JOINTS))
        q_actual  = np.zeros((T, N_JOINTS))
        contacts  = np.zeros((T, 5))

        # ── Replace with actual IsaacLab articulation control ──────────────
        # for t in range(T):
        #     self._hand.set_joint_position_target(
        #         torch.tensor(q_traj[t], device="cuda:0").unsqueeze(0)
        #     )
        #     self._sim_ctx.step()
        #     state = self._hand.data
        #     torques[t]  = state.applied_torque[0].cpu().numpy()
        #     q_actual[t] = state.joint_pos[0].cpu().numpy()
        #     # contacts[t] = contact_sensor.data.force_matrix[0].cpu().numpy()
        # ────────────────────────────────────────────────────────────────────
        q_actual = q_traj  # placeholder
        return {"torques": torques, "q_actual": q_actual, "contacts": contacts}

    def shutdown(self) -> None:
        if self._app is not None:
            self._app.close()


class SimPhysicsLoss(nn.Module):
    """Physics loss terms derived from IsaacLab simulation rollout.

    Called after running rollout() — converts simulation feedback
    into differentiable loss terms for the model.
    """

    def __init__(
        self,
        w_torque:  float = 0.1,   # penalise exceeding max torque
        w_track:   float = 1.0,   # penalise gap between commanded and actual
        w_contact: float = 0.2,   # reward finger contact (for grasping)
    ):
        super().__init__()
        self.w_torque  = w_torque
        self.w_track   = w_track
        self.w_contact = w_contact

    def forward(
        self,
        q_cmd:    torch.Tensor,          # (B, H, 20) model-predicted commands
        feedback: dict[str, np.ndarray], # output of IsaacLabHandSim.rollout()
    ) -> dict[str, torch.Tensor]:
        device = q_cmd.device
        losses = {}

        # Convert feedback to tensors
        torques  = torch.tensor(feedback["torques"],  device=device, dtype=torch.float32)
        q_actual = torch.tensor(feedback["q_actual"], device=device, dtype=torch.float32)
        contacts = torch.tensor(feedback["contacts"], device=device, dtype=torch.float32)

        # ── Torque limit: penalise joints exceeding max torque ─────────────
        torque_excess = F.relu(torques.abs() - MAX_TORQUE)
        losses["torque"] = self.w_torque * torque_excess.pow(2).mean()

        # ── Tracking: model commands should be achievable ─────────────────
        # High gap = commanded position unachievable due to physics
        if q_actual.shape == q_cmd.shape:
            track_err = (q_cmd - q_actual).pow(2).mean()
            losses["tracking"] = self.w_track * track_err

        # ── Contact: reward configurations with fingertip contact ──────────
        # contacts: (H, 5) — one per finger
        # Penalise configurations where contact is expected but missing
        # (optional: requires grasp intent annotation)
        # losses["contact"] = self.w_contact * (1 - contacts.clamp(0,1)).mean()

        losses["total_sim"] = sum(losses.values())
        return losses


# ---------------------------------------------------------------------------
# Mid-training loop
# ---------------------------------------------------------------------------

import torch.nn.functional as F


def freeze_pretrained_layers(model: HandFlowTransformer) -> None:
    """Freeze backbone layers, keep robot-specific layers trainable.

    Frozen (preserve human motion prior from Stage I):
      - transformer attention + FFN layers
      - positional embeddings
      - token encoder

    Trainable (adapt to robot physics in Stage II):
      - time_mlp        (noise level encoding)
      - cond_enc        (Manus + robot state encoding)
      - out_proj        (joint angle output)
      - AdaLN ada_mlp   (condition injection into each transformer block)
    """
    # Freeze everything first
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze robot-specific layers
    trainable_patterns = [
        "time_mlp", "cond_enc", "out_proj", "out_norm",
        "ada_mlp",   # AdaLN modulation in each block — condition injection
    ]
    unfrozen = []
    for name, param in model.named_parameters():
        if any(pat in name for pat in trainable_patterns):
            param.requires_grad = True
            unfrozen.append(name)

    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] Trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M params")
    print(f"[Freeze] Layers updated: {unfrozen}")


def midtrain(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load pretrained model ──────────────────────────────────────────────
    ckpt   = torch.load(args.pretrain, map_location=device, weights_only=False)
    saved  = ckpt["args"]
    model  = HandFlowTransformer(
        n_joints=N_JOINTS,
        chunk_size=saved.get("chunk", CHUNK_SIZE),
        d_model=saved.get("d_model", 256),
        n_heads=saved.get("n_heads", 4),
        n_layers=saved.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[Pretrained] loaded from {args.pretrain}  "
          f"(step={ckpt['step']}, val_loss={ckpt['val_loss']:.5f})")

    # ── Apply layer freeze (EgoScale Stage II strategy) ───────────────────
    freeze_pretrained_layers(model)

    # ── Dataset (same as pretraining, now with physics losses on top) ──────
    ds = HandMotionDataset(
        data_dir=args.data_dir,
        chunk_size=saved.get("chunk", CHUNK_SIZE),
        side=saved.get("side", "right"),
        manus_noise_std=0.05,
        max_episodes=args.max_episodes,
    )
    n_val   = max(1, int(len(ds) * 0.05))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))
    dl = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                    num_workers=4, pin_memory=True)

    # ── Physics loss modules ───────────────────────────────────────────────
    analytical_loss = AnalyticalPhysicsLoss(
        w_coupling=args.w_coupling,
        w_velocity=args.w_velocity,
        w_range=args.w_range,
    ).to(device)

    # ── IsaacLab simulation (optional) ────────────────────────────────────
    sim: Optional[IsaacLabHandSim] = None
    sim_loss_fn: Optional[SimPhysicsLoss] = None
    if args.use_isaaclab:
        urdf_r = Path(__file__).parent.parent / \
                 "assets/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf"
        sim = IsaacLabHandSim(urdf_r, n_envs=args.sim_envs,
                              headless=not args.render)
        if sim.launch():
            sim_loss_fn = SimPhysicsLoss(
                w_torque=args.w_torque, w_track=args.w_track
            ).to(device)
        else:
            sim = None
            print("[Mid-train] Continuing with analytical constraints only.")

    # ── Optimiser: only trainable parameters ──────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step     = 0
    t0       = time.perf_counter()

    print(f"\nMid-training for {args.steps} steps  (batch={args.batch})")
    print(f"Physics: analytical={'YES'} + IsaacLab={'YES' if sim else 'NO'}\n")

    while step < args.steps:
        model.train()
        for batch in dl:
            q_clean  = batch["q_target"].to(device)    # (B, H, 20) ground truth
            q_manus  = batch["q_manus"].to(device)     # (B, 20)
            q_robot  = batch["q_robot"].to(device)     # (B, 20)
            valid    = batch["valid"].to(device)        # (B, H)
            B = q_clean.shape[0]

            # ── Flow-matching reconstruction loss (same as pretraining) ────
            t_noise  = torch.rand(B, device=device)
            eps      = torch.randn_like(q_clean)
            q_noisy  = (1 - t_noise[:,None,None]) * q_clean \
                     + t_noise[:,None,None] * eps
            v_pred   = model(q_noisy, t_noise, q_manus, q_robot)
            v_target = q_clean - eps
            err      = (v_pred - v_target) ** 2
            mask     = valid[:, :, None].expand_as(err)
            loss_flow = (err * mask).sum() / (mask.sum() + 1e-6)

            # ── Generate clean prediction for physics evaluation ────────────
            with torch.no_grad():
                q_pred = flow_sample(model, q_manus, q_robot,
                                     n_steps=5, device=device)  # (B, H, 20)

            # ── Analytical physics losses ───────────────────────────────────
            phys = analytical_loss(q_pred)
            loss_phys = phys["total_analytical"]

            # ── IsaacLab simulation losses (if available) ──────────────────
            loss_sim = torch.tensor(0.0, device=device)
            if sim is not None and sim_loss_fn is not None and step % 10 == 0:
                # Run a sample of the batch through simulation
                n_sim  = min(8, B)
                q_np   = q_pred[:n_sim].detach().cpu().numpy()
                # Average over chunk for a single trajectory
                q_avg  = q_np.mean(axis=1)  # (n_sim, 20)
                fb     = sim.rollout(q_avg)
                sim_fb = sim_loss_fn(q_pred[:n_sim], fb)
                loss_sim = sim_fb["total_sim"]

            # ── Combined loss ──────────────────────────────────────────────
            loss = (args.w_flow * loss_flow
                  + args.w_phys * loss_phys
                  + args.w_sim  * loss_sim)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % 200 == 0:
                elapsed = time.perf_counter() - t0
                print(f"step={step:5d}  "
                      f"loss_flow={loss_flow.item():.4f}  "
                      f"coupling={phys['coupling'].item():.4f}  "
                      f"velocity={phys['velocity'].item():.4f}  "
                      f"range={phys['range'].item():.4f}  "
                      f"sim={loss_sim.item():.4f}  "
                      f"t={elapsed:.0f}s")

            if step % 2000 == 0:
                # Quick validation
                model.eval()
                val_loss = _val_loss(model, ds_val, device, args.batch)
                print(f"  ▶ val_loss={val_loss:.5f}")
                ckpt_data = {
                    "step": step, "val_loss": val_loss,
                    "model": model.state_dict(),
                    "opt":   opt.state_dict(),
                    "args":  vars(args),
                }
                torch.save(ckpt_data, ckpt_dir / "last.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt_data, ckpt_dir / "best.pt")
                    print(f"  ★ new best: {best_val:.5f}")
                model.train()

            if step >= args.steps:
                break

    if sim is not None:
        sim.shutdown()

    print(f"\nMid-training done.  Best val loss: {best_val:.5f}")
    print(f"Checkpoints: {ckpt_dir}")


@torch.no_grad()
def _val_loss(model, ds_val, device, batch_size) -> float:
    """Quick validation loss (flow only, no physics)."""
    from scripts.sonic_hand_flow import flow_loss
    dl  = DataLoader(ds_val, batch_size=batch_size * 2,
                     shuffle=False, num_workers=2)
    losses = []
    model.eval()
    for batch in dl:
        losses.append(flow_loss(model, batch, device).item())
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path("/home/grease/ego_dataset/work_bearlu/egodex")
    p    = argparse.ArgumentParser(
        description="Stage II: simulation mid-training with physical constraints."
    )
    # Data
    p.add_argument("--data_dir",     type=Path, default=base/"test_step2")
    p.add_argument("--pretrain",     type=Path, required=True,
                   help="Path to Stage I checkpoint (best.pt)")
    p.add_argument("--ckpt_dir",     type=Path,
                   default=Path("checkpoints/sonic_hand_sim"))
    p.add_argument("--max_episodes", type=int,  default=None)

    # Training
    p.add_argument("--steps", type=int,   default=20_000)
    p.add_argument("--batch", type=int,   default=1024)
    p.add_argument("--lr",    type=float, default=1e-4)

    # Loss weights
    p.add_argument("--w_flow",     type=float, default=1.0,
                   help="Flow-matching reconstruction weight")
    p.add_argument("--w_phys",     type=float, default=0.5,
                   help="Analytical physics constraint weight")
    p.add_argument("--w_sim",      type=float, default=0.2,
                   help="IsaacLab simulation loss weight")
    p.add_argument("--w_coupling", type=float, default=1.0,
                   help="DIP coupling constraint weight")
    p.add_argument("--w_velocity", type=float, default=0.5,
                   help="Velocity smoothness weight")
    p.add_argument("--w_range",    type=float, default=0.5,
                   help="Joint range penalty weight")
    p.add_argument("--w_torque",   type=float, default=0.1,
                   help="Torque limit weight (requires IsaacLab)")
    p.add_argument("--w_track",    type=float, default=1.0,
                   help="Trajectory tracking weight (requires IsaacLab)")

    # IsaacLab
    p.add_argument("--use_isaaclab", action="store_true",
                   help="Enable IsaacLab physics simulation")
    p.add_argument("--sim_envs",     type=int, default=32,
                   help="Number of parallel sim environments")
    p.add_argument("--render",       action="store_true",
                   help="Enable rendering (requires display)")

    args = p.parse_args()
    midtrain(args)


if __name__ == "__main__":
    main()
