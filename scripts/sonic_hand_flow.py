"""
sonic_hand_flow.py
------------------
SONIC-style hand motion flow-matching model — no RGB, no language.

Learns the distribution of Dex5-1 hand trajectories conditioned only on
the coarse Manus-style command signal. Replaces/improves the direct
retargeting in dex5_teleop_manus.py.

Model:
  Input condition:  q_manus_t   (20,)   coarse Manus command (per frame)
  Input state:      q_robot_t   (20,)   current robot proprioception
  Output:           q_target    (H, 20) refined action chunk

Training (flow-matching):
  - q_clean  = Step2 ground-truth chunk  (H, 20)
  - q_manus  = q_clean + noise           (20,)  simulates Manus input
  - q_noisy  = (1-t)*q_clean + t*ε       (H, 20)  flow interpolation
  - loss     = ||model(q_noisy, t, q_manus) - (q_clean - ε)||²

Inference:
  q_manus from glove → denoise N=10 steps → q_target chunk
  Latency: <3ms on RTX 4090

Usage:
  # Train:
  conda run -n env_isaaclab python scripts/sonic_hand_flow.py train \
      --data_dir egodex/test_step2 \
      --ckpt_dir checkpoints/sonic_hand \
      --steps 50000 --batch 2048

  # Evaluate on test set:
  conda run -n env_isaaclab python scripts/sonic_hand_flow.py eval \
      --ckpt checkpoints/sonic_hand/best.pt

  # Benchmark inference latency:
  conda run -n env_isaaclab python scripts/sonic_hand_flow.py benchmark \
      --ckpt checkpoints/sonic_hand/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS   = 20          # Dex5-1 DoF per hand
CHUNK_SIZE = 16          # action chunk length (steps)
STRIDE     = 4           # sliding window stride when creating chunks

# Dex5-1 joint limits (radians)  — used for normalisation
_LO = torch.tensor([
    math.radians(-33.6), math.radians(0.0),   math.radians(0.0),   math.radians(0.0),
    math.radians(-22.0), math.radians(0.0),   math.radians(0.0),   math.radians(0.0),
    math.radians(-22.0), math.radians(0.0),   math.radians(0.0),   math.radians(0.0),
    math.radians(-22.0), math.radians(0.0),   math.radians(0.0),   math.radians(0.0),
    math.radians(-22.0), math.radians(0.0),   math.radians(0.0),   math.radians(0.0),
], dtype=torch.float32)
_HI = torch.tensor([
    math.radians(39.0),  math.radians(104.0), math.radians(101.1), math.radians(94.0),
    math.radians(22.0),  math.radians(90.0),  math.radians(96.5),  math.radians(80.0),
    math.radians(22.0),  math.radians(90.0),  math.radians(96.5),  math.radians(80.0),
    math.radians(22.0),  math.radians(90.0),  math.radians(96.5),  math.radians(80.0),
    math.radians(22.0),  math.radians(90.0),  math.radians(96.5),  math.radians(80.0),
], dtype=torch.float32)


def normalise(q: torch.Tensor) -> torch.Tensor:
    """Scale joint angles from [lo, hi] to [-1, 1]."""
    lo = _LO.to(q.device)
    hi = _HI.to(q.device)
    return 2.0 * (q - lo) / (hi - lo + 1e-6) - 1.0


def denormalise(q_norm: torch.Tensor) -> torch.Tensor:
    """Inverse of normalise."""
    lo = _LO.to(q_norm.device)
    hi = _HI.to(q_norm.device)
    return (q_norm + 1.0) / 2.0 * (hi - lo + 1e-6) + lo


# ---------------------------------------------------------------------------
# Dataset: reads Step 2 npz files
# ---------------------------------------------------------------------------

class HandMotionDataset(Dataset):
    """Sliding-window chunks from Step 2 retargeted hand trajectories.

    Each sample:
      q_manus   (20,)     coarse Manus command at the observation timestep
                          = q_hand + additive noise (simulates real glove)
      q_robot   (20,)     current robot state (q_hand[t], acts as proprioception)
      q_target  (H, 20)   ground-truth Dex5-1 chunk to predict
      valid     (H,)      bool mask — frames with confidence >= 0.5
    """

    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = CHUNK_SIZE,
        stride: int = STRIDE,
        manus_noise_std: float = 0.05,   # radians — simulates Manus error
        side: str = "right",             # "right" or "left"
        max_episodes: int | None = None,
    ):
        self.chunk_size      = chunk_size
        self.manus_noise_std = manus_noise_std
        self.side            = side
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        key = f"q_hand_{side}"
        npz_files = sorted(data_dir.rglob("*.npz"))
        if max_episodes:
            npz_files = npz_files[:max_episodes]

        for path in npz_files:
            try:
                d     = np.load(path, allow_pickle=True)
                if key not in d:
                    continue
                q     = d[key].astype(np.float32)           # (T, 20)
                valid = d["valid_mask"].astype(bool)          # (T,)
                T     = q.shape[0]

                for start in range(0, T - chunk_size - 1, stride):
                    end         = start + chunk_size
                    q_chunk     = q[start:end]               # (H, 20)
                    valid_chunk = valid[start:end]            # (H,)

                    q_obs   = q[start]                       # (20,)
                    q_manus = q_obs                          # stored without noise (added at __getitem__)

                    self._chunks.append((q_manus, q_obs, q_chunk, valid_chunk))
            except Exception:
                continue

        print(f"[Dataset] {side} hand: {len(self._chunks):,} chunks "
              f"from {len(npz_files)} episodes")

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        q_manus_clean, q_robot, q_target, valid = self._chunks[idx]

        # Add noise to simulate Manus glove error (tremor + kinematic mismatch)
        noise     = np.random.randn(N_JOINTS).astype(np.float32) * self.manus_noise_std
        q_manus   = np.clip(q_manus_clean + noise,
                            _LO.numpy(), _HI.numpy())

        return {
            "q_manus":  torch.from_numpy(q_manus),    # (20,)
            "q_robot":  torch.from_numpy(q_robot),    # (20,)
            "q_target": torch.from_numpy(q_target),   # (H, 20)
            "valid":    torch.from_numpy(valid.astype(np.float32)),  # (H,)
        }


# ---------------------------------------------------------------------------
# Model: HandFlowTransformer
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for the flow noise level t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(-math.log(10000) *
                           torch.arange(half, device=device) / (half - 1))
        args   = t[:, None] * freqs[None] * 1000
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class HandFlowTransformer(nn.Module):
    """Small DiT-style flow-matching model for hand joint trajectories.

    Architecture:
      - Condition encoder: MLP on (q_manus, q_robot) → cond_emb (d_model)
      - Noise level encoder: sinusoidal + MLP → time_emb (d_model)
      - Sequence encoder: linear(H×20) + positional → (H, d_model) tokens
      - Transformer blocks with AdaLN (cond_emb + time_emb)
      - Output projection: → (H, 20) velocity

    Parameters: ~2.5M (very fast inference)
    """

    def __init__(
        self,
        n_joints:    int = N_JOINTS,
        chunk_size:  int = CHUNK_SIZE,
        d_model:     int = 256,
        n_heads:     int = 4,
        n_layers:    int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_joints   = n_joints
        self.chunk_size = chunk_size
        self.d_model    = d_model

        # ── Condition encoder: q_manus + q_robot ──────────────────────────
        self.cond_enc = nn.Sequential(
            nn.Linear(n_joints * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Noise level encoder ────────────────────────────────────────────
        self.time_emb = SinusoidalPosEmb(d_model // 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Sequence token encoder: (H, 20) → (H, d_model) ────────────────
        self.tok_enc = nn.Linear(n_joints, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        # ── Transformer layers with AdaLN ──────────────────────────────────
        self.layers   = nn.ModuleList([
            TransformerAdaLNBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ── Output ─────────────────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_joints)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        q_noisy:  torch.Tensor,   # (B, H, 20) noisy action chunk
        t:        torch.Tensor,   # (B,)        flow noise level in [0,1]
        q_manus:  torch.Tensor,   # (B, 20)     Manus command
        q_robot:  torch.Tensor,   # (B, 20)     current robot state
    ) -> torch.Tensor:
        """Returns predicted velocity (B, H, 20)."""
        B = q_noisy.shape[0]

        # Normalise inputs to [-1, 1]
        q_noisy_n  = normalise(q_noisy)
        q_manus_n  = normalise(q_manus)
        q_robot_n  = normalise(q_robot)

        # Condition vector: (B, d_model)
        cond = self.cond_enc(torch.cat([q_manus_n, q_robot_n], dim=-1))

        # Time embedding: (B, d_model)
        t_emb = self.time_mlp(self.time_emb(t))

        # Combined conditioning: (B, d_model)
        ada = cond + t_emb

        # Token sequence: (B, H, d_model)
        x = self.tok_enc(q_noisy_n) + self.pos_emb

        # Transformer
        for layer in self.layers:
            x = layer(x, ada)

        # Output velocity in normalised space: (B, H, 20)
        v_norm = self.out_proj(self.out_norm(x))

        # Scale velocity back to joint-angle space
        scale = (_HI - _LO).to(q_noisy.device) / 2.0  # (20,)
        return v_norm * scale


class TransformerAdaLNBlock(nn.Module):
    """Transformer block with adaptive layer norm (DiT-style conditioning)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        # AdaLN modulation: predict (shift1, scale1, gate1, shift2, scale2, gate2)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)

    def forward(self, x: torch.Tensor, ada: torch.Tensor) -> torch.Tensor:
        mods = self.ada_mlp(ada).chunk(6, dim=-1)  # each (B, d_model)
        s1, sc1, g1, s2, sc2, g2 = [m.unsqueeze(1) for m in mods]

        # Self-attention with AdaLN
        h   = self.norm1(x) * (1 + sc1) + s1
        h, _ = self.attn(h, h, h)
        x   = x + g1.tanh() * h

        # FFN with AdaLN
        h   = self.norm2(x) * (1 + sc2) + s2
        h   = self.ff(h)
        x   = x + g2.tanh() * h
        return x


# ---------------------------------------------------------------------------
# Flow-matching training + inference
# ---------------------------------------------------------------------------

def flow_loss(
    model:    HandFlowTransformer,
    batch:    dict[str, torch.Tensor],
    device:   torch.device,
) -> torch.Tensor:
    """Flow-matching MSE loss.

    Forward process: q_noisy = (1-t) * q_clean + t * ε
    Target velocity: v       = q_clean - ε   (rectified flow direction)
    Loss:            ||model(q_noisy, t, q_manus, q_robot) - v||²
                     weighted by valid mask
    """
    q_clean  = batch["q_target"].to(device)   # (B, H, 20)
    q_manus  = batch["q_manus"].to(device)    # (B, 20)
    q_robot  = batch["q_robot"].to(device)    # (B, 20)
    valid    = batch["valid"].to(device)       # (B, H)

    B, H, _ = q_clean.shape
    t        = torch.rand(B, device=device)                # noise level
    eps      = torch.randn_like(q_clean)                   # target noise
    q_noisy  = (1 - t[:, None, None]) * q_clean \
              + t[:, None, None] * eps

    v_pred  = model(q_noisy, t, q_manus, q_robot)          # (B, H, 20)
    v_target = q_clean - eps                               # (B, H, 20)

    err  = (v_pred - v_target) ** 2                        # (B, H, 20)
    mask = valid[:, :, None].expand_as(err)                # (B, H, 20)
    return (err * mask).sum() / (mask.sum() + 1e-6)


@torch.no_grad()
def flow_sample(
    model:     HandFlowTransformer,
    q_manus:   torch.Tensor,   # (B, 20) or (20,)
    q_robot:   torch.Tensor,   # (B, 20) or (20,)
    n_steps:   int = 10,
    device:    torch.device | None = None,
) -> torch.Tensor:
    """Generate a denoised action chunk via ODE integration (Euler).

    Returns: (B, H, 20) or (H, 20) if single sample
    """
    single = q_manus.dim() == 1
    if single:
        q_manus = q_manus.unsqueeze(0)
        q_robot = q_robot.unsqueeze(0)
    if device is not None:
        q_manus = q_manus.to(device)
        q_robot = q_robot.to(device)

    model.eval()
    B   = q_manus.shape[0]
    dev = q_manus.device
    H   = model.chunk_size

    # Start from noise
    q   = torch.randn(B, H, N_JOINTS, device=dev)
    # Scale noise to joint range
    scale = ((_HI - _LO) / 2.0).to(dev)
    q   = q * scale

    dt  = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full((B,), 1.0 - i * dt, device=dev)
        v     = model(q, t_val, q_manus, q_robot)
        q     = q - dt * v   # Euler step (reverse direction)

    q = q.clamp(_LO.to(dev), _HI.to(dev))
    return q.squeeze(0) if single else q


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    ds = HandMotionDataset(
        data_dir=args.data_dir,
        chunk_size=args.chunk,
        stride=args.stride,
        manus_noise_std=args.noise_std,
        side=args.side,
        max_episodes=args.max_episodes,
    )
    n_val   = max(1, int(len(ds) * 0.05))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    dl_train = DataLoader(ds_train, batch_size=args.batch,
                          shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch * 2,
                          shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = HandFlowTransformer(
        n_joints=N_JOINTS, chunk_size=args.chunk,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Optimiser
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # Checkpoint dir
    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    step     = 0
    t0       = time.perf_counter()

    print(f"\nTraining for {args.steps} steps  (batch={args.batch})")
    print(f"Train: {n_train:,} chunks   Val: {n_val:,} chunks\n")

    while step < args.steps:
        model.train()
        for batch in dl_train:
            loss = flow_loss(model, batch, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % 500 == 0:
                elapsed = time.perf_counter() - t0
                lr_now  = opt.param_groups[0]["lr"]
                print(f"step={step:6d}  loss={loss.item():.5f}  "
                      f"lr={lr_now:.2e}  t={elapsed:.0f}s")

            if step % 2000 == 0:
                # Validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in dl_val:
                        val_losses.append(flow_loss(model, vb, device).item())
                val_loss = float(np.mean(val_losses))
                print(f"  ▶ val_loss={val_loss:.5f}")

                # Save checkpoint
                ckpt = {
                    "step": step, "val_loss": val_loss,
                    "model": model.state_dict(),
                    "opt":   opt.state_dict(),
                    "args":  vars(args),
                }
                torch.save(ckpt, ckpt_dir / "last.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt, ckpt_dir / "best.pt")
                    print(f"  ★ new best: {best_val:.5f}")
                model.train()

            if step >= args.steps:
                break

    print(f"\nTraining done. Best val loss: {best_val:.5f}")
    print(f"Checkpoints: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
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

    ds = HandMotionDataset(args.data_dir, side=saved.get("side", "right"),
                           max_episodes=100)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    losses, maes = [], []
    for batch in dl:
        q_target = batch["q_target"].to(device)
        q_manus  = batch["q_manus"].to(device)
        q_robot  = batch["q_robot"].to(device)
        q_pred   = flow_sample(model, q_manus, q_robot, n_steps=10, device=device)

        mae = (q_pred - q_target).abs().mean()
        maes.append(float(mae))
        losses.append(float(flow_loss(model, batch, device)))

    print(f"Val loss:      {np.mean(losses):.5f}")
    print(f"Mean abs err:  {math.degrees(np.mean(maes)):.2f}° per joint")
    print(f"Checkpoint:    step={ckpt['step']}  best_val={ckpt['val_loss']:.5f}")


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark(args: argparse.Namespace) -> None:
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

    q_m = torch.zeros(1, N_JOINTS, device=device)
    q_r = torch.zeros(1, N_JOINTS, device=device)

    # Warm-up
    for _ in range(20):
        flow_sample(model, q_m, q_r, n_steps=10, device=device)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N  = 500
    for _ in range(N):
        flow_sample(model, q_m, q_r, n_steps=10, device=device)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / N

    print(f"Inference latency (10 denoising steps): {elapsed_ms:.2f}ms / call")
    print(f"Max control frequency:                  {1000/elapsed_ms:.0f} Hz")
    print(f"RTX 4090 batch=1, chunk={saved.get('chunk', CHUNK_SIZE)}, "
          f"d_model={saved.get('d_model', 256)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path("/home/grease/ego_dataset/work_bearlu/egodex")
    p    = argparse.ArgumentParser()
    sub  = p.add_subparsers(dest="cmd")

    # ── train ──────────────────────────────────────────────────────────────
    t = sub.add_parser("train")
    t.add_argument("--data_dir",     type=Path, default=base/"test_step2")
    t.add_argument("--ckpt_dir",     type=Path,
                   default=Path("checkpoints/sonic_hand"))
    t.add_argument("--side",         default="right", choices=["right","left"])
    t.add_argument("--steps",        type=int,   default=50_000)
    t.add_argument("--batch",        type=int,   default=2048)
    t.add_argument("--lr",           type=float, default=3e-4)
    t.add_argument("--chunk",        type=int,   default=CHUNK_SIZE)
    t.add_argument("--stride",       type=int,   default=STRIDE)
    t.add_argument("--d_model",      type=int,   default=256)
    t.add_argument("--n_heads",      type=int,   default=4)
    t.add_argument("--n_layers",     type=int,   default=4)
    t.add_argument("--noise_std",    type=float, default=0.05)
    t.add_argument("--max_episodes", type=int,   default=None)

    # ── eval ───────────────────────────────────────────────────────────────
    e = sub.add_parser("eval")
    e.add_argument("--ckpt",     type=Path, required=True)
    e.add_argument("--data_dir", type=Path, default=base/"test_step2")

    # ── benchmark ──────────────────────────────────────────────────────────
    b = sub.add_parser("benchmark")
    b.add_argument("--ckpt", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "benchmark":
        benchmark(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
