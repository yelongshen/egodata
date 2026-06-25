# TODO — Manus Glove → Unitree Dex5-1 Teleoperation Roadmap

## 4-Step Training + Deployment Plan

---

### ✅ Step 1 — Pretraining on EgoDex (SONIC-style, no RGB/language)

**Goal:** Learn the distribution of natural hand motion trajectories from human data.

**Script:** `scripts/sonic_hand_flow.py`

**Data:** `egodex/test_step2/` — 3,243 episodes, (T, 20) Dex5-1 joint angles
         (produced by egodex_extract_egoscale.py → egodex_retarget_dex5.py)

**Run:**
```bash
conda run -n env_isaaclab python scripts/sonic_hand_flow.py train \
    --data_dir /home/grease/ego_dataset/work_bearlu/egodex/test_step2 \
    --ckpt_dir checkpoints/sonic_hand \
    --side right --steps 50000 --batch 2048
```

**What it learns:**
- Temporal smoothness of hand motion
- Joint correlations (DIP ≈ 0.8 × PIP, finger spread patterns)
- Natural grasp and release trajectories
- NOT: contact awareness, force modulation, object adaptation

**Output:** `checkpoints/sonic_hand/best.pt`

---

### ⬜ Step 2 — Analytical Constraint Mid-Training (quick, no simulation)

**Goal:** Enforce Dex5-1-specific physical constraints that human data doesn't capture.

**Script:** `scripts/sonic_hand_midtrain_sim.py`

**Layer freeze strategy (same as EgoScale Stage II):**
- FROZEN: transformer attention + FFN blocks (preserve human motion prior)
- UPDATED: time_mlp, cond_enc, out_proj, AdaLN (robot-specific adaptation)

**Constraints added:**
```
DIP coupling:   Pitch_X4 ≈ 0.8 × Pitch_X3  (mechanical coupling in Dex5-1)
Velocity:       |Δq_t| ≤ 0.25 rad/step at 100Hz
Joint range:    soft penalty beyond Dex5-1 URDF limits
```

**Run (analytical only, no IsaacLab needed):**
```bash
conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
    --pretrain checkpoints/sonic_hand/best.pt \
    --ckpt_dir checkpoints/sonic_hand_sim \
    --steps 20000 --batch 1024
```

**Run (with IsaacLab physics simulation):**
```bash
conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
    --pretrain checkpoints/sonic_hand/best.pt \
    --ckpt_dir checkpoints/sonic_hand_sim \
    --steps 20000 --use_isaaclab --sim_envs 32
```

**Estimated time:** ~20 min (analytical), ~1 hr (with IsaacLab)

**Output:** `checkpoints/sonic_hand_sim/best.pt`

---

### ⬜ Step 3 — Real Robot Paired Data Collection (highest ROI)

**Goal:** Align the model to the actual G1 + Dex5-1 robot's sensing and actuation.
This is the equivalent of EgoScale Stage II (50hr human + 4hr robot).

**What to collect:**
- Operator wears Manus glove and performs ~20-50 tabletop manipulation tasks
- Same tasks repeated with robot teleoperation (baseline `dex5_teleop_manus.py`)
- Record: Manus keypoints + robot encoder readings + wrist RGB (optional)

**Estimated collection:** 2-4 hours human demos + 30min robot demos per task set

**Training (fine-tune on real data):**
```bash
# Re-run Steps 1-2 preprocessing on collected real data:
python scripts/egodex_extract_egoscale.py --data_root real_data/
python scripts/egodex_retarget_dex5.py   --in_dir real_data/step1 --out_dir real_data/step2

# Fine-tune pretrained model on real data (small LR, fewer steps):
conda run -n env_isaaclab python scripts/sonic_hand_flow.py train \
    --data_dir real_data/step2 \
    --pretrain checkpoints/sonic_hand_sim/best.pt \
    --ckpt_dir checkpoints/sonic_hand_real \
    --steps 10000 --batch 512 --lr 3e-5
```

**Why this matters:**
- Closes the kinematic mismatch between EgoDex (Apple Vision Pro) and Manus glove
- Adapts model to G1's specific wrist orientation and workspace
- Learns the actual Dex5-1 encoder readings vs. commanded positions
- In EgoScale: this single step improved performance by ~21% (0.50 → 0.71)

---

### ⬜ Step 4 — (Optional) IsaacLab Grasp Task Mid-Training

**Goal:** Teach the model contact awareness and force modulation via simulation.
Only worth doing AFTER Step 3 has been attempted and specific failure modes identified.

**Simulation tasks to implement in IsaacLab:**
- `GraspCylinder`: close fingers on cylinder (cup, bottle) → lift 5cm
- `GraspSphere`:   close fingers on sphere (ball, fruit)
- `GraspBox`:      parallel grip on box (phone, TV remote)
- `PinchGrasp`:    thumb+index pinch on thin objects

**Physics signals added:**
```
Contact force:  fingertip contact sensors → know when to stop closing
Torque limits:  penalise exceeding rated motor torque (3 Nm)
Grasp stability: object doesn't slip when lifted
```

**Engineering required:** ~1-2 weeks
- Define USD object assets in IsaacLab
- Implement contact sensor API calls in IsaacLabHandSim.rollout()
- Design per-task reward functions
- Tune loss weights w_contact, w_torque, w_track

**Run:**
```bash
conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
    --pretrain checkpoints/sonic_hand_real/best.pt \
    --ckpt_dir checkpoints/sonic_hand_grasp \
    --steps 30000 --use_isaaclab --sim_envs 64 \
    --w_phys 0.5 --w_sim 0.5
```

---

## Deployment

After any completed training step, deploy via:
```bash
# Baseline (no model):
python scripts/dex5_teleop_manus.py --side right --dry_run

# With trained model:
# (integrate flow_sample() call into dex5_teleop_manus.py MidTrainedController)
conda run -n env_isaaclab python scripts/dex5_teleop_manus.py \
    --side right --model checkpoints/sonic_hand_sim/best.pt
```

---

## File Map

| Script | Stage | Purpose |
|--------|-------|---------|
| `egodex_extract_egoscale.py` | Data | Step 1: extract ΔW + keypoints from HDF5 |
| `egodex_retarget_dex5.py` | Data | Step 2: retarget keypoints → Dex5-1 (20 DoF) |
| `egodex_build_pretraining_dataset.py` | Data | Pack into HDF5 training chunks |
| `egodex_visualize_skeleton.py` | Util | Visualize hand skeleton animation |
| `sonic_hand_flow.py` | Training | Stage I: flow-matching pretraining |
| `sonic_hand_midtrain_sim.py` | Training | Stage II: physics constraint mid-training |
| `dex5_teleop_manus.py` | Deploy | Real-time Manus → Dex5-1 teleoperation |
| `egodex_retarget.py` | Util | Generic retargeting framework (any URDF) |

## Assets

- `assets/dex5_1/Dex5-URDF-L/` — Unitree Dex5-1 left hand URDF + meshes
- `assets/dex5_1/Dex5-URDF-R/` — Unitree Dex5-1 right hand URDF + meshes

## EgoDex Data (on Extreme SSD)

- `/media/grease/Extreme SSD/egodex/` — 1,748 GB raw zip archives (all 7 assets)
- `/home/grease/ego_dataset/work_bearlu/egodex/test/` — extracted test split (19 GB)
- `/home/grease/ego_dataset/work_bearlu/egodex/test_step1/` — Step 1 output (1.78 GB)
- `/home/grease/ego_dataset/work_bearlu/egodex/test_step2/` — Step 2 output (1.78 GB)
