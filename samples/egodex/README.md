# EgoDex Sample Episodes

Five short episodes from the EgoDex test split, captured with **Apple Vision Pro**.
Each episode provides:
- `.hdf5` — 25-joint hand pose (SE3 per joint, 30 FPS), camera pose, tracking confidence
- `.mp4`  — egocentric RGB video (1920×1080, 30 FPS)

## Samples

| Task | Episode | Frames | Duration | Description |
|---|---|---|---|---|
| `lock_unlock_key` | 9 | 22 | 0.7s | Pick up the pink lock and lock it using the key |
| `sort_beads` | 9 | 20 | 0.7s | Sort beads of different colors into separate piles |
| `charge_uncharge_airpods` | 45 | 29 | 1.0s | Charge AirPods using the lightning connector |
| `vertical_pick_place` | 49 | 32 | 1.1s | Pick up creamer and place it vertically |
| `stock_unstock_fridge` | 14 | 27 | 0.9s | Pick up food items and place into fridge |

## HDF5 structure

```python
import h5py
with h5py.File("lock_unlock_key/9.hdf5") as f:

    # Camera
    f["camera/intrinsic"]          # (3, 3) float32  — K matrix
    f["transforms/camera"]         # (T, 4, 4) float32 — camera pose in world

    # Hand pose (25 joints per hand, SE3 in world frame)
    # Joint layout: 0=wrist, 1-4=thumb, 5-9=index,
    #               10-14=middle, 15-19=ring, 20-24=little
    f["transforms/leftHand"]       # (T, 4, 4) — left wrist
    f["transforms/rightHand"]      # (T, 4, 4) — right wrist
    f["transforms/rightIndexFingerTip"]  # (T, 4, 4) — one of 25 joints
    # ... all 25 joints available for left and right

    # Tracking confidence (Apple Vision Pro reliability, 0–1)
    f["confidences/leftHand"]      # (T,) float32
    f["confidences/rightHand"]     # (T,) float32

    # Episode metadata
    f.attrs["task"]                # e.g. "lock_unlock_key"
    f.attrs["llm_description"]     # natural language description
    f.attrs["llm_objects"]         # e.g. ['lock', 'key']
    f.attrs["llm_verbs"]           # e.g. ['pick', 'lock']
    f.attrs["environment"]         # table/background description
```

## Quick start

```python
import h5py, numpy as np

with h5py.File("samples/egodex/lock_unlock_key/9.hdf5") as f:
    T       = f["transforms/camera"].shape[0]
    W_right = f["transforms/rightHand"][:]   # (T, 4, 4) right wrist in world
    conf    = f["confidences/rightHand"][:]  # (T,) tracking confidence
    task    = f.attrs["llm_description"]

# Compute relative wrist deltas (EgoScale ΔW action signal)
def se3_inv(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -(R.T @ t)
    return Ti

delta_W = [se3_inv(W_right[t]) @ W_right[t+1] for t in range(T-1)]
# delta_W[t]: 4x4 SE3 — wrist motion from frame t to t+1
```

## Source

Full dataset: https://github.com/apple/ml-egodex  
Paper: *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video* (arXiv:2505.11709)  
Captured with: Apple Vision Pro (ARKit 6-DOF SLAM + built-in hand tracking)
