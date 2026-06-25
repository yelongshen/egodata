"""
egodex_retarget_dex5.py — Step 2: angle-based retargeting to Dex5-1 (20 DoF).
"""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import numpy as np

_LIMITS_R = [
    (math.radians(-33.6), math.radians(39.0)),
    (math.radians(0.0),   math.radians(104.0)),
    (math.radians(0.0),   math.radians(101.1)),
    (math.radians(0.0),   math.radians(94.0)),
    (math.radians(-22.0), math.radians(22.0)),
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
    (math.radians(-22.0), math.radians(22.0)),
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
    (math.radians(-22.0), math.radians(22.0)),
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
    (math.radians(-22.0), math.radians(22.0)),
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
]
_LIMITS_L = list(_LIMITS_R); _LIMITS_L[1] = (math.radians(-104.0), math.radians(0.0))
_LO_R = np.array([l for l,_ in _LIMITS_R], np.float32)
_HI_R = np.array([h for _,h in _LIMITS_R], np.float32)
_LO_L = np.array([l for l,_ in _LIMITS_L], np.float32)
_HI_L = np.array([h for _,h in _LIMITS_L], np.float32)

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, eps)

def angle_between(v1, v2):
    c = np.clip(np.sum(normalize(v1) * normalize(v2), axis=-1), -1., 1.)
    return np.arccos(c)

def project_out(v, n):
    return v - np.sum(v * n, axis=-1, keepdims=True) * n

def signed_angle_in_plane(v1, v2, n):
    cross = np.cross(v1, v2)
    return np.arctan2(np.sum(cross * n, axis=-1), np.sum(v1 * v2, axis=-1))

def se3_inv_batch(T):
    R = T[:, :3, :3]; t = T[:, :3, 3:]
    Ti = np.zeros_like(T)
    Ti[:, :3, :3] = R.transpose(0,2,1)
    Ti[:, :3, 3:] = -(R.transpose(0,2,1) @ t)
    Ti[:, 3, 3]   = 1.0
    return Ti

def retarget_hand(kp_se3, side="right", smoothing_alpha=0.3):
    """(T,25,4,4) → (T,20) Dex5-1 joint angles [rad]."""
    T = kp_se3.shape[0]
    # Wrist-local 3D positions
    wi = se3_inv_batch(kp_se3[:, 0])               # (T,4,4)
    kp = np.einsum("tij,tkj->tki",
                   wi[:, :3, :3],
                   kp_se3[:, :, :3, 3] - kp_se3[:, :1, :3, 3])  # (T,25,3)

    # Palm frame
    palm_fwd  = normalize(kp[:, 10])               # toward middle metacarpal
    palm_lat  = normalize(kp[:, 20] - kp[:, 5])   # index→little (lateral)
    palm_norm = normalize(np.cross(palm_fwd, palm_lat))

    q = np.zeros((T, 20), np.float32)

    # ── Thumb ──
    b_cmc = normalize(kp[:, 1])
    b_t1  = normalize(kp[:, 2] - kp[:, 1])
    b_t2  = normalize(kp[:, 3] - kp[:, 2])
    b_t3  = normalize(kp[:, 4] - kp[:, 3])
    q[:,0] = signed_angle_in_plane(palm_fwd, normalize(project_out(b_cmc, palm_norm)), palm_norm)
    q[:,1] = angle_between(b_cmc, b_t1)
    q[:,2] = angle_between(b_t1,  b_t2)
    q[:,3] = angle_between(b_t2,  b_t3)

    # ── 4 fingers ──
    for fi, (mi,ki,ii,iti,ti) in enumerate([(5,6,7,8,9),(10,11,12,13,14),(15,16,17,18,19),(20,21,22,23,24)]):
        jb = 4 + fi*4
        b_meta = normalize(kp[:,ki] - kp[:,mi])
        b_prox = normalize(kp[:,ii] - kp[:,ki])
        b_mid  = normalize(kp[:,iti]- kp[:,ii])
        b_dist = normalize(kp[:,ti] - kp[:,iti])
        neutral = normalize(kp[:,mi])
        q[:,jb]   = signed_angle_in_plane(normalize(project_out(neutral, palm_norm)),
                                           normalize(project_out(b_meta,  palm_norm)), palm_norm)
        q[:,jb+1] = angle_between(b_meta, b_prox)
        q[:,jb+2] = angle_between(b_prox, b_mid)
        q[:,jb+3] = angle_between(b_mid,  b_dist)

    lo, hi = (_LO_L, _HI_L) if side == "left" else (_LO_R, _HI_R)
    np.clip(q, lo, hi, out=q)
    if smoothing_alpha > 0:
        for t in range(1, T):
            q[t] = smoothing_alpha * q[t-1] + (1-smoothing_alpha) * q[t]
    return q

def process_episode(args):
    in_path, out_path, alpha = args
    try:
        data = dict(np.load(in_path, allow_pickle=True))
        data["q_hand_right"] = retarget_hand(data["hand_keypoints_right"], "right", alpha)
        data["q_hand_left"]  = retarget_hand(data["hand_keypoints_left"],  "left",  alpha)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **data)
        return str(in_path.relative_to(in_path.parent.parent)), True, f"T={data['hand_keypoints_right'].shape[0]}"
    except Exception as e:
        return str(in_path), False, str(e)

def main():
    p = argparse.ArgumentParser()
    base = Path("/home/grease/ego_dataset/work_bearlu/egodex")
    p.add_argument("--in_dir",   type=Path, default=base/"test_step1")
    p.add_argument("--out_dir",  type=Path, default=base/"test_step2")
    p.add_argument("--workers",  type=int,  default=min(32, os.cpu_count() or 8))
    p.add_argument("--task",     type=str,  default=None)
    p.add_argument("--episode",  type=int,  default=None)
    p.add_argument("--smoothing",type=float,default=0.3)
    args = p.parse_args()

    pat = (f"{args.task}/{'%06d'%args.episode if args.episode is not None else '*'}.npz"
           if args.task else "**/*.npz")
    in_files = sorted(args.in_dir.glob(pat))
    print(f"Episodes: {len(in_files)}  Workers: {args.workers}")

    work = [(f, args.out_dir/f.relative_to(args.in_dir), args.smoothing) for f in in_files]
    t0 = time.perf_counter()
    if args.workers == 1 or len(in_files) == 1:
        results = [process_episode(w) for w in work]
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(process_episode, work, chunksize=8))
    elapsed = time.perf_counter() - t0

    ok  = sum(1 for _,s,_ in results if s)
    err = sum(1 for _,s,_ in results if not s)
    print(f"Done in {elapsed:.1f}s ({elapsed/max(len(in_files),1)*1000:.1f}ms/ep)  OK={ok}  ERR={err}")
    for r,s,m in results:
        if not s: print(f"  ERROR {r}: {m}")

    # Sample output
    good = [r for r,s,_ in results if s]
    if good:
        sample_out = args.out_dir / good[0]
        d = np.load(sample_out, allow_pickle=True)
        qr, ql = d["q_hand_right"], d["q_hand_left"]
        print(f"\nSample: {good[0]}")
        print(f"  q_hand_right: {qr.shape}  [{np.degrees(qr.min()):.1f}°, {np.degrees(qr.max()):.1f}°]")
        print(f"  q_hand_left:  {ql.shape}  [{np.degrees(ql.min()):.1f}°, {np.degrees(ql.max()):.1f}°]")
        jnames = ["Yaw_11","Roll_12","Pitch_13","Pitch_14",
                  "Roll_21","Pitch_22","Pitch_23","Pitch_24",
                  "Roll_31","Pitch_32","Pitch_33","Pitch_34",
                  "Roll_41","Pitch_42","Pitch_43","Pitch_44",
                  "Roll_51","Pitch_52","Pitch_53","Pitch_54"]
        print("  Right joints (ep mean):")
        for j,name in enumerate(jnames):
            lo_d = math.degrees(_LO_R[j]); hi_d = math.degrees(_HI_R[j])
            mean_d = math.degrees(float(qr[:,j].mean()))
            pct = (mean_d-lo_d)/max(hi_d-lo_d,1e-3)
            bar = "█"*int(max(0,min(1,pct))*12)+"░"*(12-int(max(0,min(1,pct))*12))
            print(f"    {name:12s}[{bar}] {mean_d:6.1f}°  [{lo_d:.0f}°..{hi_d:.0f}°]")

if __name__ == "__main__":
    main()
