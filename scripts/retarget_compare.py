"""
retarget_compare.py — Compare 4 Dex3-1 retargeting algorithms on EgoDex AVP data.

Usage:
  python scripts/retarget_compare.py --all_samples
  python scripts/retarget_compare.py --hdf5 samples/egodex/lock_unlock_key/9.hdf5
"""
from __future__ import annotations
import argparse, math, time, sys
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.egodex_retarget_dex3 import (
    retarget_dex3, fk_dex3, JOINT_NAMES, N_JOINTS,
    _LO, _HI, _to_wrist, ALGORITHMS, KP,
)

BG = "#0d1117"
ALGO_COLORS = {"angle":"#3498db","ik":"#2ecc71","pca":"#e67e22","linear":"#e74c3c"}
FCOLS = {"thumb":"#ff9ff3","index":"#54a0ff","middle":"#ffd32a","ring":"#aaa","little":"#aaa"}

SKEL = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(8,9),(0,10),(10,11),
        (11,12),(12,13),(13,14),(0,15),(15,16),(16,17),(17,18),(18,19),(0,20),
        (20,21),(21,22),(22,23),(23,24),(5,10),(10,15),(15,20)]

DEX3_BONES = [("palm","thumb_0"),("thumb_0","thumb_1"),("thumb_1","thumb_tip"),
              ("palm","index_0"),("index_0","index_tip"),
              ("palm","middle_0"),("middle_0","middle_tip")]

RIGHT_JOINTS = [
    "rightHand","rightThumbKnuckle","rightThumbIntermediateBase",
    "rightThumbIntermediateTip","rightThumbTip","rightIndexFingerMetacarpal",
    "rightIndexFingerKnuckle","rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerMetacarpal","rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip","rightRingFingerMetacarpal","rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase","rightRingFingerIntermediateTip",
    "rightRingFingerTip","rightLittleFingerMetacarpal","rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]

def _bone_col(a, b):
    if a<=4 or b<=4:  return FCOLS["thumb"]
    if a<=9 or b<=9:  return FCOLS["index"]
    if a<=14 or b<=14: return FCOLS["middle"]
    return "#555"

def load_ep(path):
    with h5py.File(path) as f:
        kp = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_JOINTS], axis=1)
        conf = f["confidences/rightHand"][:] if "confidences/rightHand" in f else np.ones(kp.shape[0])
        desc = str(f.attrs.get("llm_description") or f.attrs.get("description",""))
        task = str(f.attrs.get("task",""))
    return dict(kp=kp, kp_local=_to_wrist(kp), conf=conf, desc=desc, task=task, T=kp.shape[0])

def _style(ax):
    ax.set_facecolor(BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("none")
    ax.grid(True, color="white", alpha=0.04, linewidth=0.3)
    ax.tick_params(colors="#444", labelsize=6)
    for lab in [ax.xaxis.label,ax.yaxis.label,ax.zaxis.label]:
        lab.set_color("#555"); lab.set_fontsize(7)

def _eq(ax, pts):
    mn,mx = pts.min(0),pts.max(0); c=(mn+mx)/2; h=(mx-mn).max()/2*0.65
    ax.set_xlim(c[0]-h,c[0]+h); ax.set_ylim(c[1]-h,c[1]+h); ax.set_zlim(c[2]-h,c[2]+h)

def draw_human(ax, kp, alpha=1.0):
    for a,b in SKEL:
        ax.plot(*zip(kp[a],kp[b]), color=_bone_col(a,b), lw=1.6, alpha=alpha)
    ax.scatter(*kp[0], s=35, c="white", edgecolors="#888", lw=0.8, zorder=5)
    for i in [4,9,14]: ax.scatter(*kp[i], s=20, c="white", zorder=5, alpha=alpha)

def draw_dex3(ax, q, color, alpha=0.9, scale=1.0):
    fk = {k:v*scale for k,v in fk_dex3(q).items()}
    for a,b in DEX3_BONES:
        if a in fk and b in fk:
            ax.plot(*zip(fk[a],fk[b]), color=color, lw=2.2, alpha=alpha, ls="--")
    for k,v in fk.items():
        if k!="palm":
            ax.scatter(*v, s=22, c=color, edgecolors="white", lw=0.4, zorder=4, alpha=alpha)

def run(hdf5_path, out_dir, frame=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{hdf5_path.parent.name}_ep{hdf5_path.stem}"
    ep   = load_ep(hdf5_path)
    print(f"\n{'─'*55}\n{ep['task']}  T={ep['T']}  {ep['desc'][:65]}")

    all_q = {}; timings = {}
    for algo in ALGORITHMS:
        t0 = time.perf_counter()
        all_q[algo] = retarget_dex3(ep["kp"], algo=algo)
        timings[algo] = (time.perf_counter()-t0)*1000
        qd = np.degrees(all_q[algo])
        lo = np.degrees(_LO); hi = np.degrees(_HI)
        viol = ((qd<lo[None])|(qd>hi[None])).sum()
        print(f"  {algo:8s} {timings[algo]:5.1f}ms  "
              f"[{qd.min():+.1f}°,{qd.max():+.1f}°]  violations={viol}")

    fr = frame if frame is not None else ep["T"]//2
    kp = ep["kp_local"][fr]

    # ── Figure 1: Frame comparison (5 panels) ─────────────────────────────
    egodex_span = float(np.linalg.norm(kp[9]-kp[0]))
    dex3_span   = 0.077+0.046
    scale        = egodex_span/dex3_span if dex3_span>0 else 1.0

    fig = plt.figure(figsize=(18,5), facecolor=BG)
    gs  = gridspec.GridSpec(1,5,figure=fig,wspace=0.05)
    labels = ["Human (AVP)"] + [f"{a.upper()}\n{timings[a]:.1f}ms" for a in ALGORITHMS]
    axes = [fig.add_subplot(gs[i],projection="3d") for i in range(5)]

    for i,(ax,lbl) in enumerate(zip(axes,labels)):
        _style(ax)
        draw_human(ax, kp, alpha=0.25 if i>0 else 0.95)
        if i>0:
            algo = list(ALGORITHMS)[i-1]
            draw_dex3(ax, all_q[algo][fr], ALGO_COLORS[algo], scale=scale)
        _eq(ax, kp); ax.set_title(lbl, color="white", fontsize=9, pad=5)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    fig.suptitle(f"[{ep['task']}  fr{fr}/{ep['T']-1}]  {ep['desc'][:75]}",
                 color="white", fontsize=9, y=1.01)
    out1 = out_dir/f"{stem}_fr{fr}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig); print(f"  → {out1}")

    # ── Figure 2: Trajectory (7 rows × 4 methods) ─────────────────────────
    T = ep["T"]; t = np.arange(T)/30.
    fig,axes = plt.subplots(N_JOINTS,1,figsize=(13,11),facecolor=BG,sharex=True)
    fig.suptitle(f"Joint trajectories — {ep['task']}  {ep['desc'][:60]}",
                 color="white", fontsize=9, y=1.005)
    for j,(ax,jn) in enumerate(zip(axes,JOINT_NAMES)):
        ax.set_facecolor(BG)
        lo_d,hi_d = math.degrees(_LO[j]),math.degrees(_HI[j])
        ax.axhline(lo_d,color="#333",lw=0.5,ls=":"); ax.axhline(hi_d,color="#333",lw=0.5,ls=":")
        ax.fill_between(t,lo_d,hi_d,color="#fff",alpha=0.02)
        for algo,q in all_q.items():
            ax.plot(t, np.degrees(q[:,j]), color=ALGO_COLORS[algo], lw=1.4,
                    alpha=0.85, label=algo)
        ax.set_ylabel(jn,color="white",fontsize=8,rotation=0,ha="right",labelpad=55)
        ax.set_ylim(lo_d-5,hi_d+5); ax.tick_params(colors="#555",labelsize=7)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#222"); ax.spines["left"].set_color("#222")
        if j==0:
            ax.legend(loc="upper right",fontsize=8,facecolor=BG,
                      edgecolor="none",labelcolor="white",ncol=4)
    axes[-1].set_xlabel("time (s)",color="white",fontsize=8)
    plt.tight_layout()
    out2 = out_dir/f"{stem}_traj.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig); print(f"  → {out2}")

    # ── Figure 3: Metric bar chart ─────────────────────────────────────────
    metrics = {}
    for algo,q in all_q.items():
        dq = np.diff(q,axis=0)*180/math.pi
        qd = np.degrees(q)
        lo_d = np.degrees(_LO); hi_d = np.degrees(_HI)
        metrics[algo] = {
            "Velocity\n(°/step)": float(np.abs(dq).mean()),
            "Joint\nstd (°)":     float(qd.std()),
            "Violations\n(%)":    float(((qd<lo_d[None])|(qd>hi_d[None])).sum()/q.size*100),
        }
    mkeys = list(next(iter(metrics.values())).keys())
    algos = list(metrics.keys())
    x = np.arange(len(mkeys)); w = 0.18
    fig,ax = plt.subplots(figsize=(10,4),facecolor=BG)
    ax.set_facecolor(BG)
    for i,algo in enumerate(algos):
        vals = [metrics[algo][m] for m in mkeys]
        bars = ax.bar(x+i*w, vals, w, label=algo, color=ALGO_COLORS[algo], alpha=0.85)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=7)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(mkeys,color="white",fontsize=9)
    ax.tick_params(colors="white"); ax.set_facecolor(BG)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.legend(facecolor=BG,edgecolor="none",labelcolor="white",fontsize=9)
    ax.set_title(f"Algorithm metrics — {ep['task']}",color="white",fontsize=10)
    ax.set_ylabel("value",color="white",fontsize=9)
    plt.tight_layout()
    out3 = out_dir/f"{stem}_metrics.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig); print(f"  → {out3}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",        type=Path, default=None)
    p.add_argument("--out_dir",     type=Path, default=Path("outputs/retarget_compare"))
    p.add_argument("--frame",       type=int,  default=None)
    p.add_argument("--all_samples", action="store_true")
    args = p.parse_args()
    if args.all_samples:
        for hdf5 in sorted(Path("samples/egodex").rglob("*.hdf5")):
            run(hdf5, args.out_dir, args.frame)
    else:
        run(args.hdf5 or Path("samples/egodex/lock_unlock_key/9.hdf5"),
            args.out_dir, args.frame)

if __name__ == "__main__":
    main()
