"""
visualize_retargeted_dex3.py
-----------------------------
Visualize EgoDex AVP hand data retargeted to Unitree Dex3-1 (7 DoF).
Uses angle-based geometric retargeting (egodex_retarget_dex3.py)
and yourdfpy FK for accurate 3-D link positions.

Layout:  [ Human (AVP) | Dex3-1 robot hand ]

Usage:
  python scripts/visualize_retargeted_dex3.py --all_samples
  python scripts/visualize_retargeted_dex3.py --hdf5 samples/egodex/lock_unlock_key/9.hdf5
  python scripts/visualize_retargeted_dex3.py --static --frame 10
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.egodex_retarget_dex3 import retarget_dex3, JOINT_NAMES, N_JOINTS, _LO, _HI
import yourdfpy as _ydfpy

URDF_PATH = REPO_ROOT / "assets/dex3_1/dex3_1_r.urdf"

# ─── Colours ─────────────────────────────────────────────────────────────────
BG   = "#0d1117"
FCOL = {"thumb":"#FF6B6B","index":"#4ECDC4","middle":"#FFD93D"}
ROB_COL = "#a29bfe"

# ─── Dex3-1 yourdfpy FK ───────────────────────────────────────────────────────
JOINT_NAMES_URDF = [
    "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
    "right_hand_index_0_joint","right_hand_index_1_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint",
]
# retarget_dex3 order: thumb_0,thumb_1,thumb_2, index_0,index_1, middle_0,middle_1
# yourdfpy actuated:   thumb_0,thumb_1,thumb_2, middle_0,middle_1, index_0,index_1
REORDER_TO_URDF = [0, 1, 2, 5, 6, 3, 4]

LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link","right_hand_thumb_1_link","right_hand_thumb_2_link",
    "right_hand_index_0_link","right_hand_index_1_link",
    "right_hand_middle_0_link","right_hand_middle_1_link",
]
BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(0,6),(6,7),(4,6)]
def _bcol(p,c):
    if p<=3 or c<=3: return FCOL["thumb"]
    if p<=5 or c<=5: return FCOL["index"]
    return FCOL["middle"]

_robot_obj = None
def _robot():
    global _robot_obj
    if _robot_obj is None:
        _robot_obj = _ydfpy.URDF.load(str(URDF_PATH))
    return _robot_obj

def fk(q7: np.ndarray) -> np.ndarray:
    """(7,) in retarget order → (8,3) link positions."""
    rob = _robot()
    q_urdf = q7[REORDER_TO_URDF]
    rob.update_cfg({n: float(v) for n,v in zip(rob.actuated_joint_names, q_urdf)})
    return np.array([rob.get_transform(l)[:3,3] for l in LINKS], dtype=np.float32)

def fk_batch(q_seq: np.ndarray) -> np.ndarray:
    return np.stack([fk(q_seq[t]) for t in range(len(q_seq))])

# joint limits for bar chart — retarget_dex3 order
_LO_DEG = np.degrees(_LO)
_HI_DEG = np.degrees(_HI)
DEX3_LIM = dict(x=(-0.02,0.15),y=(-0.12,0.12),z=(-0.06,0.06))

# ─── EgoDex joints ───────────────────────────────────────────────────────────
RIGHT_JOINTS = [
    "rightHand","rightThumbKnuckle","rightThumbIntermediateBase",
    "rightThumbIntermediateTip","rightThumbTip","rightIndexFingerMetacarpal",
    "rightIndexFingerKnuckle","rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerMetacarpal","rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip","rightRingFingerMetacarpal","rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerMetacarpal","rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]
H_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(8,9),(0,10),(10,11),(11,12),
           (12,13),(13,14),(0,15),(15,16),(16,17),(17,18),(18,19),(0,20),(20,21),(21,22),
           (22,23),(23,24),(5,10),(10,15),(15,20)]
def _hcol(a,b):
    if a<=4 or b<=4: return FCOL["thumb"]
    if a<=9 or b<=9: return FCOL["index"]
    if a<=14 or b<=14: return FCOL["middle"]
    return "#555"

def load_ep(path):
    with h5py.File(path) as f:
        kp = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_JOINTS], axis=1)
        conf = f["confidences/rightHand"][:] if "confidences/rightHand" in f else np.ones(kp.shape[0])
        desc = str(f.attrs.get("llm_description") or f.attrs.get("description",""))
        task = str(f.attrs.get("task", path.parent.name)); ep = int(path.stem)
    kp_local = kp[:,:,:3,3] - kp[:,0:1,:3,3]
    return dict(kp=kp, kp_local=kp_local, conf=conf, desc=desc, task=task, ep=ep, T=kp.shape[0])

# ─── Drawing ─────────────────────────────────────────────────────────────────
def _sax(ax, robot=False):
    ax.set_facecolor(BG)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor("none")
    ax.grid(True,color="white",alpha=0.04,lw=0.3)
    ax.tick_params(colors="#444",labelsize=5)
    for l in [ax.xaxis.label,ax.yaxis.label,ax.zaxis.label]:
        l.set_color("#666"); l.set_fontsize(6)
    if robot:
        ax.view_init(elev=20, azim=160)
        ax.set_xlabel("X (ext→)"); ax.set_ylabel("Y (curl↑)"); ax.set_zlabel("Z (lat)")
        ax.set_xlim(*DEX3_LIM["x"]); ax.set_ylim(*DEX3_LIM["y"]); ax.set_zlim(*DEX3_LIM["z"])

def _eq(ax, pts):
    c=(pts.max(0)+pts.min(0))/2; h=(pts.max(0)-pts.min(0)).max()/2*0.7
    ax.set_xlim(c[0]-h,c[0]+h); ax.set_ylim(c[1]-h,c[1]+h); ax.set_zlim(c[2]-h,c[2]+h)

def draw_human(ax, kp, alpha=0.9):
    for a,b in H_BONES:
        ax.plot(*zip(kp[a],kp[b]),color=_hcol(a,b),lw=1.4,alpha=alpha)
    ax.scatter(*kp[0],s=30,c="white",edgecolors="#888",lw=0.8,zorder=5)
    for i in [4,9,14,19,24]: ax.scatter(*kp[i],s=14,c="white",zorder=5,alpha=alpha)

def draw_dex3(ax, pts, color=ROB_COL, alpha=0.92):
    for a,b in BONES:
        lw = 2.5 if (a,b)!=(4,6) else 0.7
        al = alpha if (a,b)!=(4,6) else 0.2
        ax.plot(*zip(pts[a],pts[b]),color=color,lw=lw,alpha=al)
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=22,c=color,
               edgecolors="white",lw=0.4,zorder=4,alpha=alpha)
    ax.scatter(*pts[0],s=55,c="white",edgecolors="#555",lw=1,zorder=6)
    for i,lbl in {0:"palm",3:"thumb",5:"idx",7:"mid"}.items():
        ax.text(*pts[i],f" {lbl}",color="#aaa",fontsize=5)

def _jbar(ax, q_deg, color=ROB_COL):
    ax.set_facecolor(BG)
    r=_HI_DEG-_LO_DEG; pct=(q_deg-_LO_DEG)/np.where(r>1e-4,r,1.)
    cols=[FCOL["thumb"]]*3+[FCOL["index"]]*2+[FCOL["middle"]]*2
    ax.barh(range(7),pct,color=cols,edgecolor="none",height=0.65,alpha=0.85)
    ax.barh(range(7),[1]*7,color="none",edgecolor="#444",height=0.65,lw=0.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"{n} {v:+.0f}°" for n,v in zip(JOINT_NAMES,q_deg)],
                        fontsize=5.5,color="white")
    ax.set_xlim(0,1.1); ax.set_xticks([0,1])
    ax.set_xticklabels(["lo","hi"],fontsize=5,color="white")
    for s in ax.spines.values(): s.set_edgecolor("#333")

# ─── Static plot ─────────────────────────────────────────────────────────────
def plot_static(ep, q_seq, fk_seq, frame, out_path):
    kp  = ep["kp_local"][frame]
    all_kp = ep["kp_local"].reshape(-1,3)
    q_deg = np.degrees(q_seq[frame])

    fig = plt.figure(figsize=(15,9), facecolor=BG)
    gs  = gridspec.GridSpec(2,2,figure=fig,hspace=0.3,wspace=0.1,
                            height_ratios=[2.5,1],width_ratios=[1,1])

    ax_h = fig.add_subplot(gs[0,0],projection="3d")
    _sax(ax_h); _eq(ax_h, all_kp)
    draw_human(ax_h, kp)
    ax_h.set_title("Human hand (Apple Vision Pro)", color="white", fontsize=9, pad=5)

    ax_r = fig.add_subplot(gs[0,1],projection="3d")
    _sax(ax_r, robot=True)
    draw_dex3(ax_r, fk_seq[frame])
    ax_r.set_title("Dex3-1 retargeted (angle-based geometric)", color=ROB_COL, fontsize=9, pad=5)

    ax_info = fig.add_subplot(gs[1,0]); ax_info.axis("off")
    ax_info.text(0.5, 0.5,
        f"Task: {ep['task']}\nEpisode {ep['ep']}  Frame {frame}/{ep['T']-1}\n{ep['desc'][:65]}",
        color="white", fontsize=8, ha="center", va="center",
        transform=ax_info.transAxes, wrap=True)

    ax_j = fig.add_subplot(gs[1,1])
    _jbar(ax_j, q_deg)
    ax_j.set_title("Dex3-1 joint angles (7 DoF)",color=ROB_COL,fontsize=8,pad=3)

    fig.suptitle(
        f"EgoDex → Dex3-1  [{ep['task']}  ep{ep['ep']}  fr{frame}/{ep['T']-1}]  "
        f"{ep['desc'][:70]}",
        color="white",fontsize=9,y=1.005)
    fig.savefig(str(out_path),dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close(fig)
    print(f"  → {out_path}")

# ─── Animation ────────────────────────────────────────────────────────────────
def make_animation(ep, q_seq, fk_seq, out_path, fps=30, stride=1):
    T=ep["T"]; frames=list(range(0,T,stride))
    all_kp=ep["kp_local"].reshape(-1,3)

    fig=plt.figure(figsize=(14,5),facecolor=BG)
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.05,width_ratios=[1.2,1.2,0.6])
    ax_h=fig.add_subplot(gs[0,0],projection="3d")
    ax_r=fig.add_subplot(gs[0,1],projection="3d")
    ax_j=fig.add_subplot(gs[0,2])
    title_obj=fig.suptitle("",color="white",fontsize=8)

    def update(fi):
        t=frames[fi]
        ax_h.cla(); _sax(ax_h); _eq(ax_h,all_kp)
        draw_human(ax_h,ep["kp_local"][t])
        ax_h.set_title("Human (AVP)",color="white",fontsize=8,pad=4)

        ax_r.cla(); _sax(ax_r,robot=True)
        draw_dex3(ax_r,fk_seq[t])
        ax_r.set_title("Dex3-1 (geometric retarget)",color=ROB_COL,fontsize=8,pad=4)

        ax_j.cla(); _jbar(ax_j,np.degrees(q_seq[t]))
        ax_j.set_title("Joint angles",color=ROB_COL,fontsize=7,pad=2)
        title_obj.set_text(
            f"[{ep['task']}  ep{ep['ep']}]  fr{t}/{T-1}  {ep['desc'][:55]}")
        return []

    anim=animation.FuncAnimation(fig,update,frames=len(frames),interval=1000//fps,blit=False)
    writer=animation.FFMpegWriter(fps=fps,bitrate=2000,extra_args=["-pix_fmt","yuv420p"])
    out_path.parent.mkdir(parents=True,exist_ok=True)
    print(f"  Rendering {len(frames)} frames …")
    anim.save(str(out_path),writer=writer)
    plt.close(fig)
    print(f"  → {out_path}")

# ─── Run one episode ─────────────────────────────────────────────────────────
def run_one(hdf5_path, out_dir, args):
    ep = load_ep(hdf5_path)
    task = ep["task"].replace(" ","_").replace("/","_"); epn = ep["ep"]
    print(f"\n{'─'*55}\n{ep['task']}  T={ep['T']}  conf={ep['conf'].mean():.3f}")
    print(f"  {ep['desc'][:70]}")

    t0 = time.perf_counter()
    q_seq = retarget_dex3(ep["kp"], algo="angle", smoothing=0.3)
    print(f"  Retargeting (angle-based): {(time.perf_counter()-t0)*1000:.1f}ms")
    print(f"  q range: [{np.degrees(q_seq.min()):+.1f}°, {np.degrees(q_seq.max()):+.1f}°]")
    # Confirm motion
    q_var = np.degrees(q_seq.std(axis=0))
    movers = [(JOINT_NAMES[j], q_var[j]) for j in range(N_JOINTS) if q_var[j] > 1.0]
    print(f"  Moving joints (std > 1°): {[(n, f'{v:.1f}°') for n,v in movers]}")

    t0 = time.perf_counter()
    fk_seq = fk_batch(q_seq)
    print(f"  FK (yourdfpy): {(time.perf_counter()-t0)*1000:.1f}ms")

    out_dir.mkdir(parents=True, exist_ok=True)
    fr = args.frame if args.frame is not None else ep["T"]//2
    plot_static(ep, q_seq, fk_seq, fr,
                out_dir/f"{task}_ep{epn}_dexretarget_frame{fr}.png")
    if not args.static:
        make_animation(ep, q_seq, fk_seq,
                       out_dir/f"{task}_ep{epn}_dexretarget.mp4",
                       fps=args.fps, stride=args.stride)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",        type=Path, default=None)
    p.add_argument("--out_dir",     type=Path, default=Path("outputs"))
    p.add_argument("--static",      action="store_true")
    p.add_argument("--frame",       type=int,  default=None)
    p.add_argument("--fps",         type=int,  default=30)
    p.add_argument("--stride",      type=int,  default=1)
    p.add_argument("--all_samples", action="store_true")
    args = p.parse_args()

    if args.all_samples:
        files = sorted((REPO_ROOT/"samples/egodex").rglob("*.hdf5"))
    elif args.hdf5:
        files = [args.hdf5]
    else:
        files = [REPO_ROOT/"samples/egodex/lock_unlock_key/9.hdf5"]

    print(f"Processing {len(files)} episodes → {args.out_dir}/")
    for f in files:
        run_one(f, args.out_dir, args)
    print("\nDone.")

if __name__ == "__main__":
    main()
