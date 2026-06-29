"""
visualize_retargeted_dex3.py — All 4 Dex3-1 retargeting algorithms visualized.

Layout:  [ Human (AVP) | ANGLE | IK | PCA | LINEAR ]
Each robot panel uses yourdfpy FK for accurate link positions.

Coordinate system (Dex3-1 palm frame):
  X = extension  (fingers point forward along +X, 0→12 cm)
  Y = curl       (+Y when fingers close)
  Z = lateral    (index +29 mm, middle −29 mm)
  → best view: elev=20, azim=180 (looking from fingertips toward palm)

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
import yourdfpy

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.egodex_retarget_dex3 import retarget_dex3, JOINT_NAMES, N_JOINTS, _LO, _HI, ALGORITHMS

# ─── Dex3-1 URDF FK setup ─────────────────────────────────────────────────────
URDF_R = REPO_ROOT / "assets/dex3_1/dex3_1_r.urdf"
# yourdfpy actuated order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
# retarget_dex3 order:    thumb_0, thumb_1, thumb_2, index_0,  index_1,  middle_0, middle_1
REORDER = [0, 1, 2, 5, 6, 3, 4]   # retarget → yourdfpy order
LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link",
    "right_hand_index_0_link", "right_hand_index_1_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
]
# Bone connectivity (indices into LINKS)
BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(0,6),(6,7),(4,6)]

BG = "#0d1117"
ALGO_COLORS = {"angle":"#3498db","ik":"#2ecc71","pca":"#e67e22","linear":"#e74c3c"}
FCOL = {"thumb":"#FF6B6B","index":"#4ECDC4","middle":"#FFD93D"}

def _bcol(p,c):
    if p<=3 or c<=3: return FCOL["thumb"]
    if p<=5 or c<=5: return FCOL["index"]
    return FCOL["middle"]

def build_fk_fn():
    robot = yourdfpy.URDF.load(str(URDF_R))
    jnames = robot.actuated_joint_names
    def fk(q_retarget):
        """q_retarget: (7,) in retarget_dex3 order → (8,3) link positions."""
        q_urdf = q_retarget[REORDER]
        robot.update_cfg({n: float(v) for n,v in zip(jnames, q_urdf)})
        return np.array([robot.get_transform(l)[:3,3] for l in LINKS], dtype=np.float32)
    return fk

def fk_batch(fk_fn, q_seq):
    return np.stack([fk_fn(q_seq[t]) for t in range(len(q_seq))])

# ─── Dex3-1 axis limits (metres) ──────────────────────────────────────────────
DEX3_LIM = {"x":(-0.02,0.15), "y":(-0.08,0.12), "z":(-0.06,0.06)}

# ─── EgoDex 25-joint right hand ───────────────────────────────────────────────
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
    if a<=4 or b<=4:  return FCOL["thumb"]
    if a<=9 or b<=9:  return FCOL["index"]
    if a<=14 or b<=14:return FCOL["middle"]
    return "#555"

def load_ep(path):
    with h5py.File(path) as f:
        kp = np.stack([f[f"transforms/{j}"][:] for j in RIGHT_JOINTS], axis=1)
        conf = f["confidences/rightHand"][:] if "confidences/rightHand" in f else np.ones(kp.shape[0])
        desc = str(f.attrs.get("llm_description") or f.attrs.get("description",""))
        task = str(f.attrs.get("task", path.parent.name)); ep = int(path.stem)
    kp_local = kp[:,:,:3,3] - kp[:,0:1,:3,3]
    return dict(kp=kp, kp_local=kp_local, conf=conf, desc=desc, task=task, ep=ep, T=kp.shape[0])

# ─── Axes helpers ─────────────────────────────────────────────────────────────
def _style_human(ax):
    ax.set_facecolor(BG)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor("none")
    ax.grid(True,color="white",alpha=0.04,lw=0.3)
    ax.tick_params(colors="#444",labelsize=5)

def _style_robot(ax):
    _style_human(ax)
    # Look from fingertips toward palm (−X direction), slight elevation
    ax.view_init(elev=20, azim=180)
    ax.set_xlabel("X ext→",color="#888",fontsize=6)
    ax.set_ylabel("Y curl↑",color="#888",fontsize=6)
    ax.set_zlabel("Z lat",color="#888",fontsize=6)
    ax.set_xlim(*DEX3_LIM["x"])
    ax.set_ylim(*DEX3_LIM["y"])
    ax.set_zlim(*DEX3_LIM["z"])

def _eq_human(ax, pts):
    c=(pts.max(0)+pts.min(0))/2; h=(pts.max(0)-pts.min(0)).max()/2*0.7
    ax.set_xlim(c[0]-h,c[0]+h); ax.set_ylim(c[1]-h,c[1]+h); ax.set_zlim(c[2]-h,c[2]+h)

def draw_human(ax, kp, alpha=0.9):
    for a,b in H_BONES:
        ax.plot(*zip(kp[a],kp[b]),color=_hcol(a,b),lw=1.4,alpha=alpha)
    ax.scatter(*kp[0],s=30,c="white",edgecolors="#888",lw=0.8,zorder=5)
    for i in [4,9,14]: ax.scatter(*kp[i],s=18,c="white",zorder=5)
    # Wrist label
    ax.text(*kp[0], "wrist", color="#aaa", fontsize=5)

def draw_dex3(ax, pts, color, alpha=0.92):
    """Draw Dex3-1 skeleton in palm coordinate frame (metres)."""
    for a,b in BONES:
        lw = 2.5 if (a,b)!=(4,6) else 0.8
        al = alpha if (a,b)!=(4,6) else 0.25
        ax.plot(*zip(pts[a],pts[b]),color=color,lw=lw,alpha=al)
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=22,c=color,edgecolors="white",lw=0.4,zorder=4,alpha=alpha)
    ax.scatter(*pts[0],s=55,c="white",edgecolors="black",lw=1.2,zorder=6)
    # Label key points
    labels={0:"palm",3:"thumb",5:"index",7:"middle"}
    for i,lbl in labels.items():
        ax.text(*pts[i],lbl,color="#aaa",fontsize=5)

def _joint_bar(ax, q_t, title, color):
    ax.set_facecolor(BG)
    r=_HI-_LO; pct=(q_t-_LO)/np.where(r>1e-4,r,1.)
    cols=[FCOL["thumb"]]*3+[FCOL["index"]]*2+[FCOL["middle"]]*2
    ax.barh(range(7),pct,color=cols,edgecolor="none",height=0.65,alpha=0.85)
    ax.barh(range(7),[1]*7,color="none",edgecolor="#444",height=0.65,lw=0.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"{n} {math.degrees(v):+.0f}°" for n,v in zip(JOINT_NAMES,q_t)],
                        fontsize=5.5,color="white")
    ax.set_xlim(0,1.1); ax.set_xticks([0,1])
    ax.set_xticklabels(["lo","hi"],fontsize=5,color="white")
    ax.set_title(title,color=color,fontsize=7,pad=2)
    for s in ax.spines.values(): s.set_edgecolor("#333")

# ─── Compute all 4 retargets + FK ─────────────────────────────────────────────
def compute_all(ep, fk_fn):
    all_q={}; all_fk={}; timings={}
    for algo in ALGORITHMS:
        t0=time.perf_counter()
        q=retarget_dex3(ep["kp"],algo=algo)
        timings[algo]=(time.perf_counter()-t0)*1000
        all_q[algo]=q; all_fk[algo]=fk_batch(fk_fn,q)
        print(f"    {algo:8s} {timings[algo]:5.1f}ms  [{math.degrees(q.min()):+.1f}°,{math.degrees(q.max()):+.1f}°]")
    return all_q,all_fk,timings

# ─── Static plot ──────────────────────────────────────────────────────────────
def plot_static(ep, all_q, all_fk, timings, frame, out_path):
    kp  = ep["kp_local"][frame]
    all_kp = ep["kp_local"].reshape(-1,3)

    fig = plt.figure(figsize=(22,10), facecolor=BG)
    gs  = gridspec.GridSpec(2,5,figure=fig,hspace=0.35,wspace=0.05,height_ratios=[2.5,1])

    # 3D skeletons row
    ax_h = fig.add_subplot(gs[0,0],projection="3d")
    _style_human(ax_h); _eq_human(ax_h,all_kp)
    draw_human(ax_h,kp)
    ax_h.set_title("Human (AVP)", color="white", fontsize=9, pad=5)

    for i,algo in enumerate(ALGORITHMS):
        ax = fig.add_subplot(gs[0,i+1],projection="3d")
        _style_robot(ax)
        pts=all_fk[algo][frame]
        draw_dex3(ax,pts,ALGO_COLORS[algo])
        draw_human(ax,kp * 0.0, alpha=0)  # invisible, just for consistency
        ax.set_title(f"{algo.upper()}\n{timings[algo]:.1f}ms",
                     color=ALGO_COLORS[algo],fontsize=9,pad=5)

    # Joint bar row
    ax_blank=fig.add_subplot(gs[1,0]); ax_blank.axis("off")
    for i,algo in enumerate(ALGORITHMS):
        ax=fig.add_subplot(gs[1,i+1])
        _joint_bar(ax,all_q[algo][frame],f"{algo.upper()} fr{frame}",ALGO_COLORS[algo])

    # Legend
    from matplotlib.lines import Line2D
    leg=[Line2D([0],[0],color=v,lw=2,label=k) for k,v in FCOL.items()]
    fig.legend(handles=leg,loc="lower center",ncol=3,facecolor=BG,edgecolor="none",
               labelcolor="white",fontsize=8,bbox_to_anchor=(0.5,-0.01))

    fig.suptitle(f"[{ep['task']}  ep{ep['ep']}  fr{frame}/{ep['T']-1}]  {ep['desc'][:80]}",
                 color="white",fontsize=9,y=1.005)
    fig.savefig(str(out_path),dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close(fig)
    print(f"  → {out_path}")

# ─── Animation ────────────────────────────────────────────────────────────────
def make_animation(ep, all_q, all_fk, timings, out_path, fps=30, stride=1):
    T=ep["T"]; frames=list(range(0,T,stride))
    all_kp=ep["kp_local"].reshape(-1,3)

    fig=plt.figure(figsize=(22,5),facecolor=BG)
    gs=gridspec.GridSpec(1,5,figure=fig,wspace=0.05)
    ax_h=fig.add_subplot(gs[0,0],projection="3d")
    robot_axs=[fig.add_subplot(gs[0,i+1],projection="3d") for i in range(4)]
    title_obj=fig.suptitle("",color="white",fontsize=8)

    def update(fi):
        t=frames[fi]
        ax_h.cla(); _style_human(ax_h); _eq_human(ax_h,all_kp)
        draw_human(ax_h,ep["kp_local"][t])
        ax_h.set_title("Human (AVP)",color="white",fontsize=8,pad=4)
        for ax,algo in zip(robot_axs,ALGORITHMS):
            ax.cla(); _style_robot(ax)
            draw_dex3(ax,all_fk[algo][t],ALGO_COLORS[algo])
            ax.set_title(f"{algo.upper()} ({timings[algo]:.1f}ms)",
                         color=ALGO_COLORS[algo],fontsize=8,pad=4)
        title_obj.set_text(
            f"[{ep['task']}  ep{ep['ep']}]  fr{t}/{T-1}  {ep['desc'][:60]}")
        return []

    anim=animation.FuncAnimation(fig,update,frames=len(frames),interval=1000//fps,blit=False)
    writer=animation.FFMpegWriter(fps=fps,bitrate=2000,extra_args=["-pix_fmt","yuv420p"])
    out_path.parent.mkdir(parents=True,exist_ok=True)
    print(f"  Rendering {len(frames)} frames …")
    anim.save(str(out_path),writer=writer)
    plt.close(fig)
    print(f"  → {out_path}")

# ─── Run ──────────────────────────────────────────────────────────────────────
def run_one(hdf5_path, out_dir, fk_fn, args):
    ep=load_ep(hdf5_path)
    task=ep["task"].replace(" ","_").replace("/","_"); epn=ep["ep"]
    print(f"\n{'─'*55}\n{ep['task']}  T={ep['T']}  conf={ep['conf'].mean():.3f}")
    print(f"  {ep['desc'][:70]}")
    print("  Retargeting + FK …")
    all_q,all_fk,timings=compute_all(ep,fk_fn)
    fr = args.frame if args.frame is not None else ep["T"]//2
    plot_static(ep,all_q,all_fk,timings,fr,
                out_dir/f"{task}_ep{epn}_all4_frame{fr}.png")
    if not args.static:
        make_animation(ep,all_q,all_fk,timings,
                       out_dir/f"{task}_ep{epn}_all4_retarget.mp4",
                       fps=args.fps,stride=args.stride)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--hdf5",        type=Path, default=None)
    p.add_argument("--out_dir",     type=Path, default=Path("outputs"))
    p.add_argument("--static",      action="store_true")
    p.add_argument("--frame",       type=int,  default=None)
    p.add_argument("--fps",         type=int,  default=30)
    p.add_argument("--stride",      type=int,  default=1)
    p.add_argument("--all_samples", action="store_true")
    args=p.parse_args()
    if args.all_samples:
        files=sorted((REPO_ROOT/"samples/egodex").rglob("*.hdf5"))
    elif args.hdf5:
        files=[args.hdf5]
    else:
        files=[REPO_ROOT/"samples/egodex/lock_unlock_key/9.hdf5"]
    print(f"Loading URDF FK …")
    fk_fn=build_fk_fn()
    print(f"Processing {len(files)} episodes → {args.out_dir}/")
    for f in files:
        run_one(f, args.out_dir, fk_fn, args)
    print("\nDone.")

if __name__=="__main__":
    main()
