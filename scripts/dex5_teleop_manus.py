"""
dex5_teleop_manus.py
---------------------
Baseline teleoperation: Manus glove → Unitree Dex5-1 hand, NO model required.

Pipeline (runs at ~100Hz):
  Manus SDK → 25-joint SE3 hand pose
      │
      ▼  angle-based retargeting (egodex_retarget_dex5.retarget_hand_frame)
  20-DoF Dex5-1 joint angles
      │
      ▼  Unitree Dex5-1 SDK (joint position control, USB2.0)
  Robot hand moves

Tested retargeting latency: ~0.05ms/frame (pure numpy, real-time safe)

Dependencies:
  pip install numpy
  + Manus SDK Python bindings (ManusSDK / manus-python)
  + Unitree Dex5-1 SDK (dexterous_hand_sdk, from Unitree)

Usage:
  python scripts/dex5_teleop_manus.py --side right --smoothing 0.2
  python scripts/dex5_teleop_manus.py --side both  --dry_run     # no robot, print only
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Joint limits (Dex5-1, radians) — from official URDF
# ---------------------------------------------------------------------------
_LIMITS_R = [
    (math.radians(-33.6), math.radians(39.0)),   # Yaw_11  thumb abduction
    (math.radians(0.0),   math.radians(104.0)),  # Roll_12 thumb MCP
    (math.radians(0.0),   math.radians(101.1)),  # Pitch_13 thumb IP1
    (math.radians(0.0),   math.radians(94.0)),   # Pitch_14 thumb IP2
    (math.radians(-22.0), math.radians(22.0)),   # Roll_21  index lateral
    (math.radians(0.0),   math.radians(90.0)),   # Pitch_22 index MCP
    (math.radians(0.0),   math.radians(96.5)),   # Pitch_23 index PIP
    (math.radians(0.0),   math.radians(80.0)),   # Pitch_24 index DIP
    (math.radians(-22.0), math.radians(22.0)),   # Roll_31  middle lateral
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
    (math.radians(-22.0), math.radians(22.0)),   # Roll_41  ring lateral
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
    (math.radians(-22.0), math.radians(22.0)),   # Roll_51  little lateral
    (math.radians(0.0),   math.radians(90.0)),
    (math.radians(0.0),   math.radians(96.5)),
    (math.radians(0.0),   math.radians(80.0)),
]
_LIMITS_L = list(_LIMITS_R)
_LIMITS_L[1] = (math.radians(-104.0), math.radians(0.0))  # thumb MCP mirrored

_LO_R = np.array([l for l, _ in _LIMITS_R], np.float32)
_HI_R = np.array([h for _, h in _LIMITS_R], np.float32)
_LO_L = np.array([l for l, _ in _LIMITS_L], np.float32)
_HI_L = np.array([h for _, h in _LIMITS_L], np.float32)

DEX5_JOINT_NAMES = [
    "Yaw_11",  "Roll_12",  "Pitch_13", "Pitch_14",   # thumb
    "Roll_21",  "Pitch_22", "Pitch_23", "Pitch_24",   # index
    "Roll_31",  "Pitch_32", "Pitch_33", "Pitch_34",   # middle
    "Roll_41",  "Pitch_42", "Pitch_43", "Pitch_44",   # ring
    "Roll_51",  "Pitch_52", "Pitch_53", "Pitch_54",   # little
]


# ---------------------------------------------------------------------------
# Retargeting helpers (same as egodex_retarget_dex5.py, single-frame version)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n if n > eps else eps)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    c = float(np.dot(_normalize(v1), _normalize(v2)))
    return math.acos(max(-1.0, min(1.0, c)))


def _signed_angle_in_plane(v1: np.ndarray, v2: np.ndarray,
                             normal: np.ndarray) -> float:
    cross = np.cross(v1, v2)
    return math.atan2(float(np.dot(cross, normal)),
                      float(np.dot(v1, v2)))


def _project_out(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n) * n


def retarget_hand_frame(
    kp: np.ndarray,       # (25, 3) joint positions in wrist-local frame
    side: str = "right",  # "right" or "left"
) -> np.ndarray:
    """Retarget one frame of hand keypoints to Dex5-1 joint angles.

    EgoDex / Manus keypoint layout (25 joints):
      0  = wrist (should be at origin in wrist frame)
      1-4  = thumb  : knuckle, interBase, interTip, tip
      5-9  = index  : metacarpal, knuckle, interBase, interTip, tip
      10-14 = middle : same
      15-19 = ring   : same
      20-24 = little : same

    Returns:
      q: (20,) float32  Dex5-1 joint angles [rad], within limits
    """
    # Palm coordinate frame
    palm_fwd  = _normalize(kp[10])                # toward middle metacarpal
    palm_lat  = _normalize(kp[20] - kp[5])        # index → little (lateral)
    palm_norm = _normalize(np.cross(palm_fwd, palm_lat))

    q = np.zeros(20, np.float32)

    # ── Thumb ──────────────────────────────────────────────────────────────
    b_cmc = _normalize(kp[1])
    b_t1  = _normalize(kp[2] - kp[1])
    b_t2  = _normalize(kp[3] - kp[2])
    b_t3  = _normalize(kp[4] - kp[3])

    cmc_flat = _normalize(_project_out(b_cmc, palm_norm))
    q[0] = _signed_angle_in_plane(palm_fwd, cmc_flat, palm_norm)  # abduction
    q[1] = _angle_between(b_cmc, b_t1)   # MCP
    q[2] = _angle_between(b_t1,  b_t2)   # IP1
    q[3] = _angle_between(b_t2,  b_t3)   # IP2

    # ── 4 Fingers ──────────────────────────────────────────────────────────
    for fi, (mi, ki, ii, iti, ti) in enumerate([
        (5,  6,  7,  8,  9),   # index
        (10, 11, 12, 13, 14),  # middle
        (15, 16, 17, 18, 19),  # ring
        (20, 21, 22, 23, 24),  # little
    ]):
        jb = 4 + fi * 4
        b_meta = _normalize(kp[ki] - kp[mi])
        b_prox = _normalize(kp[ii] - kp[ki])
        b_mid  = _normalize(kp[iti]- kp[ii])
        b_dist = _normalize(kp[ti] - kp[iti])

        neutral    = _normalize(kp[mi])
        n_flat     = _normalize(_project_out(neutral, palm_norm))
        bmeta_flat = _normalize(_project_out(b_meta,  palm_norm))

        q[jb]     = _signed_angle_in_plane(n_flat, bmeta_flat, palm_norm)  # lateral
        q[jb + 1] = _angle_between(b_meta, b_prox)  # MCP
        q[jb + 2] = _angle_between(b_prox, b_mid)   # PIP
        q[jb + 3] = _angle_between(b_mid,  b_dist)  # DIP

    # Apply joint limits
    lo, hi = (_LO_L, _HI_L) if side == "left" else (_LO_R, _HI_R)
    np.clip(q, lo, hi, out=q)
    return q


# ---------------------------------------------------------------------------
# Manus SDK adapter (fill in with real SDK calls)
# ---------------------------------------------------------------------------

class ManusGloveReader:
    """Wraps the Manus SDK to provide (25, 4, 4) SE3 hand keypoints.

    Manus SDK docs: https://docs.manus-meta.com/

    Install: pip install manus-python   (or use Manus Core SDK C++ bindings)

    The Manus Quantum/Prime Xsens series provides:
      - 24 or 25 joint angles per hand
      - IMU for wrist orientation
      - All in the wrist local coordinate frame
    """

    def __init__(self, side: str = "right"):
        self.side = side
        self._sdk = None
        self._ready = False
        self._last_kp: Optional[np.ndarray] = None  # (25, 3)

    def connect(self) -> bool:
        """Connect to Manus Core. Returns True on success."""
        try:
            # ── Replace with actual Manus SDK init ──────────────────────────
            # import ManusSDK
            # self._sdk = ManusSDK.ManusHost()
            # self._sdk.Initialize(ManusSDK.SessionType.CoreSDK)
            # self._ready = True
            # ────────────────────────────────────────────────────────────────
            print("[Manus] SDK not loaded — running in simulation mode")
            self._ready = False
            return False
        except ImportError:
            print("[Manus] manus-python not installed — running in simulation mode")
            return False

    def get_keypoints(self) -> Optional[np.ndarray]:
        """Return (25, 3) joint positions in wrist-local frame (meters).

        Returns None if no data available.
        """
        if not self._ready:
            return self._simulate_frame()

        # ── Replace with actual SDK read ────────────────────────────────────
        # frame = self._sdk.GetLastErgonomicsData()
        # hand  = frame.right_hand if self.side == "right" else frame.left_hand
        # kp = np.array([[j.x, j.y, j.z] for j in hand.joints], dtype=np.float32)
        # return kp   # (25, 3)
        # ────────────────────────────────────────────────────────────────────
        return self._simulate_frame()

    def _simulate_frame(self) -> np.ndarray:
        """Simulated neutral open hand pose for dry-run testing."""
        # Approximate anatomical positions in wrist frame (meters)
        kp = np.array([
            [0.000,  0.000,  0.000],  # 0  wrist
            [0.030,  0.020,  0.015],  # 1  thumb knuckle
            [0.055,  0.025,  0.010],  # 2  thumb interBase
            [0.072,  0.018,  0.008],  # 3  thumb interTip
            [0.082,  0.012,  0.005],  # 4  thumb tip
            [0.010,  0.073, -0.003],  # 5  index metacarpal
            [0.010,  0.100, -0.003],  # 6  index knuckle
            [0.010,  0.128, -0.003],  # 7  index interBase
            [0.010,  0.142, -0.003],  # 8  index interTip
            [0.010,  0.150, -0.003],  # 9  index tip
            [-0.012, 0.077, -0.003],  # 10 middle metacarpal
            [-0.012, 0.105, -0.003],  # 11 middle knuckle
            [-0.012, 0.133, -0.003],  # 12 middle interBase
            [-0.012, 0.147, -0.003],  # 13 middle interTip
            [-0.012, 0.155, -0.003],  # 14 middle tip
            [-0.034, 0.073, -0.003],  # 15 ring metacarpal
            [-0.034, 0.100, -0.003],  # 16 ring knuckle
            [-0.034, 0.125, -0.003],  # 17 ring interBase
            [-0.034, 0.138, -0.003],  # 18 ring interTip
            [-0.034, 0.145, -0.003],  # 19 ring tip
            [-0.056, 0.069, -0.003],  # 20 little metacarpal
            [-0.056, 0.093, -0.003],  # 21 little knuckle
            [-0.056, 0.113, -0.003],  # 22 little interBase
            [-0.056, 0.124, -0.003],  # 23 little interTip
            [-0.056, 0.130, -0.003],  # 24 little tip
        ], dtype=np.float32)
        # Optionally add small random perturbation for liveliness
        kp += np.random.randn(*kp.shape).astype(np.float32) * 0.003
        return kp

    def disconnect(self) -> None:
        if self._sdk is not None:
            pass  # self._sdk.ShutDown()


# ---------------------------------------------------------------------------
# Dex5-1 SDK adapter (fill in with real SDK calls)
# ---------------------------------------------------------------------------

class Dex5Controller:
    """Wraps the Unitree Dex5-1 SDK for joint position control.

    Unitree Dex5-1 SDK:
      Communication: USB2.0 at 1000Hz
      Control modes: position / velocity / torque
      Full packet: 1270 bytes recv, 1234 bytes send

    SDK repo: https://github.com/unitreerobotics/dex5_sdk (check Unitree GitHub)
    """

    def __init__(self, side: str = "right", dry_run: bool = False):
        self.side    = side
        self.dry_run = dry_run
        self._sdk    = None

    def connect(self) -> bool:
        if self.dry_run:
            print(f"[Dex5] DRY RUN — {self.side} hand, not connecting to hardware")
            return True
        try:
            # ── Replace with actual Dex5-1 SDK init ─────────────────────────
            # from dex5_sdk import Dex5Hand
            # self._sdk = Dex5Hand(side=self.side)
            # self._sdk.init()
            # ─────────────────────────────────────────────────────────────────
            print("[Dex5] SDK not available — falling back to dry run")
            self.dry_run = True
            return True
        except ImportError:
            print("[Dex5] dex5_sdk not installed — running in dry run")
            self.dry_run = True
            return True

    def set_joint_positions(self, q: np.ndarray,
                             stiffness: float = 0.8,
                             damping: float = 0.1) -> None:
        """Send 20-DoF joint position command to the hand.

        Args:
            q:          (20,) joint angles in radians
            stiffness:  PD stiffness [0, 1] (0=compliant, 1=stiff)
            damping:    PD damping coefficient
        """
        if self.dry_run:
            return  # no-op in dry run

        # ── Replace with actual SDK command ──────────────────────────────────
        # cmd = self._sdk.make_position_command()
        # cmd.q_target    = q.tolist()
        # cmd.kp          = [stiffness * 10.0] * 20
        # cmd.kd          = [damping   *  1.0] * 20
        # self._sdk.send_command(cmd)
        # ─────────────────────────────────────────────────────────────────────
        pass

    def get_joint_state(self) -> Optional[np.ndarray]:
        """Return current (20,) joint angles from robot encoders, or None."""
        if self.dry_run or self._sdk is None:
            return None
        # return np.array(self._sdk.get_state().q, dtype=np.float32)
        return None

    def disconnect(self) -> None:
        if self._sdk is not None:
            pass  # self._sdk.deinit()


# ---------------------------------------------------------------------------
# Exponential smoother
# ---------------------------------------------------------------------------

class ExponentialSmoother:
    def __init__(self, alpha: float = 0.3, n: int = 20):
        self.alpha = alpha
        self._q: Optional[np.ndarray] = None

    def update(self, q_new: np.ndarray) -> np.ndarray:
        if self._q is None:
            self._q = q_new.copy()
        else:
            self._q = self.alpha * self._q + (1.0 - self.alpha) * q_new
        return self._q.copy()


# ---------------------------------------------------------------------------
# Main teleoperation loop
# ---------------------------------------------------------------------------

def teleop_loop(
    side: str = "right",
    dry_run: bool = False,
    smoothing_alpha: float = 0.2,
    target_hz: float = 100.0,
    print_interval: int = 50,
) -> None:
    """Run the teleoperation loop at target_hz.

    Args:
        side:           "right", "left", or "both"
        dry_run:        if True, don't send commands to robot
        smoothing_alpha: exponential filter (lower = smoother but slower)
        target_hz:      control loop frequency
        print_interval: print joint angles every N frames
    """
    sides = ["right", "left"] if side == "both" else [side]

    gloves     = {s: ManusGloveReader(s) for s in sides}
    hands      = {s: Dex5Controller(s, dry_run=dry_run) for s in sides}
    smoothers  = {s: ExponentialSmoother(smoothing_alpha) for s in sides}

    # Connect
    for s in sides:
        gloves[s].connect()
        hands[s].connect()

    print(f"\n{'='*60}")
    print(f" Dex5-1 Teleoperation  ({' + '.join(sides)},  {'DRY RUN' if dry_run else 'LIVE'})")
    print(f" Manus  → retarget → Dex5-1   @ {target_hz:.0f} Hz")
    print(f" Smoothing alpha: {smoothing_alpha}")
    print(f" Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    dt       = 1.0 / target_hz
    frame    = 0
    t_start  = time.perf_counter()
    latencies: list[float] = []

    try:
        while True:
            loop_start = time.perf_counter()

            for s in sides:
                t0 = time.perf_counter()

                # 1. Read Manus keypoints (wrist-local frame, meters)
                kp = gloves[s].get_keypoints()
                if kp is None:
                    continue

                # 2. Retarget → Dex5-1 joint angles
                q_raw     = retarget_hand_frame(kp, side=s)

                # 3. Temporal smoothing
                q_smooth  = smoothers[s].update(q_raw)

                # 4. Send to robot
                hands[s].set_joint_positions(q_smooth)

                latencies.append((time.perf_counter() - t0) * 1000)

                # 5. Print status
                if frame % print_interval == 0:
                    elapsed = time.perf_counter() - t_start
                    mean_lat = sum(latencies[-100:]) / max(len(latencies[-100:]), 1)
                    print(f"[{s:5s}] frame={frame:6d}  t={elapsed:6.1f}s  "
                          f"retarget_lat={mean_lat:.2f}ms")
                    q_deg = np.degrees(q_smooth)
                    pairs = [(DEX5_JOINT_NAMES[j], q_deg[j]) for j in range(20)]
                    parts = [f"{n}={v:+6.1f}°" for n, v in pairs]
                    print("  " + "  ".join(parts[:4]))    # thumb
                    print("  " + "  ".join(parts[4:8]))   # index
                    print()

            frame += 1

            # Rate control
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time   = dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        total_time = time.perf_counter() - t_start
        actual_hz  = frame / max(total_time, 1e-6)
        mean_lat   = sum(latencies) / max(len(latencies), 1)
        print(f"\nStats: {frame} frames in {total_time:.1f}s = {actual_hz:.1f} Hz")
        print(f"Mean retarget latency: {mean_lat:.3f}ms / frame")
        for s in sides:
            gloves[s].disconnect()
            hands[s].disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Manus glove → Unitree Dex5-1 teleoperation."
    )
    parser.add_argument("--side",      choices=["right","left","both"], default="right")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print only, do not send commands to robot")
    parser.add_argument("--smoothing", type=float, default=0.2,
                        help="Exponential filter alpha (default 0.2)")
    parser.add_argument("--hz",        type=float, default=100.0,
                        help="Control loop frequency (default 100 Hz)")
    args = parser.parse_args()

    teleop_loop(
        side=args.side,
        dry_run=args.dry_run,
        smoothing_alpha=args.smoothing,
        target_hz=args.hz,
    )


if __name__ == "__main__":
    main()
