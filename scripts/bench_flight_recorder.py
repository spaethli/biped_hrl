#!/usr/bin/env python3
"""Score a hardware or sim-deploy flight-recorder session into the `[BENCH]` schema
`play.py` prints, so a run on the robot is directly comparable with a sim benchmark.

The point is a shared axis, not a shared file format: a key is emitted under its sim name
ONLY when the hardware quantity is definitionally the same one. Where it is not, the key is
renamed (`err_vx_est`) or omitted with a reason, because `plot_radar.py` normalizes every
axis against a baseline run and cannot tell that one side measured something else.

Two logs per session (a "session pair", CONTEXT.md):

  flight recorder      logs/deploy_safety/<ts>.csv -- the COMMANDED side. raw_q (policy
                       intent), the operator command, measured q/dq, IMU, the passive
                       base-estimator bank, safety triggers. Decimated to ~500 Hz.
  joint telemetry      ~/ramlab_ws/trajectories/all_joints_<ts>*.csv -- clean 50 Hz, and
                       the ONLY source of `tau_est`, the measured joint torque. Without it
                       `mech_power_w` and `cot` cannot be computed at all.

The pair is proposed by overlapping recording intervals (filename time to mtime) and
CONFIRMED by cross-correlating a shared measured joint, because neither filename is a
first-sample time: the flight recorder's is process launch, and logging starts whenever the
RL state is entered (+16 s filename offset vs a -3.3 s signal lag on 2026-08-27, +125 s vs
-10.3 s on 2026-09-07). The two clocks differ in RATE, measured
-2600 to -3067 ppm across three sessions, so the alignment is affine in time: over a 308 s
session a constant offset is wrong by 40 policy steps.

Regimes are never pooled (CONTEXT.md "Regime"). Each session writes BOTH
`<session>.walking.json` and `<session>.standing.json`; pooling them reads a session as 4x
smoother than it is, since a joystick session is 86-95% standing while the sim bench is ~95%
walking.

Usage:
  python scripts/bench_flight_recorder.py logs/deploy_safety/2026-08-27_13-42-55.csv --robot
  python scripts/bench_flight_recorder.py logs/deploy_safety/*.csv --robot --require-torque
  python scripts/bench_flight_recorder.py bridge_run.csv --sim

`--robot` / `--sim` only chooses the output directory (`logs/robot_logs/` vs
`logs/sim_logs/`) and is never inferred from the file, per the deploy convention that a
plant is set explicitly and read back, never guessed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# Reuse, don't duplicate: `load` already handles the meta-json pairing (including the
# hand-renamed `X_meta_A0_fix.json` case) and `policy_steps` is the validated 50 Hz decode.
from score_joint_hold import load, policy_steps, parse_slots  # noqa: E402

LEG_SLOTS = list(range(12))
UB_SLOTS = list(range(12, 27))
KNEE_SLOT = 3          # large-amplitude, present in both logs -- the alignment signal
CTRL_DT = 0.02         # policy step (50 Hz); the recorder's own rows are ~500 Hz
MASS_G = 75.0 * 9.81   # H1-2 ~75 kg, same constant as play.py's CoT
CMD_THRESHOLD = 0.1    # A0 `command_threshold`; gate form matches mdp/rewards.py:157-159
THIN_STEPS = 500       # below this a regime is flagged `thin` (same floor score_joint_hold uses)
TRAJ_DIR = Path.home() / "ramlab_ws" / "trajectories"

# Sim-name keys this tool can emit. Anything not here is either renamed or omitted; the
# omissions and their reasons are in `OMITTED` so `--explain` can print them.
OMITTED = {
    "act_legs": "raw action units; CONTEXT.md 'Physical units rule' -> act_legs_rad",
    "action_rate": "whole-body raw norm; over-weights arms 3-6.7x, avoid in cross-policy claims",
    "jacc": "sim uses MuJoCo qacc at PHYSICS rate; only d(meas_dq) at the non-uniform ~500 Hz log rate exists",
    "jacc_legs": "see jacc; measured 3.1-4.2 at the log rate vs sim 30.3 at the physics rate",
    "jacc_arms": "see jacc; the arm slots have the same rate mismatch",
    "jacc_legs_p95": "see jacc; a p95 of a rate-mismatched series is still rate-mismatched",
    "jacc_arms_p95": "see jacc; a p95 of a rate-mismatched series is still rate-mismatched",
    "qjit_legs": "see jacc, one further derivative of the same rate-mismatched signal",
    "qjit_legs_p95": "see jacc, one further derivative of the same rate-mismatched signal",
    "height_dev": "body_height is in the dead sportmodestate block (identically zero)",
    "gait_match": "needs foot contact state; foot_force_* is dead",
    "fall_rate": "no episode structure on hardware; the per-env episode denominator does not exist",
    "mean_ep_len": "no episode structure",
    "ss_err_vx": "defined against a PINNED command; joystick sessions have none",
    "ss_err_vy": "defined against a pinned command",
    "t90_s": "defined against a pinned command",
    "ss_vx_var": "cross-env variance over N parallel envs; N = 1 robot",
    "mech_power_copper_w": "k*tau^2 is dominated by static holding torque; reads 26-33 vs sim ~0.5",
    "cot_copper": "see mech_power_copper_w; the same static-torque domination in the numerator",
    "payload_kg": "a PRIVILEGED sim extrinsic (env_latent_e); the robot carries no payload record",
    # Stance width (2026-09-07). NOT dead-signal omissions like the rest of this table --
    # `meas_q` carries all 12 leg slots, so `leg_odom.foot_sites_b` would give the walking
    # and standing widths directly. Deferred rather than impossible: it needs the SDK slot
    # order verified against a real session before the number can be trusted, and a wrong
    # width is worse than an absent one. `_td` is the exception and IS impossible here.
    "stance_w_walk": "computable from meas_q via foot_sites_b; DEFERRED pending SDK slot-order check on a real log",
    "stance_w_stand": "computable from meas_q via foot_sites_b; DEFERRED pending SDK slot-order check on a real log",
    "stance_w_td": "needs foot contact state to find touchdown; foot_force_* is dead (same reason as gait_match)",
}


def _recording_interval(path, stamp):
    """[start, end] of a log in wall time: filename timestamp to last write (mtime)."""
    start = pd.to_datetime(stamp, format="%Y-%m-%d_%H-%M-%S")
    return start, pd.Timestamp.fromtimestamp(Path(path).stat().st_mtime)


def find_pair(flight_csv, traj_dir):
    """The joint-telemetry log whose recording interval OVERLAPS this flight session's.

    Neither filename is a first-sample time, so neither can be compared to the other on its
    own. The flight recorder's filename is PROCESS LAUNCH, and logging starts only when the
    RL state is entered, which is operator timing: 2026-09-07_11-44-50 was last written at
    11:49:42 yet holds 156 s of ticks. A "telemetry starts within N s after" rule therefore
    encodes how long someone waited, and broke at +125 s. What is actually true of a real
    pair is that both files were being written at the same time, so the interval from
    filename to mtime must overlap. Largest overlap wins; the cross-correlation still has to
    confirm it. Returns (path, start_offset_s, overlap_s) or None.
    """
    d = Path(traj_dir)
    if not d.is_dir():
        # A typo'd directory globbed to nothing and read as "no pair found" -- which looks
        # like a property of the session rather than a mistake in the command.
        raise SystemExit(f"--traj-dir {traj_dir} does not exist")
    try:
        start_f, end_f = _recording_interval(flight_csv, flight_csv.name[:19])
    except ValueError:
        return None
    best = None
    for p in sorted(d.glob("all_joints_*.csv")):
        try:
            start_a, end_a = _recording_interval(p, p.name[11:30])
        except ValueError:
            continue
        overlap = (min(end_f, end_a) - max(start_f, start_a)).total_seconds()
        if overlap > 0 and (best is None or overlap > best[2]):
            best = (p, (start_a - start_f).total_seconds(), overlap)
    return best


def _xcorr(a, f, dt, maxlag):
    """Peak lag (s) and correlation of f against a, searched within +-maxlag."""
    n = len(a)
    L = int(maxlag / dt)
    if n < 4 or L < 1:
        return np.nan, 0.0
    a = (a - a.mean()) / (a.std() + 1e-12)
    f = (f - f.mean()) / (f.std() + 1e-12)
    cc = np.fft.irfft(np.fft.rfft(f, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)), 2 * n) / n
    cc = np.concatenate([cc[-L:], cc[:L]])
    k = int(np.argmax(cc))
    return (np.arange(-L, L) * dt)[k], float(cc[k])


def coarse_lag(t_f, knee_f, t_a, knee_a, win_s=20.0, min_corr=0.9):
    """Whole-session offset L in  t_flight = t_aj + L, before any rate is fitted.

    Each telemetry window is slid across the ENTIRE flight series with a normalized
    correlation, and the median lag of the windows that match well is taken. Per-window
    matching is what makes this robust: a single global correlation is dominated by the
    highest-variance stretch of either log, and on 2026-09-07 it locked a walking tail onto
    a walking head at -119 s (corr 0.52) against a true -10.3 s. The two logs read the SAME
    encoder, so even a quiet standing window carries shared noise and matches at 0.99.
    Returns None when fewer than two windows agree.
    """
    g = CTRL_DT
    F = np.interp(np.arange(t_f[0], t_f[-1], g), t_f, knee_f)
    ga = np.arange(t_a[0], t_a[-1], g)
    A = np.interp(ga, t_a, knee_a)
    M = int(win_s / g)
    if len(F) <= M or len(A) < M:
        return None
    c1 = np.concatenate(([0.0], np.cumsum(F)))
    c2 = np.concatenate(([0.0], np.cumsum(F * F)))
    mu = (c1[M:] - c1[:-M]) / M
    sd = np.sqrt(np.maximum((c2[M:] - c2[:-M]) / M - mu * mu, 0.0))
    lags = []
    for i in range(0, len(A) - M + 1, M // 2):
        w = A[i:i + M]
        if w.std() < 1e-6:
            continue
        w = (w - w.mean()) / w.std()
        corr = np.correlate(F, w, "valid") / (M * np.maximum(sd, 1e-9))
        corr[sd < 1e-6] = 0.0
        j = int(np.argmax(corr))
        if corr[j] > min_corr:
            lags.append((t_f[0] + j * g) - ga[i])
    return float(np.median(lags)) if len(lags) >= 2 else None


def fit_alignment(t_f, knee_f, t_a, knee_a, maxlag=8.0):
    """Affine time map  t_flight = t_aj + (b + m * t_aj), fitted on windowed cross-correlation.

    A single constant offset is NOT sufficient: the two loggers' clocks differ in rate by a
    measured -2600 to -3067 ppm, which over a 308 s session is 0.8 s -- 40 policy steps. The
    per-window lags are fitted with one robust pass (drop >0.5 s outliers) because a quiet
    standing stretch has a nearly flat autocorrelation and occasionally peaks at a false lag.
    """
    # Coarse first: the fine search below is +-maxlag per window, and the offset between the
    # two loggers' t=0 is operator timing (-3.3..-4.4 s on 2026-08-27, -10.3 s on 2026-09-07,
    # unbounded in principle). Pre-shift by it, fit the residual lag and the rate, compose.
    b0 = coarse_lag(t_f, knee_f, t_a, knee_a) or 0.0
    t_a = t_a + b0
    lo, hi = max(t_f[0], t_a[0]), min(t_f[-1], t_a[-1])
    grid = np.arange(lo, hi, CTRL_DT)
    if len(grid) < 100:
        return None
    A = np.interp(grid, t_a, knee_a)
    F = np.interp(grid, t_f, knee_f)
    win = max(int(min(30.0, (hi - lo) / 3.0) / CTRL_DT), 200)
    ts, ls, cs = [], [], []
    for i in range(0, max(len(grid) - win, 1), max(win // 2, 1)):
        lag, c = _xcorr(A[i:i + win], F[i:i + win], CTRL_DT, maxlag)
        if np.isfinite(lag):
            ts.append(grid[i] + 0.5 * win * CTRL_DT)
            ls.append(lag)
            cs.append(c)
    if not ts:
        return None
    ts, ls, cs = np.array(ts), np.array(ls), np.array(cs)
    good = cs > 0.5
    if good.sum() < 2:
        # too little structure to fit a rate; fall back to the single best window
        k = int(np.argmax(cs))
        return {"b": float(b0 + ls[k]), "m": 0.0, "coarse_lag_s": float(b0), "xcorr": float(cs[k]),
                "resid_ms": float("nan"), "windows": 1, "windows_total": len(ts)}
    ts, ls, cs = ts[good], ls[good], cs[good]
    # Resolve the stride-period alias BEFORE fitting. A walking knee trace is near-periodic,
    # so a window's cross-correlation peaks almost equally at lag, lag+-P, lag+-2P, ... and
    # a single window can lock onto the wrong one (measured on a synthetic 0.6 s gait: true
    # -3.38 s reported as -0.38 s, i.e. +5P, at correlation 0.90). Seed the line from the
    # best-correlated windows, then pull every window onto the nearest alias of that line.
    # Without this a steady-walking session -- exactly the one worth benching -- can fit a
    # confident, wrong rate.
    P = dominant_period(A - A.mean(), CTRL_DT)
    # Seed from the TWO best-correlated windows that are well separated in time. The
    # best-correlated windows are the ones straddling a start/stop transition, where the
    # aperiodic envelope breaks the alias (0.99 there vs 0.90 for an aliased steady-gait
    # window); a top-quartile seed averages those together with aliased ones and locks the
    # whole line one period off. Two anchors are enough because the rest is then unwrapped.
    order = np.argsort(-cs)
    a0 = int(order[0])
    span = ts[-1] - ts[0]
    far = [i for i in order[1:] if abs(ts[i] - ts[a0]) > 0.2 * span]
    if far:
        a1 = int(far[0])
        m = (ls[a1] - ls[a0]) / (ts[a1] - ts[a0])
        b = float(ls[a0] - m * ts[a0])
    else:
        m, b = 0.0, float(ls[a0])
    if np.isfinite(P) and P > 0:
        for _ in range(3):
            k = np.round(((m * ts + b) - ls) / P)
            cand = ls + k * P
            inl = np.abs(cand - (m * ts + b)) < 0.5
            if inl.sum() < 2:
                break
            m, b = np.polyfit(ts[inl], cand[inl], 1)
        ls = ls + np.round(((m * ts + b) - ls) / P) * P
    # Final fit on the WELL-DETERMINED windows only, weighted by correlation, with robust
    # rejection scaled to the actual scatter. A fixed 0.5 s gate is two orders looser than
    # the residuals it is gating (tens of ms): on 13-42-55 the first 60 s carry a genuine
    # non-affine excursion (+0.30 s lag at corr 0.996, back to 0.00 ten seconds later --
    # 15000 ppm, which no clock drifts at, so a discrete event), and inside a 0.5 s gate it
    # bent the whole fit to -1648 ppm at 116 ms. 3 x MAD (floor 50 ms) drops those three
    # windows and recovers -2482 ppm at 18 ms, in line with the other sessions.
    keep = (np.abs(ls - (m * ts + b)) < 0.5) & (cs > 0.7)
    if keep.sum() < 2:
        keep = np.abs(ls - (m * ts + b)) < 0.5
    for _ in range(5):
        if keep.sum() < 3:
            break
        m, b = np.polyfit(ts[keep], ls[keep], 1, w=cs[keep])
        r = ls - (m * ts + b)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        refit = (np.abs(r) < max(0.05, 3.0 * mad)) & (cs > 0.7)
        if refit.sum() < 3 or np.array_equal(refit, keep):
            break
        keep = refit
    if keep.sum() < 2:
        keep = np.ones(len(ls), bool)
    resid = ls[keep] - (m * ts[keep] + b)
    # t_flight = (t_aj + b0) + b + m (t_aj + b0)  =  t_aj + (b0 + b + m b0) + m t_aj
    b, m = b0 + b + m * b0, m
    return {"b": float(b), "m": float(m), "coarse_lag_s": float(b0),
            "xcorr": float(np.median(cs[keep])),
            "resid_ms": float(resid.std() * 1000.0), "windows": int(keep.sum()),
            "windows_total": int(len(ts))}


def projected_gravity_xy(qw, qx, qy, qz):
    """||proj_gravity_b[:2]||, the same quantity play.py reads off `robot.projected_gravity_b`."""
    gx = 2.0 * (qx * qz - qw * qy)
    gy = 2.0 * (qy * qz + qw * qx)
    return np.hypot(gx, gy)


def dominant_period(sig, dt, lo=0.30, hi=1.20):
    """Achieved oscillation period by autocorrelation, parabolically refined.

    The realized counterpart of the commanded gait clock: sim's `stride_period_s` counts
    same-foot touchdowns, which needs contact state, and every contact column in both logs
    is dead. A leg joint's dominant period IS the same-foot interval, and it reproduces sim
    to 0.2-1.5% with left/right agreeing to 0.5%.
    """
    x = np.asarray(sig, float)
    x = x - x.mean()
    n = len(x)
    if n < int(2 * hi / dt) or x.std() < 1e-9:
        return float("nan")
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    a, b = int(lo / dt), min(int(hi / dt), n - 1)
    if b <= a:
        return float("nan")
    k = a + int(np.argmax(ac[a:b]))
    if 0 < k < n - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            k = k + (y0 - y2) / (2 * den)
    return float(k * dt)


def joint_ranges(steps, joints, slots):
    """p1/p50/p99/span/headroom/pinned per joint, on POST-clip commanded targets.

    Same shape as play.py's `--check-joint-limits` block so sim and hardware compare
    directly. `headroom` is the diagnostic: ~0 with pinned > 0 is a clip truncating an
    overshoot, ~1 action-sigma is a penalty that pushed the policy off the bound (ADR-0009).
    Bounds come from the session's OWN meta json, which is written at 4 decimals -- the same
    precision the CSV is written at, so a clipped value compares exactly.
    """
    out = {}
    for s in slots:
        j = joints[s]
        x = steps[:, s]
        p1, p50, p99 = np.percentile(x, [1, 50, 99])
        pin = float(((np.abs(x - j["min"]) < 1e-9) | (np.abs(x - j["max"]) < 1e-9)).mean())
        out[j["name"]] = {
            "p1": round(float(p1), 4), "p50": round(float(p50), 4), "p99": round(float(p99), 4),
            "span": round(float(p99 - p1), 4), "pinned": round(pin, 5),
            "headroom": round(float(min(j["max"] - p99, p1 - j["min"])), 4),
        }
    return out


def pinned_any(steps, joints, slots):
    """Fraction of policy steps with ANY of `slots` sitting on a bound."""
    lo = np.array([joints[s]["min"] for s in slots])
    hi = np.array([joints[s]["max"] for s in slots])
    x = steps[:, slots]
    return float((((np.abs(x - lo) < 1e-9) | (np.abs(x - hi) < 1e-9)).any(axis=1)).mean())


DEFAULT_POSE_YAML = Path(__file__).resolve().parents[1] / (
    "deploy/robots/h1_2/config/policy/velocity/v0/params/deploy_real.yaml")


def default_joint_pos(yaml_path=None):
    """The nominal pose `ub_pose_dev` is a deviation FROM.

    It is not in the flight recorder's meta json, and defaulting the missing value to zero
    silently scores the upper body against a pose the robot never holds -- measured, that
    reads ub_pose_dev 147x sim instead of ~1x, because the nominal shoulder pitch is 0.28
    and the elbow 0.52. Read from the deploy yaml the C++ actually loads; all eight configs
    carry the same pose, so any of them is authoritative. Refuses rather than guessing.
    """
    import yaml
    p = Path(yaml_path) if yaml_path else DEFAULT_POSE_YAML
    cfg = yaml.safe_load(p.read_text())
    if "default_joint_pos" not in cfg:
        raise SystemExit(f"{p}: no default_joint_pos; ub_pose_dev cannot be scored")
    return np.array(cfg["default_joint_pos"], float)


def ref_columns(vel_ref):
    """(x, y) column names for a velocity reference.

    `ach_v` breaks the `<prefix>_<axis>` pattern the estimator bank follows -- the columns
    are `ach_vx`/`ach_vy`, not `ach_v_x`/`ach_v_y`. Only the sim branch reaches that name
    (on hardware `ach_v*` is dead), so a naive suffix concat is a defect that hides until
    the tool is pointed at a sim-bridge log.
    """
    if vel_ref == "ach_v":
        return "ach_vx", "ach_vy"
    return f"{vel_ref}_x", f"{vel_ref}_y"


def walking_mask(cmd):
    """Rows in the WALKING regime, as A0's `command_threshold` defines it.

    `total_command` EXACTLY as mdp/rewards.py:157-159 builds it -- linear L2 PLUS |yaw|,
    which is neither a 3-vector norm nor the 1e-6 stillness gate score_joint_hold uses for
    a different question. The three disagree by up to 2x in walking-step count on the
    2026-08-27 sessions, and pooling or mis-gating reads a session ~4x smoother than it is.
    Independently corroborated: the commanded gait phase is live on 99.6% of the rows this
    gate calls walking.
    """
    cmd = np.asarray(cmd, float)
    total_command = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
    return total_command > CMD_THRESHOLD


def read_flight(csv_path, assume_hold, pose_yaml=None):
    """Flight recorder -> the arrays every metric is built from, on the POLICY-step grid."""
    df, hold, hold_src = load(csv_path, assume_hold)
    meta_p = Path(csv_path).with_name(Path(csv_path).stem + "_meta.json")
    if not meta_p.exists():
        sib = sorted(Path(csv_path).parent.glob(f"{Path(csv_path).name[:19]}*meta*.json"))
        meta_p = sib[0] if len(sib) == 1 else meta_p
    meta = json.loads(meta_p.read_text())
    joints = {j["slot"]: j for j in meta["joints"]}

    # The recorder's schema grew over time (the operator command block, the estimator bank
    # and the gait phase all postdate the earliest captures). Say which columns are missing
    # rather than dying in pandas three frames deeper: an old log is a legitimate input to
    # point this at, and "KeyError: Index([...])" does not tell you the log is simply older.
    need = (["t", "alpha", "cmd_vx", "cmd_vy", "cmd_wz", "quat_w", "quat_x", "quat_y",
             "quat_z", "phase_sin", "phase_cos", "ach_vx", "ach_vy",
             "est_gyro_x", "est_gyro_y", "est_gyro_z"]
            + [f"{p_}{j}" for p_ in ("raw_q", "meas_q", "meas_dq") for j in range(27)])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{Path(csv_path).name}: flight recorder predates {len(missing)} column(s) this "
            f"tool needs, e.g. {missing[:6]} -- it cannot be scored into the BENCH schema")

    raw = df[[f"raw_q{j}" for j in range(27)]].to_numpy(float)
    steps, sidx = policy_steps(raw)
    t = df["t"].to_numpy(float)
    cmd = df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float)
    walk_rows = walking_mask(cmd)
    return {
        "df": df, "meta": meta, "joints": joints, "hold": hold, "hold_src": hold_src,
        "raw": raw, "steps": steps, "sidx": sidx, "t": t, "cmd": cmd,
        "walk_rows": walk_rows, "default_pose": default_joint_pos(pose_yaml),
        "step_hz": float(len(steps) / (t[-1] - t[0])) if t[-1] > t[0] else 0.0,
    }


def read_telemetry(path):
    """Joint telemetry log -> time, torque, joint velocity, knee. Nothing else is alive."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    tau = np.stack([d[f"tau_est{i}"] for i in range(27)], axis=1)
    dq = np.stack([d[f"dq{i}"] for i in range(27)], axis=1)
    return {"t": d["time"], "tau": tau, "dq": dq, "knee": d[f"q{KNEE_SLOT}"]}


def score_regime(F, T, align, regime, vel_ref):
    """One regime of one session -> the metric dict that becomes a BENCH json."""
    df, joints, steps, sidx, t = F["df"], F["joints"], F["steps"], F["sidx"], F["t"]
    walk = F["walk_rows"]
    in_reg = walk[sidx] if regime == "walking" else ~walk[sidx]
    n_steps = int(in_reg.sum())
    # A transition counts only when BOTH endpoints were issued in this regime; differencing
    # across a regime gap turns every re-entry into a spurious large step.
    tsel = in_reg[:-1] & in_reg[1:]
    out = {"regime": regime, "n_steps": n_steps,
           "duration_s": round(n_steps * CTRL_DT, 2), "thin": bool(n_steps < THIN_STEPS)}
    if n_steps < 20 or tsel.sum() < 10:
        out["empty"] = True
        return out

    S = steps[in_reg]
    d = np.diff(steps, axis=0)[tsel]
    out["act_legs_rad"] = float(np.linalg.norm(d[:, LEG_SLOTS], axis=1).mean())

    # Action jitter: third backward difference of the commanded target. q_des = q_def +
    # kappa*a and q_def is constant, so raw_q's third difference IS play.py's, with no
    # kappa needed -- the one key that needs no constant from the training side at all.
    q4 = in_reg[3:] & in_reg[2:-1] & in_reg[1:-2] & in_reg[:-3]
    if q4.sum() > 10:
        d3 = (steps[3:] - 3.0 * steps[2:-1] + 3.0 * steps[1:-2] - steps[:-3]) / CTRL_DT ** 3
        aj = np.linalg.norm(d3[q4][:, LEG_SLOTS], axis=1)
        out["ajit_legs"] = float(aj.mean())
        out["ajit_legs_p95"] = float(np.percentile(aj, 95))

    rows = walk if regime == "walking" else ~walk
    q = df[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(float)[rows]
    out["orient_dev"] = float(projected_gravity_xy(*q.T).mean())
    gyro = df[["est_gyro_x", "est_gyro_y", "est_gyro_z"]].to_numpy(float)[rows]
    out["omega_xy"] = float(np.linalg.norm(gyro[:, :2], axis=1).mean())
    # err_yaw is exempt from the estimator question: gyro z is a DIRECT measurement, the
    # same quantity sim reads off root_link_ang_vel_b[:, 2].
    out["err_yaw"] = float(np.abs(F["cmd"][rows, 2] - gyro[:, 2]).mean())

    ref = df[list(ref_columns(vel_ref))].to_numpy(float)[rows]
    suffix = "" if vel_ref == "ach_v" else "_est"
    out["err_vx" + suffix] = float(np.abs(F["cmd"][rows, 0] - ref[:, 0]).mean())
    out["err_vy" + suffix] = float(np.abs(F["cmd"][rows, 1] - ref[:, 1]).mean())
    out["err_v_reference"] = vel_ref

    mq = df[[f"meas_q{j}" for j in range(27)]].to_numpy(float)[rows]
    mdq = df[[f"meas_dq{j}" for j in range(27)]].to_numpy(float)[rows]
    dflt = F["default_pose"]
    out["ub_pose_dev"] = float(((mq[:, UB_SLOTS] - dflt[UB_SLOTS]) ** 2).mean(axis=1).mean())
    out["ub_arm_vel"] = float(np.abs(mdq[:, UB_SLOTS]).mean())

    ank = {"ank_roll_span_l": "left_ankle_roll", "ank_roll_span_r": "right_ankle_roll",
           "ank_pitch_span_l": "left_ankle_pitch", "ank_pitch_span_r": "right_ankle_pitch"}
    jr = joint_ranges(S, joints, LEG_SLOTS)
    for k, name in ank.items():
        out[k] = jr[name]["span"]
    out["joint_range"] = jr
    out["pinned_any_leg"] = round(pinned_any(S, joints, LEG_SLOTS), 5)

    if regime == "walking":
        out.update(_cadence(df, t, rows))
        if T is not None and align is not None:
            out.update(_energy(F, T, align, vel_ref))
    return out


def _cadence(df, t, rows):
    """Commanded gait clock (from the logged phase) + the ACHIEVED period (from kinematics).

    The phase is the clock the low level is entrained to, so it is directly observable; for
    A0 it recovers the fixed 0.6 s reference and for A1a it is the HL's commanded period.
    `stride_period_s` keeps sim's name because it is sim's quantity -- realized, not
    commanded -- just recovered from leg kinematics rather than from footfalls.
    """
    ps, pc = df["phase_sin"].to_numpy(float), df["phase_cos"].to_numpy(float)
    live = ((np.abs(ps) + np.abs(pc)) > 1e-6) & rows
    idx = np.flatnonzero(live)
    if len(idx) < 600:
        return {}
    blocks = [b for b in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if len(b) >= 600]
    cmd_p, per_l, per_r, w = [], [], [], []
    kL = df[f"meas_q{KNEE_SLOT}"].to_numpy(float)
    kR = df[f"meas_q{KNEE_SLOT + 6}"].to_numpy(float)
    for b in blocks:
        ang = np.unwrap(np.arctan2(ps[b], pc[b]))
        dur, dang = t[b][-1] - t[b][0], ang[-1] - ang[0]
        if abs(dang) < 1e-6 or dur <= 0:
            continue
        # resample the block onto a uniform grid: the log is decimated, and autocorrelation
        # on non-uniform samples reads a period that is a function of the logging, not the gait
        gr = np.arange(t[b][0], t[b][-1], 0.002)
        cmd_p.append(2 * np.pi * dur / abs(dang))
        per_l.append(dominant_period(np.interp(gr, t[b], kL[b]), 0.002))
        per_r.append(dominant_period(np.interp(gr, t[b], kR[b]), 0.002))
        w.append(len(b))
    if not w:
        return {}
    w = np.array(w, float)
    w /= w.sum()

    def wmean(v):
        v = np.array(v, float)
        m = np.isfinite(v)
        return float(np.dot(v[m], w[m] / w[m].sum())) if m.any() else float("nan")

    pl, pr = wmean(per_l), wmean(per_r)
    return {"cadence_period_cmd_s": wmean(cmd_p),
            "stride_period_s": float(np.nanmean([pl, pr])),
            "stride_period_lr": [pl, pr], "cadence_blocks": int(len(w))}


def _energy(F, T, align, vel_ref):
    """mech_power_w and cot from the MEASURED torque, gated the way play.py gates them.

    play.py's CoT gate is on the commanded LINEAR speed only (`cmd[:, :2].norm > 0.1`), not
    on the walking regime's linear+yaw form, so it is reproduced exactly here rather than
    reusing the regime mask. The distance denominator integrates the velocity ESTIMATE,
    since no ground-truth odometry survives on hardware -- recorded in `cot_distance_reference`.
    """
    ta = T["t"]
    t_in_flight = ta + (align["b"] + align["m"] * ta)
    tf = F["t"]
    lin = np.linalg.norm(F["cmd"][:, :2], axis=1)
    # np.interp CLAMPS outside its range, so telemetry samples mapping past either end of the
    # Run inherit its first/last gate and speed held constant. A Run that ENDS while walking
    # then earns distance for every trailing telemetry sample (gate held at 1) and dilutes
    # power[gate].mean() with samples from after it stopped. Latent while a flight recorder
    # file spanned a whole controller process; it bites under Run-scoped scoring (ADR-0012),
    # where a 186 s Run pairs with a 232 s telemetry log. Measured on 2026-09-14_14-48-55:
    # 45.12 m against an independently computed 11.5 m, with mech_power_w reading the lowest
    # of the session at 79 W. Runs that end standing hold gate=0 and were never affected,
    # which is why only one run in eleven looked wrong.
    inside = (t_in_flight >= tf[0]) & (t_in_flight <= tf[-1])
    gate = (np.interp(t_in_flight, tf, (lin > CMD_THRESHOLD).astype(float)) > 0.5) & inside
    spd = np.interp(t_in_flight, tf,
                    np.linalg.norm(F["df"][list(ref_columns(vel_ref))].to_numpy(float), axis=1))
    if gate.sum() < 50:
        return {}
    power = np.abs(T["tau"] * T["dq"]).sum(axis=1)
    dt = np.diff(ta, prepend=ta[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.02
    dist = float((spd * gate * dt).sum())
    out = {"mech_power_w": float(power[gate].mean()),
           "cot_distance_reference": vel_ref, "cot_distance_m": round(dist, 2)}
    if dist > 0.1:
        out["cot"] = float((power * gate * dt).sum() / (dist * MASS_G))
    return out


def command_block(F):
    """What the operator actually did. A joystick session is not command-matched to any
    other session, so this is the provenance that makes two hardware points comparable (or
    visibly not): 2026-08-27's three sessions ran median cmd_vx 0.220 / 0.363 / 0.000."""
    c, walk = F["cmd"], F["walk_rows"]
    vx = c[walk, 0]
    if not walk.any():
        return {"walk_frac": 0.0}
    return {"walk_frac": round(float(walk.mean()), 4),
            "cmd_vx_mean": round(float(vx.mean()), 4),
            "cmd_vx_med": round(float(np.median(vx)), 4),
            "cmd_vx_p10": round(float(np.percentile(vx, 10)), 4),
            "cmd_vx_p90": round(float(np.percentile(vx, 90)), 4),
            "cmd_vy_abs_mean": round(float(np.abs(c[walk, 1]).mean()), 4),
            "cmd_wz_abs_mean": round(float(np.abs(c[walk, 2]).mean()), 4)}


def process(csv_path, args):
    csv_path = Path(csv_path)
    print(f"\n=== {csv_path.name} ===")
    F = read_flight(csv_path, args.assume_hold, args.default_pose_yaml)
    flag = "" if 40.0 <= F["step_hz"] <= 60.0 else "   <-- STEP RATE OFF, every rate below is meaningless"
    print(f"  policy steps {len(F['steps'])} at {F['step_hz']:.2f} Hz{flag}")
    if F["hold_src"] == "--assume-hold" and not F["hold"]:
        print("  WARN meta json has no hold_joint_ids; assuming a FREE upper body. "
              "ub_pose_dev/ub_arm_vel measure the policy only if that is right (--assume-hold to override)")

    # `ach_v*` is written from SportModeState, which is alive in sim and dead on the robot.
    # The reference CLASS decides the key name: ground truth keeps sim's `err_vx`, an
    # estimate gets `_est` (CONTEXT.md, "Fused base velocity").
    ach = F["df"][["ach_vx", "ach_vy"]].to_numpy(float)
    vel_ref = "ach_v" if np.any(ach != 0.0) else args.vel_reference
    print(f"  velocity reference: {vel_ref}" + ("  (ground truth)" if vel_ref == "ach_v"
          else "  (an ESTIMATE -> keys are err_vx_est/err_vy_est)"))

    T = align = None
    pair = find_pair(csv_path, args.traj_dir)
    if pair is None:
        msg = (f"no joint-telemetry log in {args.traj_dir} was being written while "
               f"{csv_path.name} was (filename time to mtime)")
        if args.require_torque:
            raise SystemExit(f"  ERROR {msg} (--require-torque)")
        print(f"  WARN {msg}\n       -> omitting mech_power_w, cot (both need tau_est)")
    else:
        p, dt, overlap = pair
        T = read_telemetry(p)
        align = fit_alignment(F["t"], F["df"][f"meas_q{KNEE_SLOT}"].to_numpy(float),
                              T["t"], T["knee"])
        if align is None or align["xcorr"] < args.xcorr_floor:
            got = "unfittable" if align is None else f"{align['xcorr']:.2f}"
            msg = (f"pair {p.name} ({dt:+.0f}s, {overlap:.0f}s overlap) REJECTED: knee xcorr {got} "
                   f"< floor {args.xcorr_floor:.2f}")
            if args.require_torque:
                raise SystemExit(f"  ERROR {msg} (--require-torque)")
            print(f"  WARN {msg}\n       -> omitting mech_power_w, cot")
            T = align = None
        else:
            print(f"  paired: {p.name}  ({dt:+.0f}s, {overlap:.0f}s overlap)")
            print(f"  xcorr(knee) {align['xcorr']:.2f}  [floor {args.xcorr_floor:.2f}, PASS]"
                  f"   windows {align['windows']}/{align['windows_total']}")
            # 100 ms = 5 policy steps. The alignment only decides which REGIME a torque
            # sample is gated into, so its error costs a handful of samples at each regime
            # boundary out of thousands; it becomes meaningful only as it approaches the
            # length of a regime block. Reported always, flagged above that.
            print(f"  lag(t) = {align['b']:+.3f} {align['m'] * 1e6:+.0f}ppm*t"
                  f"   resid {align['resid_ms']:.0f} ms"
                  f"{'' if align['resid_ms'] < 100.0 else '   <-- HIGH, check the pair'}")
            print("  torque: 27 ch live -> mech_power_w, cot are MEASURED")

    session = {
        "label": csv_path.name[:19],
        "session_note": (pair[0].name[31:-4] or None) if pair else None,
        "source": "flight_recorder",
        "flight_recorder": csv_path.name,
        "all_joints": pair[0].name if (pair and T is not None) else None,
        "hold_joint_ids": F["hold"], "hold_src": F["hold_src"],
        "step_hz": round(F["step_hz"], 3),
        "session_duration_s": round(float(F["t"][-1] - F["t"][0]), 2),
        "command": command_block(F),
        # Whole-session commanded range: the quantity ADR-0009 reports, kept alongside the
        # regime-scoped spans because authority is a session property, not a regime one.
        "session_joint_range": joint_ranges(F["steps"], F["joints"], LEG_SLOTS),
        "session_pinned_any_leg": round(pinned_any(F["steps"], F["joints"], LEG_SLOTS), 5),
    }
    if align is not None:
        session.update({"pair_lag_s": round(align["b"], 4),
                        "pair_coarse_lag_s": round(align["coarse_lag_s"], 3),
                        "pair_skew_ppm": round(align["m"] * 1e6, 1),
                        "pair_xcorr": round(align["xcorr"], 4),
                        "pair_resid_ms": round(align["resid_ms"], 1)})

    sr = session["session_joint_range"]
    print(f"  session ankle roll span L/R  {sr['left_ankle_roll']['span']:.4f} / "
          f"{sr['right_ankle_roll']['span']:.4f}    pinned(any leg) "
          f"{session['session_pinned_any_leg'] * 100:.2f}%")

    # `label` is the csv name's first 19 chars, so several runs sliced out of ONE controller
    # process (different FSM entries, e.g. 2026-09-14_15-00-37 holds three) share a label and
    # would overwrite each other in a shared directory. --out-dir lets a per-run bundle keep
    # its own metrics beside its own logs.
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "logs/robot_logs" if args.robot else "logs/sim_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for regime in ("walking", "standing"):
        m = score_regime(F, T, align, regime, vel_ref)
        if m.get("empty"):
            print(f"  {regime:9s} SKIP: {m['n_steps']} steps, too few to score")
            continue
        if m["thin"]:
            print(f"  WARN {regime}: {m['n_steps']} steps ({m['duration_s']:.1f} s) below the "
                  f"{THIN_STEPS}-step floor; emitted but treat as indicative")
        payload = {**session, **m, "omitted": OMITTED}
        out = out_dir / f"{session['label']}.{regime}.json"
        out.write_text(json.dumps(payload, indent=2))
        written.append(out)
        key = "act_legs_rad"
        print(f"  {regime:9s} {m['n_steps']:6d} steps ({m['duration_s']:7.1f} s)  "
              f"{key} {m.get(key, float('nan')):.5f}  " +
              (f"cot {m['cot']:.4f}  " if "cot" in m else "") +
              (f"stride {m['stride_period_s']:.4f}s  " if "stride_period_s" in m else "") +
              f"-> {out.name}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="flight-recorder csv(s) from logs/deploy_safety/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--robot", action="store_true", help="write to logs/robot_logs/")
    g.add_argument("--sim", action="store_true", help="write to logs/sim_logs/")
    ap.add_argument("--traj-dir", default=str(TRAJ_DIR),
                    help="where the joint-telemetry (all_joints_*.csv) logs live")
    ap.add_argument("--out-dir", default=None,
                    help="write the regime jsons here instead of logs/{robot,sim}_logs/; "
                         "needed when several runs sliced from one controller process share "
                         "a label (see bundle_hardware_run.py)")
    ap.add_argument("--xcorr-floor", type=float, default=0.70,
                    help="reject a proposed pair whose knee cross-correlation is below this")
    ap.add_argument("--require-torque", action="store_true",
                    help="fail instead of degrading when no telemetry pair is accepted")
    ap.add_argument("--vel-reference", default="est_v_compl",
                    help="estimator column prefix used as the tracking reference when ach_v "
                         "is dead (default: est_v_compl, the WL-G arm B verdict)")
    ap.add_argument("--assume-hold", default=None,
                    help="hold slots for logs whose meta json predates the field, e.g. 12-26")
    ap.add_argument("--default-pose-yaml", default=None,
                    help="deploy yaml supplying default_joint_pos for ub_pose_dev "
                         "(default: the A0 deploy_real config; all 8 carry the same pose)")
    ap.add_argument("--explain", action="store_true",
                    help="print the omitted BENCH keys and why, then exit")
    args = ap.parse_args()

    if args.explain:
        print("BENCH keys this tool does NOT emit from a flight recorder:\n")
        for k, why in sorted(OMITTED.items()):
            print(f"  {k:22s} {why}")
        print("\nrenamed: err_vx -> err_vx_est, err_vy -> err_vy_est when the velocity "
              "reference is an estimate (ach_v dead). Ground truth keeps the sim name.")
        return 0

    args.assume_hold = parse_slots(args.assume_hold) if args.assume_hold is not None else []
    written = []
    for c in args.csvs:
        written += process(c, args)
    print(f"\nwrote {len(written)} json file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
