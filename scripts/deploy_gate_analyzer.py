"""Deploy-gate analyzer (plan W3). Two input schemas, auto-detected by column presence:

- `<base>_hrl.csv` (State_RLHRL's `hrl::Telemetry`, A1 only): goal-state `s` columns give
  height directly, plus a logged `act_rate`/`period`.
- `<base>.csv` (the shared `SafetyLogger`, both A0 and A1, `cmd_vx.../ach_vx...` columns
  added 2026-07-21 WL-B0): no height or logged act_rate/period, so those are derived
  differently (see `analyze_base_csv`) — this is what makes A0 readable by this tool at
  all, and gives A1's base CSV a from-triggers view independent of `_hrl.csv`. `ach_*` is
  SIM-ONLY ground truth (SportModeState); reads ~0 on real hardware (same gap as E1).

Segments the run by held-command changes and scores each segment: achieved-vs-commanded
velocity (the G2 parity numbers), fall detection, action rate, commanded period (HRL only),
and a stride-period proxy (autocorrelation of the measured left hip pitch — one full hip
cycle = one stride, the same-foot touchdown interval). Prints a table + `[DEPLOY-GATE]
{json}` for the plan's pass criteria.

Usage: python scripts/deploy_gate_analyzer.py logs/deploy_safety/<ts>_hrl.csv   # A1 HRL telemetry
       python scripts/deploy_gate_analyzer.py logs/deploy_safety/<ts>.csv      # A0 or A1 base CSV
"""

import argparse
import csv
import json

import numpy as np

FALL_HEIGHT = 0.9  # imu z below this = fallen (nominal standing 1.3076); _hrl.csv path only
MIN_SEG_S = 3.0  # ignore shorter command segments (transients between key presses)


def stride_proxy(hip: np.ndarray, dt: float) -> float:
  """Dominant hip-pitch cycle period via autocorrelation; NaN if no clear lock."""
  x = hip - hip.mean()
  if len(x) < int(2.0 / dt) or x.std() < 1e-4:
    return float("nan")
  ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
  ac /= ac[0] + 1e-12
  lo, hi = int(0.2 / dt), min(int(1.5 / dt), len(ac) - 1)
  if hi <= lo:
    return float("nan")
  k = lo + int(np.argmax(ac[lo:hi]))
  return k * dt if ac[k] > 0.3 else float("nan")


def _segment_bounds(t: np.ndarray, cmd: np.ndarray, entry: np.ndarray | None) -> list[int]:
  """Segment on command changes (held between inputs) + `entry` changes (2026-07-17:
  separate FSM re-entries/retries — e.g. a fall then a successful retry — no longer
  overwrite each other's telemetry, but at the same held command they'd otherwise merge
  into one segment here without this)."""
  change = np.any(np.diff(cmd, axis=0) != 0, axis=1)
  if entry is not None:
    change = change | (np.diff(entry) != 0)
  return [0, *(np.nonzero(change)[0] + 1), len(t)]


def analyze_hrl_csv(path: str) -> dict:
  """A1's `hrl::Telemetry` schema (`<base>_hrl.csv`): goal-state `s` gives height + the
  achieved-velocity prefix directly; `act_rate`/`period` are pre-computed columns."""
  data = np.genfromtxt(path, delimiter=",", names=True)
  t = data["t"]
  dt = float(np.median(np.diff(t)))
  cmd = np.stack([data["cmd_vx"], data["cmd_vy"], data["cmd_wz"]], axis=1)
  s_cols = sorted(n for n in data.dtype.names if n.startswith("s") and n[1:].isdigit())
  s = np.stack([data[n] for n in s_cols], axis=1)
  height = s[:, 6] if s.shape[1] >= 7 else None

  bounds = _segment_bounds(t, cmd, data["entry"] if "entry" in data.dtype.names else None)
  segs = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    dur = t[b - 1] - t[a]
    fell = height is not None and bool(height[a:b].min() < FALL_HEIGHT)
    # A fall segment is exactly the event this tool must not hide, even if it happened
    # fast (the takeover falls found 2026-07-17 collapsed in ~1-1.5s, well under this
    # floor) -- only the MIN_SEG_S floor filters out ordinary sub-3s key-press blips.
    if dur < MIN_SEG_S and not fell:
      continue
    ach = s[a:b, :3].mean(axis=0)
    err = np.abs(cmd[a:b] - s[a:b, :3]).mean(axis=0)
    seg = {
      "t0": round(float(t[a]), 2),
      "dur_s": round(float(dur), 2),
      "cmd": [round(float(v), 3) for v in cmd[a]],
      "achieved": [round(float(v), 3) for v in ach],
      "err_vx": round(float(err[0]), 3),
      "err_vy": round(float(err[1]), 3),
      "err_yaw": round(float(err[2]), 3),
      "act_rate": round(float(data["act_rate"][a:b].mean()), 3),
      "period_cmd": round(float(data["period"][a:b].mean()), 3),
    }
    stride = stride_proxy(data["hip_pitch_l"][a:b], dt)
    seg["stride_s"] = None if np.isnan(stride) else round(stride, 3)
    if height is not None:
      seg["height_min"] = round(float(height[a:b].min()), 3)
      seg["fell"] = fell
    segs.append(seg)

  return {
    "file": path,
    "schema": "hrl_telemetry",
    "duration_s": round(float(t[-1] - t[0]), 1),
    "any_fall": any(seg.get("fell", False) for seg in segs),
    "segments": segs,
  }


def analyze_base_csv(path: str) -> dict:
  """Shared `SafetyLogger` schema (`<base>.csv`, A0 or A1's base CSV, `cmd_*`/`ach_*`
  added 2026-07-21 WL-B0). No height column and no pre-computed act_rate/period, so:
  - "fell" is read off the triggers themselves (trig_fall firing at all, OR trig_tilt
    active over more than half the segment — a brief graze vs. actually stuck tilted),
    not a height threshold.
  - act_rate is computed here from consecutive raw_q ticks (mean |Δraw_q| across joints),
    the same quantity `hrl::Telemetry`'s act_rate column already logs for A1.
  - period_cmd has no equivalent (A0's clock is a fixed 0.6s YAML constant, not logged;
    A1's HL-owned period lives in `_hrl.csv` only) — omitted, not faked as 0.
  - "achieved" is [ach_vx, ach_vy] only (vx/vy, comparable to cmd); ach_vz (vertical) is
    reported separately as a diagnostic, not scored against any commanded axis. There is
    no achieved yaw-rate column (SportModeState.velocity() is linear-only) — err_yaw is
    not computable from this file.
  - ach_* is SIM-ONLY ground truth; on a real-hardware CSV it reads ~0 throughout (E1) —
    "achieved"/err_vx/err_vy would be meaningless there, not just imprecise. Detect via
    the caller-visible `sim_only_velocity` flag in the returned dict.
  - phase_alive (added 2026-07-21 with the defect-0 fix) = fraction of ticks in the
    segment where the logged gait-clock obs (phase_sin, phase_cos) is non-zero. The clock
    is stand-masked by design, so the READ is: ~0.0 on a cmd-0 segment is correct, and
    ~1.0 on any commanded segment is what a working clock looks like. **0.0 on a
    commanded segment is the dead-clock bug** (joystick-hardcoded stand-mask). Reported
    as None on CSVs captured before the columns existed.
  """
  with open(path) as f:
    header = next(csv.reader(f))
  if "cmd_vx" not in header:
    raise SystemExit(
      f"{path}: pre-2026-07-21 SafetyLogger CSV (no cmd_vx/ach_vx columns) — this file "
      "predates the WL-B0 telemetry addition and can't be read by this tool. Re-capture "
      "with the rebuilt h1_2_ctrl, or fall back to scripts/safety_analyzer.py for the "
      "trigger/violation stats this file still supports.")
  raw_q_cols = sorted((c for c in header if c.startswith("raw_q")),
                       key=lambda c: int(c[5:]))
  has_phase = "phase_sin" in header
  usecols = (["t", "meas_q1", "cmd_vx", "cmd_vy", "cmd_wz", "ach_vx", "ach_vy", "ach_vz",
              "alpha", "trig_joint", "trig_tilt", "trig_fall", "entry"] + raw_q_cols
             + (["phase_sin", "phase_cos"] if has_phase else []))
  data = np.genfromtxt(path, delimiter=",", names=True, usecols=usecols, dtype=float)
  # genfromtxt reorders/validates against `usecols` by name match against the header, so
  # column order in `usecols` above doesn't need to match the file's actual layout.
  t = data["t"]
  dt = float(np.median(np.diff(t)))
  cmd = np.stack([data["cmd_vx"], data["cmd_vy"], data["cmd_wz"]], axis=1)
  ach = np.stack([data["ach_vx"], data["ach_vy"], data["ach_vz"]], axis=1)
  sim_only_velocity = bool(np.all(ach == 0))  # real-hardware CSVs: ach_* never populated
  raw_q = np.stack([data[c] for c in raw_q_cols], axis=1)
  act_rate_tick = np.concatenate([[0.0], np.abs(np.diff(raw_q, axis=0)).mean(axis=1)])
  # The term writes an exact 0.0/0.0 when stand-masked, so "alive" is a plain non-zero
  # test, not a threshold on a noisy quantity.
  phase_live = ((data["phase_sin"] != 0) | (data["phase_cos"] != 0)) if has_phase else None

  bounds = _segment_bounds(t, cmd, data["entry"])
  segs = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    dur = t[b - 1] - t[a]
    fell = bool(data["trig_fall"][a:b].any()) or bool(data["trig_tilt"][a:b].mean() > 0.5)
    if dur < MIN_SEG_S and not fell:
      continue
    err = np.abs(cmd[a:b, :2] - ach[a:b, :2]).mean(axis=0)
    seg = {
      "t0": round(float(t[a]), 2),
      "dur_s": round(float(dur), 2),
      "cmd": [round(float(v), 3) for v in cmd[a]],
      "achieved_vxvy": [round(float(v), 3) for v in ach[a:b, :2].mean(axis=0)],
      "achieved_vz": round(float(ach[a:b, 2].mean()), 3),
      "err_vx": round(float(err[0]), 3),
      "err_vy": round(float(err[1]), 3),
      "act_rate": round(float(act_rate_tick[a:b].mean()), 3),
      "alpha_max": round(float(data["alpha"][a:b].max()), 3),
      "trig_joint_frac": round(float(data["trig_joint"][a:b].mean()), 4),
      "phase_alive": None if phase_live is None else round(float(phase_live[a:b].mean()), 3),
      "fell": fell,
    }
    stride = stride_proxy(data["meas_q1"][a:b], dt)
    seg["stride_s"] = None if np.isnan(stride) else round(stride, 3)
    segs.append(seg)

  return {
    "file": path,
    "schema": "safety_logger_base",
    "sim_only_velocity": sim_only_velocity,
    "has_phase": has_phase,
    "duration_s": round(float(t[-1] - t[0]), 1),
    "any_fall": any(seg["fell"] for seg in segs),
    "segments": segs,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("csv", help="telemetry file: <base>_hrl.csv (A1) or <base>.csv (A0/A1 base)")
  args = parser.parse_args()

  with open(args.csv) as f:
    header = next(csv.reader(f))
  is_hrl = any(c == "s0" for c in header)
  out = analyze_hrl_csv(args.csv) if is_hrl else analyze_base_csv(args.csv)

  if is_hrl:
    hdr = ("t0", "dur_s", "cmd", "achieved", "err_vx", "err_vy", "err_yaw",
           "act_rate", "period_cmd", "stride_s", "height_min", "fell")
  else:
    if out["sim_only_velocity"]:
      print("[DEPLOY-GATE] WARNING: ach_v* is all-zero throughout (real-hardware CSV, or "
            "sim run with no SportModeState publisher) -- achieved/err_vx/err_vy below are "
            "not meaningful, not just imprecise.")
    if not out["has_phase"]:
      print("[DEPLOY-GATE] NOTE: no phase_sin/phase_cos columns (CSV predates the "
            "2026-07-21 gait-clock fix) -- phase_alive is blank, and such a run was "
            "captured with the dead-clock build.")
    else:
      dead = [s for s in out["segments"]
              if max(abs(v) for v in s["cmd"]) >= 0.1 and s["phase_alive"] < 0.5]
      if dead:
        print(f"[DEPLOY-GATE] WARNING: {len(dead)} commanded segment(s) with a DEAD gait "
              "clock (phase_alive < 0.5 while a command was held) -- defect 0 is present "
              "in this build/config, results are not interpretable.")
    hdr = ("t0", "dur_s", "cmd", "achieved_vxvy", "achieved_vz", "err_vx", "err_vy",
           "act_rate", "alpha_max", "trig_joint_frac", "phase_alive", "stride_s", "fell")
  print(" | ".join(hdr))
  for seg in out["segments"]:
    print(" | ".join(str(seg.get(k, "-")) for k in hdr))

  print(f"[DEPLOY-GATE] {json.dumps(out)}")


if __name__ == "__main__":
  main()
