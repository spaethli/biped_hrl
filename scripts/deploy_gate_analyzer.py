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
from pathlib import Path

import numpy as np

FALL_HEIGHT = 0.9  # imu z below this = fallen (nominal standing 1.3076); _hrl.csv path only
MIN_SEG_S = 3.0  # ignore shorter command segments (transients between key presses)


def row_dt(t: np.ndarray) -> float:
  """Mean seconds per ROW over `t`, for converting an autocorrelation lag into a period.

  NOT `median(diff(t))`, which was the D2 bug (2026-08-03): the base CSV's `t` column was
  written with 4 SIGNIFICANT digits, so past t=10 s it quantised to 0.01 s bins and ~71%
  of consecutive rows shared a printed timestamp -> median diff 0.0 -> ZeroDivisionError
  in stride_proxy. Mean spacing is immune to that (the quantisation is +-10 ms over a
  15 s segment, 0.07%), so this reads correctly on both the pre-fix captures and the
  post-fix ones. Returns 0.0 for a degenerate segment; callers must guard.

  Caveat, unchanged by this fix: rows are NOT uniformly spaced. Quiet ticks are decimated
  to ~500 Hz while any tick with a safety trigger is logged at the full control rate, so a
  heavily-clamped run (rolloverfix: 93% trig_joint) is sampled ~2x denser than a clean one
  (A0: 10%). The autocorrelation still treats rows as evenly spaced, which is why this is
  a stride PROXY and not a measurement.
  """
  return float(t[-1] - t[0]) / max(len(t) - 1, 1) if len(t) > 1 else 0.0


def stride_proxy(hip: np.ndarray, dt: float) -> float:
  """Dominant hip-pitch cycle period via autocorrelation; NaN if no clear lock."""
  if not (dt > 0) or not np.isfinite(dt):
    return float("nan")  # degenerate segment; never raise (D2)
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


TRANSITION_S = 1.5  # scored window after each command change
BAND_MIN_HOLD_S = 10.0   # a hold at least this long, at
BAND_MIN_CMD = 0.3       # at least this commanded speed, must produce
BAND_MIN_TRAVEL_M = 0.5  # at least this much travel -- else the elastic band never released


def stand_drift(t, cmd, ach, bounds) -> list[dict]:
  """Integrated base displacement over each cmd-0 segment, with direction.

  Added 2026-08-04 after the user observed the pelvis/torso drifting left/back at zero
  command on the shipped plant, while the training-proximate plant showed the robot
  stepping instead. Reading: a standing bias has to go somewhere -- the damped plant
  suppresses the corrective stepping so the bias integrates into drift, the lively plant
  lets it step. Same defect, two plant responses. A BACKWARD component at zero command is
  also the bridge-side echo of the WL-B0b hardware backward lean, so this is the same
  instrument that would put a number on that.

  Report-only. Ruled out as causes already: joint_offset (absent from both sim YAMLs, so
  inert) and default_joint_pos asymmetry (it is L/R symmetric).
  """
  out = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    if np.abs(cmd[a]).max() > 1e-6 or (t[b - 1] - t[a]) < 3.0:
      continue
    dt = np.diff(t[a:b], prepend=t[a])
    dx, dy = float((ach[a:b, 0] * dt).sum()), float((ach[a:b, 1] * dt).sum())
    out.append({
      "t0": round(float(t[a]), 2),
      "dur_s": round(float(t[b - 1] - t[a]), 2),
      "drift_x_m": round(dx, 3), "drift_y_m": round(dy, 3),
      "drift_m": round(float(np.hypot(dx, dy)), 3),
      # atan2(y, x) in degrees: 0 = forward, +90 = left, 180 = backward.
      "heading_deg": round(float(np.degrees(np.arctan2(dy, dx))), 1),
    })
  return out


def joint_clamp_report(raw_q, meta_path) -> dict:
  """Per-joint command-clamp rate AND magnitude past the stop.

  `trig_joint` means "at least one COMMANDED joint was clamped to its h1_2_limits.h bound
  this tick" (A1 since 2026-07-16, A0 since 2026-07-23). Root-caused 2026-08-04: the
  dominant joint is ankle_roll, whose deploy limit +-0.2618 is IDENTICAL to the training
  XML's +-0.261799 -- so the limits are right and the policies genuinely command 2.1-2.7x
  past the mechanical stop. MuJoCo silently clamps qpos, so training never charges a
  policy for commanding outside range; only the deploy clamp reveals it.

  Magnitude is reported alongside rate because rate alone hides it: it is the distance
  past the stop that sets how hard firmware PD drives a pinned joint.

  NOT GATED. What trig_joint rate is acceptable is UNKNOWN. Measured reference points:
  A0 10% (hardware-proven), keeper arm4d 57% (bridge-passed), rolloverfix0p5 87-93% (fell).
  """
  if not meta_path.exists():
    return {"available": False}
  meta = json.loads(meta_path.read_text())
  joints = {j["slot"]: j for j in meta["joints"]}
  lo = np.array([joints[i]["min"] for i in range(len(joints))])
  hi = np.array([joints[i]["max"] for i in range(len(joints))])
  over = np.maximum(lo - raw_q, 0.0) + np.maximum(raw_q - hi, 0.0)
  outside = over > 1e-6
  per = []
  for i in np.argsort(-outside.mean(axis=0))[:6]:
    if outside[:, i].mean() < 0.005:
      break
    per.append({
      "joint": joints[int(i)]["name"],
      "rate": round(float(outside[:, i].mean()), 4),
      "max_past_stop_rad": round(float(over[:, i].max()), 4),
      "p99_past_stop_rad": round(float(np.percentile(over[:, i], 99)), 4),
      "limit": [joints[int(i)]["min"], joints[int(i)]["max"]],
    })
  return {"available": True, "any_joint_rate": round(float(outside.any(axis=1).mean()), 4),
          "acceptable_rate": "UNKNOWN — A0 10% hw-proven, keeper 57% bridge-passed, "
                             "rolloverfix0p5 87-93% fell",
          "per_joint": per}


def transition_windows(t, cmd, ach, bounds, fell_mask) -> list[dict]:
  """Score the TRANSITION_S seconds after every command change (task 6, 2026-08-04).

  Steady-state holds miss decel failures. The 2026-08-03 bridge fall happened at the
  0.5 -> 0 transition: the robot was walking cleanly at t=39.6 (cmd_vx 0.5, ach_vx 0.687,
  upright 1.000), then on the deceleration ach_vx LURCHED to 1.580 while commanded to
  zero, and trig_tilt/trig_fall both fired 2.8 s later. A gate that only averages over
  held segments cannot see that -- the lurch is a transient inside a segment whose mean
  looks unremarkable.

  Blocking on falls only. `overshoot` is reported, never gated: no overshoot threshold is
  grounded in any measurement, so the pipeline prints it next to A0 on the identical
  sequence and leaves the judgement to a human.
  """
  out = []
  for b in bounds[1:-1]:  # every interior bound is a command change
    m = (t >= t[b]) & (t < t[b] + TRANSITION_S)
    if m.sum() < 2:
      continue
    to, frm = cmd[b], cmd[b - 1]
    # Overshoot on the axis that actually changed, signed by the direction of travel.
    axis = int(np.argmax(np.abs(to - frm)))
    err = ach[m, axis] - to[axis] if axis < ach.shape[1] else np.zeros(m.sum())
    out.append({
      "t0": round(float(t[b]), 2),
      "from": [round(float(v), 2) for v in frm],
      "to": [round(float(v), 2) for v in to],
      "axis": "vx vy wz".split()[axis],
      "peak_overshoot": round(float(np.abs(err).max()), 3),
      "peak_achieved": round(float(ach[m, axis][np.argmax(np.abs(err))]), 3),
      "fell": bool(fell_mask[m].any()),
    })
  return out


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
    stride = stride_proxy(data["hip_pitch_l"][a:b], row_dt(t[a:b]))
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
  has_wall = "t_wall" in header  # added 2026-08-04; absent on every earlier capture
  usecols = (["t", "meas_q1", "cmd_vx", "cmd_vy", "cmd_wz", "ach_vx", "ach_vy", "ach_vz",
              "alpha", "trig_joint", "trig_tilt", "trig_fall", "entry"] + raw_q_cols
             + (["phase_sin", "phase_cos"] if has_phase else [])
             + (["t_wall"] if has_wall else []))
  data = np.genfromtxt(path, delimiter=",", names=True, usecols=usecols, dtype=float)
  # genfromtxt reorders/validates against `usecols` by name match against the header, so
  # column order in `usecols` above doesn't need to match the file's actual layout.
  t = data["t"]
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
    stride = stride_proxy(data["meas_q1"][a:b], row_dt(t[a:b]))
    seg["stride_s"] = None if np.isnan(stride) else round(stride, 3)
    segs.append(seg)

  fell_tick = np.asarray((data["trig_fall"] != 0) | (data["trig_tilt"] != 0))
  transitions = transition_windows(t, cmd, ach, bounds, fell_tick)
  # Band-release check (2026-08-04). The bridge's elastic band is a SILENT GLFW toggle
  # (main.cc:622-631, no logging) routed by X input focus, so a swallowed release yields a
  # session that looks perfect -- FSM transitions correct, CSV written, commands held --
  # with the robot hanging on the harness the whole time. Detect it from travel: over a
  # >=10 s hold at >=0.3 a walking robot covers ~3 m and a band-held one ~0, so the margin
  # is enormous and needs no tuned velocity threshold. This is INFRASTRUCTURE, not a policy
  # verdict: the caller must treat it as exit 2, never NO-GO.
  # Only holds BEFORE the first fall can answer this. A fallen robot does not travel
  # either, so scoring every hold made a genuine fall look like a harness artifact --
  # i.e. it downgraded a NO-GO to "this run tells you nothing", the single worst way this
  # gate could be wrong. Found 2026-08-04 on the keeper's first full-battery run, which
  # walked 7.1 m / 12.5 m / 3.6 m and then fell, leaving two later holds at 0.0 m.
  # The band is released once, at the start: if any pre-fall qualifying hold shows travel,
  # it released. Nothing after the first fall is evidence about the band.
  first_fall = np.argmax(fell_tick) if fell_tick.any() else len(t)
  band_ok, band_evidence = None, []
  for a, b in zip(bounds[:-1], bounds[1:]):
    if abs(cmd[a][0]) < BAND_MIN_CMD or (t[b - 1] - t[a]) < BAND_MIN_HOLD_S:
      continue
    dt = np.diff(t[a:b], prepend=t[a])
    travel = float(np.abs((ach[a:b, 0] * dt).sum()))
    post_fall = a >= first_fall
    band_evidence.append({"t0": round(float(t[a]), 2), "cmd_vx": round(float(cmd[a][0]), 2),
                          "dur_s": round(float(t[b - 1] - t[a]), 1),
                          "travel_m": round(travel, 3), "after_fall": bool(post_fall)})
    if post_fall:
      continue  # explained by the fall, not by the harness
    band_ok = True if travel >= BAND_MIN_TRAVEL_M else (band_ok or False)

  out = {
    "file": path,
    "schema": "safety_logger_base",
    "sim_only_velocity": sim_only_velocity,
    "has_phase": has_phase,
    "duration_s": round(float(t[-1] - t[0]), 1),
    "any_fall": any(seg["fell"] for seg in segs),
    "segments": segs,
    "transitions": transitions,
    "any_transition_fall": any(w["fell"] for w in transitions),
    "stand_drift": stand_drift(t, cmd, ach, bounds),
    "joint_clamp": joint_clamp_report(raw_q, Path(path).with_name(
      Path(path).name.replace(".csv", "_meta.json"))),
    # None = no qualifying hold in this session, so the check could not run (not a pass).
    "band_released": band_ok,
    "band_evidence": band_evidence,
  }
  # Loop rate, MEASURED (2026-08-04). `t` is a tick counter x nominal control_dt, so
  # t[-1]/t_wall[-1] is exactly (achieved rate)/(nominal rate) -- the ratio the 2026-08-03
  # session had to reconstruct indirectly from bridge_session.py's scripted hold durations
  # because no wall clock was logged. Reported, never gated: an honest rate read belongs
  # next to any estimator number, since integration accuracy depends on it.
  if has_wall:
    wall = float(data["t_wall"][-1] - data["t_wall"][0])
    nominal = float(t[-1] - t[0])  # elapsed NOMINAL seconds = ticks x control_dt
    out["loop_rate_hz"] = round(nominal / wall * 1000.0, 1) if wall > 0 else None
    out["wall_duration_s"] = round(wall, 1)
  else:
    out["loop_rate_hz"] = None  # pre-2026-08-04 capture: not measurable from this file
  return out


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

  if not is_hrl:
    if out["band_released"] is False:
      print("\n[DEPLOY-GATE] BAND-NOT-RELEASED: a held command produced almost no travel "
            "-- the robot was on the harness for this session. INFRASTRUCTURE failure "
            "(exit 2), not a policy verdict; the run tells you nothing about the policy.")
      for e in out["band_evidence"]:
        print(f"    cmd_vx {e['cmd_vx']} held {e['dur_s']}s -> travelled {e['travel_m']} m")
    elif out["band_released"] is None:
      print(f"\n[DEPLOY-GATE] NOTE: no hold >= {BAND_MIN_HOLD_S}s at cmd >= {BAND_MIN_CMD} "
            "in this session, so the band-release check could not run (not a pass).")

    if out["transitions"]:
      print(f"\ntransition windows ({TRANSITION_S}s after each command change) -- "
            "falls BLOCK, overshoot is report-only (no grounded threshold):")
      print("  t0 | from -> to | axis | peak_achieved | peak_overshoot | fell")
      for w in out["transitions"]:
        print(f"  {w['t0']} | {w['from']} -> {w['to']} | {w['axis']} | "
              f"{w['peak_achieved']} | {w['peak_overshoot']} | {w['fell']}")

    if out["stand_drift"]:
      print("\ncmd-0 drift (report-only; 0deg=fwd, +90=left, 180=back):")
      for d in out["stand_drift"]:
        print(f"  t0 {d['t0']} dur {d['dur_s']}s -> {d['drift_m']} m at "
              f"{d['heading_deg']}deg  (x {d['drift_x_m']}, y {d['drift_y_m']})")

    jc = out["joint_clamp"]
    if jc.get("available") and jc["per_joint"]:
      print(f"\njoint command clamp: any-joint {jc['any_joint_rate']:.1%} of ticks "
            f"(NOT gated; acceptable rate {jc['acceptable_rate']})")
      for j in jc["per_joint"]:
        print(f"  {j['joint']:22s} {j['rate']:6.1%}  max {j['max_past_stop_rad']:+.3f} rad "
              f"past stop, p99 {j['p99_past_stop_rad']:.3f}  limit {j['limit']}")
    if out.get("loop_rate_hz"):
      print(f"\nloop rate MEASURED: {out['loop_rate_hz']} Hz over {out['wall_duration_s']}s "
            "wall (nominal 1000)")

  print(f"[DEPLOY-GATE] {json.dumps(out)}")


if __name__ == "__main__":
  main()
