"""Deploy-gate analyzer (plan W3) for the State_RLHRL telemetry CSV (<base>_hrl.csv).

Segments the run by held-command changes and scores each segment: achieved-vs-commanded
velocity (the G2 parity numbers), height (fall detection), action rate, commanded period,
and a stride-period proxy (autocorrelation of the measured left hip pitch — one full hip
cycle = one stride, the same-foot touchdown interval). Prints a table + `[DEPLOY-GATE]
{json}` for the plan's pass criteria.

Usage: python scripts/deploy_gate_analyzer.py logs/deploy_safety/<ts>_hrl.csv
"""

import argparse
import json

import numpy as np

FALL_HEIGHT = 0.9  # imu z below this = fallen (nominal standing 1.3076)
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


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("csv", help="telemetry file (<base>_hrl.csv)")
  args = parser.parse_args()

  data = np.genfromtxt(args.csv, delimiter=",", names=True)
  t = data["t"]
  dt = float(np.median(np.diff(t)))
  cmd = np.stack([data["cmd_vx"], data["cmd_vy"], data["cmd_wz"]], axis=1)
  s_cols = sorted(n for n in data.dtype.names if n.startswith("s") and n[1:].isdigit())
  s = np.stack([data[n] for n in s_cols], axis=1)
  goal_dim = s.shape[1]
  height = s[:, 6] if goal_dim >= 7 else None

  # Segment on command changes (commands are HELD between inputs).
  change = np.any(np.diff(cmd, axis=0) != 0, axis=1)
  bounds = [0, *(np.nonzero(change)[0] + 1), len(t)]
  segs = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    dur = t[b - 1] - t[a]
    if dur < MIN_SEG_S:
      continue
    c = cmd[a]
    ach = s[a:b, :3].mean(axis=0)
    err = np.abs(cmd[a:b] - s[a:b, :3]).mean(axis=0)
    seg = {
      "t0": round(float(t[a]), 2),
      "dur_s": round(float(dur), 2),
      "cmd": [round(float(v), 3) for v in c],
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
      seg["fell"] = bool(height[a:b].min() < FALL_HEIGHT)
    segs.append(seg)

  hdr = ("t0", "dur_s", "cmd", "achieved", "err_vx", "err_vy", "err_yaw",
         "act_rate", "period_cmd", "stride_s", "height_min", "fell")
  print(" | ".join(hdr))
  for seg in segs:
    print(" | ".join(str(seg.get(k, "-")) for k in hdr))

  out = {
    "file": args.csv,
    "duration_s": round(float(t[-1] - t[0]), 1),
    "any_fall": any(seg.get("fell", False) for seg in segs),
    "segments": segs,
  }
  print(f"[DEPLOY-GATE] {json.dumps(out)}")


if __name__ == "__main__":
  main()
