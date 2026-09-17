#!/usr/bin/env python3
"""Ground-truth base velocity for one hardware RUN, from a Vicon export.

A Run is one FSM `entry` (CONTEXT.md), already sliced into a bundle by
`bundle_hardware_run.py`. This tool adds the one thing no onboard log carries: the
robot's ACTUAL base velocity, against which `leg_odom`, the complementary filter and
every EKF arm can finally be scored rather than compared to each other.

TWO UNKNOWNS, FITTED SEPARATELY. The Vicon clock and the flight recorder's wall clock
`t_wall` differ by an offset (operator timing) and a small rate (+-141 ppm measured), and the
Vicon rigid-body frame is some arbitrary constant rotation R away from the robot's torso
frame. Fitting both together is ill-conditioned: a clock error and a frame error both show up
as "the traces do not match", and the optimiser trades one against the other. They are
decoupled here by choosing signals that see only one unknown at a time:

  1. TIME, from |omega|.  The MAGNITUDE of the angular velocity is invariant to R, so the
     mocap's |omega| and the flight recorder's |est_gyro| are the same scalar function of
     time whatever the frame relationship is. It is fitted against the recorder's `t_wall`,
     never its tick counter `t` (read_flight), and only on CLEAN mocap: rows refused by the
     reader (non-finite, < 3 markers) and holes in the time axis are dropouts, and nothing
     within GAP_ERODE_S of one is used anywhere (clean_mask). A masked search over every lag
     gives the whole-take offset (global_lag); each <= 20 s clean window then finds its own
     lag near it, and the windows that agree give offset + rate (window_lags, fit_windows).
     No frame knowledge is used.
  2. FRAME, from omega.  With the clocks aligned, omega_mocap(t) = R omega_torso(t) holds
     sample by sample, which is orthogonal Procrustes: one SVD, in closed form, with the
     reflection forced out (det = +1), on clean samples only. No calibration pose, no T-pose
     capture, no manual alignment of the rigid body in Nexus. Its residual is scored where
     |omega| > 0.2 rad/s, and its ACCURACY is how well per-window fits agree with the pooled
     one; disagreement above 4 deg is flagged in run.json, not refused. The accelerometer,
     which the fit never sees, checks R's roll/pitch against gravity on quasi-static rows
     (gravity_tilt); above 3 deg is flagged the same way.

Measured on 2026-09-14/16 (25 takes, dropouts of up to 28 s where the robot left the volume):
all 25 align, kept windows agree to <= 8 ms, per-window R to 0.3-2.4 deg and gravity to
0.2-1.8 deg; one take (2026-09-16 T15) is flagged by both. The previous tick-counter,
whole-take version aligned 3 of 5 and 11 of 21.

Only then is the LINEAR velocity computed, by differentiating the mocap position and
rotating it through the now-known frames.

The lever arm r (pelvis -> marker cluster) is MEASURED and passed in, not fitted from the
estimator (that would be the circularity below). Step 5 estimates it from the mocap alone, by
pivot calibration over the zero-command segments; see `pivot_lever`. It is a report, the
output never uses the fitted value and nothing warns on it: on real Runs it is not precise
enough to be a check (see pivot_lever).

⚠ THE ALIGNMENT IS NEVER FITTED ON THE SCORED SIGNAL. Angular velocity is used for both
steps precisely because the thing being validated is a LINEAR velocity estimate. Fitting
the clock (or the frame) by maximising agreement between the mocap's linear velocity and
the robot's estimate would manufacture exactly the agreement the experiment is supposed to
test: a lag fitted that way absorbs the estimator's own lag, and a rotation fitted that way
absorbs its cross-axis leakage. The two error sources the experiment cares about would be
silently removed from it. Angular rate is measured by an independent sensor (the gyro) and
is not an output of any estimator under test.

FRAMES (doc/hrl/h1_2_kinematics.md). `est_gyro` is the TORSO gyro, so the Procrustes R maps
torso -> mocap rigid body, and the marker cluster is assumed rigid with the TORSO. The pelvis
and torso origins coincide and differ only by the waist yaw psi, so the output applies the
deployed conversion `v_P^P = Rz(psi) (v_I^T - omega_T x r)` (base_state.h:18-22) with psi
read from `meas_q12`. That rotation is NOT cosmetic: split deploy is supposed to hold the
waist, but measured across the eleven 2026-09-14 runs |psi| reaches p95 0.03-0.25 rad and
peaks near 0.6, so it is applied per sample and its magnitude is reported.

FORMAT. Two readers, dispatched on layout. Raw Nexus ASCII is sniffed field by field, with the
decision printed before any work and a CLI override for each. The 2026-09-14 and 2026-09-16
captures arrived PRE-PROCESSED instead -- one header row, one rigid body, `time_s`, `x/y/z_mm`,
`qw..qz`, `wx/wy/wz_deg_s`, `speed_mm_s`, `residual_mm`, `markers_used` at 100 Hz -- and take
the flat path. ⚠ Their angular-rate triple is in the WORLD frame, which no column name says; it
is rotated through R before use. The exported triple beats differentiating attitude, which at
100 Hz turns sub-millimetre marker noise into ~0.7 rad/s of spurious rate. Their `a*_mm_s2`
columns are only d/dt of their velocity and are not used.

`--selftest` proves the pipeline end to end against a synthetic trajectory with a planted
clock offset and skew, a tick counter that runs slow and bends, two dropouts with smeared rows
before them, a frame rotation, a lever arm and an accelerometer (recovered to 3.0 ms, 0.9 ppm,
0.0015 deg, 0.5 mm/s, tilt 0.01 deg), and shows that dropping either the wall clock or the
dropout margin breaks the velocity (42x, 332x).

Usage:
  python scripts/mocap_align.py --selftest
  python scripts/mocap_align.py logs/robot_logs/2026_09_16-_13_24_A1a_DR_cotcap_s123 \\
      logs/robot_logs/2026_09_16-_13_24_A1a_DR_cotcap_s123/mocap.csv --lever 0.0 0.0 0.32
  # ... and when the sniffer guesses wrong (raw Nexus):
      --rot-format euler_xyz --rot-cols RX,RY,RZ --pos-cols TX,TY,TZ --pos-units mm --rate 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_hardware_run import MAX_SKEW, PAIR_CORR_FLOOR  # noqa: E402
from replay_base_estimators import quat_to_mat, so3_exp, so3_log  # noqa: E402

OUT_NAME = "mocap_aligned.csv"
PROC_ANGLE_MAX = 15.0   # deg; median angle between R*omega_torso and omega_mocap (above OMEGA_RESID_MIN)
MIN_EXCITATION = 0.02   # smallest singular value / largest, of the Procrustes matrix
SMOOTH_S = 0.02         # boxcar on |omega| before the clock search
FRAME_SMOOTH_S = 0.10   # boxcar on the omega VECTORS before the frame fit (see below)
PSI_WARN = 0.05         # rad; above this the waist is not "held" and the mount matters
PIVOT_WIN_S = 2.0       # s; pivot windows, each detrended on its own (absorbs slow pelvis drift)
PIVOT_CMD_EPS = 0.05    # m/s; |cmd_vx|, |cmd_vy| below this = the robot was not asked to translate
MIN_MARKERS = 3         # a rigid body seen on fewer markers has no trustworthy attitude
GAP_ERODE_S = 0.20      # s trimmed off each side of a dropout; the export's filter smears into it
WIN_S = 20.0            # s; clock windows cut from the clean stretches
WIN_MIN_S = 4.0         # s; shortest clock window (half the take, for a take shorter than 8 s)
WIN_CORR_MIN = 0.8      # |omega| correlation a window needs to vote on the lag
WIN_AGREE_S = 0.05      # s; a voting window this far from the median lag is an alias, dropped
LOCAL_S = 1.0           # s; per-window lag search half-width around the whole-take offset
GLOBAL_GRID_S = 0.02    # s; grid of the whole-take masked search
OMEGA_RESID_MIN = 0.2   # rad/s; the frame residual is scored only where the robot really rotates
FRAME_SPREAD_MAX = 4.0  # deg; median per-window R disagreement above this is FLAGGED (still written)
SKEW_WARN = 500e-6      # t_wall and Vicon are two real clocks; a larger fitted rate is flagged
GRAVITY = 9.81          # m/s^2
TILT_STILL_GYRO = 0.15  # rad/s; |gyro| below this counts as quasi-static for the gravity check
TILT_STILL_ACC = 0.3    # m/s^2; and | |acc| - g | below this
TILT_MIN_ROWS = 200     # quasi-static rows needed before the gravity check reports
TILT_FLAG_DEG = 3.0     # deg; accelerometer-vs-mocap tilt above this is FLAGGED (still written)


# --------------------------------------------------------------------------- vicon reader

def _f(s):
  try:
    return float(s)
  except (TypeError, ValueError):
    return None


def _ffill(row):
  out, cur = [], ""
  for c in row:
    c = c.strip()
    if c:
      cur = c
    out.append(cur)
  return out


def _resolve(spec, comps, full):
  """'2,3,4' or 'RX,RY,RZ' or 'H1_2:torso:RX,...' -> column indices."""
  idx = []
  for tok in spec.split(","):
    tok = tok.strip()
    if tok.isdigit():
      idx.append(int(tok))
      continue
    hit = [i for i, (c, fu) in enumerate(zip(comps, full))
           if c.upper() == tok.upper() or fu.upper() == tok.upper()]
    if not hit:
      raise SystemExit(f"column '{tok}' not in the mocap header; components present: "
                       f"{[c for c in comps if c]}")
    idx.append(hit[0])
  return idx


def read_flat_csv(path, quiet=False):
  """A PRE-PROCESSED single-header-row export -> the same dict `read_vicon` returns.

  The 2026-09-14 captures did not arrive as raw Nexus ASCII. They came already reduced to one
  header row and one rigid body, carrying `time_s`, filtered `x_mm/y_mm/z_mm`, a quaternion
  `qw/qx/qy/qz`, Euler `roll/pitch/yaw_deg`, body rates `wx/wy/wz_deg_s`, `speed_mm_s`, a fit
  `residual_mm` and `markers_used`. There is no rate line, no object-name row and no `Frame`
  column, so the Nexus sniffer cannot read them.

  Time comes from `time_s` directly rather than a frame index, so no rate is assumed. The
  quaternion is taken as world<-body, matching the Nexus path. `x_raw_mm` is ignored in favour
  of the filtered triple; the raw one is retained in the file for provenance.
  """
  d = np.genfromtxt(path, delimiter=",", names=True)
  need = ["time_s", "x_mm", "y_mm", "z_mm", "qw", "qx", "qy", "qz"]
  missing = [c for c in need if c not in (d.dtype.names or ())]
  if missing:
    raise SystemExit(f"{path}: flat CSV is missing {missing}. Expected a processed export "
                     f"with time_s, x/y/z_mm and a qw/qx/qy/qz quaternion.")
  t = np.asarray(d["time_s"], float)
  p = np.column_stack([d["x_mm"], d["y_mm"], d["z_mm"]]) * 1e-3
  q = np.column_stack([d["qw"], d["qx"], d["qy"], d["qz"]])
  ok = np.isfinite(t) & np.isfinite(p).all(1) & np.isfinite(q).all(1)
  # The exported angular-rate triple, when present, beats differentiating the attitude: it is
  # already filtered, whereas so3_log on successive 100 Hz frames amplifies marker noise
  # enormously (1 mm on a ~0.15 m cluster is ~0.4 deg of attitude, which over a 10 ms step is
  # ~0.7 rad/s of spurious rate). Measured on TRIAL05: quaternion-derived omega fits the frame
  # at 15.8 deg, the exported triple at 7.5 deg.
  # ⚠ It is in the WORLD frame, not the body frame. That is not documented anywhere in the
  # file; it was established by fitting both ways (TRIAL05 21.8 deg as-given vs 10.8 deg
  # rotated, TRIAL03 19.5 vs 12.6). `align` rotates it through R before use.
  w_world = None
  wc = ["wx_deg_s", "wy_deg_s", "wz_deg_s"]
  if all(c in (d.dtype.names or ()) for c in wc):
    w_world = np.deg2rad(np.column_stack([d[c] for c in wc]))
    ok = ok & np.isfinite(w_world).all(1)
  if "markers_used" in (d.dtype.names or ()):
    ok = ok & (np.nan_to_num(d["markers_used"], nan=0.0) >= MIN_MARKERS)
  dropped = int((~ok).sum())
  bad_t = t[~ok & np.isfinite(t)]
  t, p, q = t[ok], p[ok], q[ok]
  if w_world is not None:
    w_world = w_world[ok]
  R = rot_matrices(q, "quat", False)
  info = {"file": Path(path).name, "rate": float(1.0 / np.median(np.diff(t))),
          "rate_src": "from time_s", "object": "flat-csv", "objects": ["flat-csv"],
          "rot_format": "quat", "fmt_src": "flat-csv qw/qx/qy/qz", "rot_cols": [],
          "pos_cols": [], "rot_units": "-", "pos_units": "mm", "units_src": "column names",
          "n": len(t), "dropped": dropped, "subframe": False}
  if not quiet:
    print("== mocap format ============================================================")
    print(f"  file          {info['file']}  ({info['n']} rows, {t[-1] - t[0]:.1f} s, "
          f"{dropped} rows dropped: non-finite or < {MIN_MARKERS} markers)")
    print(f"  layout        PRE-PROCESSED flat CSV (single header row), {info['rate']:.1f} Hz "
          f"from time_s")
    print("  rotation      quaternion qw/qx/qy/qz, taken as world<-body")
    print("  position      x/y/z_mm -> m")
  return dict(t=t, R=R, p=p, w_world=w_world, bad_t=bad_t, info=info)


def read_mocap(path, **kw):
  """Dispatch on layout: a `Frame` header row means Nexus ASCII, otherwise a flat CSV."""
  with open(path, newline="") as fh:
    head = [next(fh, "") for _ in range(40)]
  if any(r.split(",")[0].strip().lower() == "frame" for r in head if r):
    return read_vicon(path, **kw)
  return read_flat_csv(path, quiet=kw.get("quiet", False))


def read_vicon(path, rot_cols=None, pos_cols=None, rot_format=None, pos_units=None,
               rate=None, quiet=False):
  """A Vicon ASCII export -> dict(t, R [N,3,3] world<-body, p [N,3] metres world, ...).

  The layout this sniffs is the Nexus ASCII shape: an optional section line, a lone numeric
  RATE line, a row of object names (sparse -- the name appears once per group and is
  forward-filled), a row of component names starting with `Frame,Sub Frame`, an optional
  units row, then data. Nothing here is hard-coded to one variant: rotations arrive as
  helical (rotation vector), Euler XYZ or quaternion depending on what was ticked in the
  export dialog, and position is usually but not always millimetres. Every field the
  sniffer decides is printed and every one has a CLI override, because a silently wrong
  guess here would show up downstream as a physics result.
  """
  with open(path, newline="") as fh:
    rows = list(csv.reader(fh))
  hdr = next((i for i, r in enumerate(rows)
              if r and r[0].strip().lower() == "frame"), None)
  if hdr is None:
    raise SystemExit(f"{path}: no 'Frame' header row found. This reader expects a Vicon "
                     f"ASCII export (rate line, name row, 'Frame,Sub Frame,...' row, data).")
  comps = [c.strip() for c in rows[hdr]]
  objs = _ffill(rows[hdr - 1]) if hdr else [""] * len(comps)
  objs += [""] * (len(comps) - len(objs))
  full = [f"{o}:{c}" if o else c for o, c in zip(objs, comps)]

  # rate: the last line before the header block that is a single bare number
  detected_rate = None
  for r in rows[:hdr]:
    cells = [c for c in r if c.strip()]
    if len(cells) == 1 and _f(cells[0]) is not None:
      detected_rate = float(cells[0])
  rate_src = "--rate" if rate else ("rate line" if detected_rate else "MISSING")
  rate = rate or detected_rate
  if not rate or rate <= 0:
    raise SystemExit(f"{path}: no sample-rate line found and --rate not given. The Vicon "
                     f"frame index is the only time base in the file, so the rate is "
                     f"required to turn it into seconds.")

  # units row: present when the row under the header holds no numbers
  units_row = None
  if hdr + 1 < len(rows) and not any(_f(c) is not None for c in rows[hdr + 1][:8]):
    units_row = [c.strip().lower() for c in rows[hdr + 1]]
  data0 = hdr + (2 if units_row else 1)

  # group the data columns by object and by what they are
  rot, pos = {}, {}
  for i, c in enumerate(comps[2:], start=2):
    u, o = c.upper(), objs[i]
    if u in ("RX", "RY", "RZ", "RW", "QX", "QY", "QZ", "QW"):
      rot.setdefault(o, {})[u[-1]] = i
    elif u in ("TX", "TY", "TZ"):
      pos.setdefault(o, {})[u[-1]] = i
    elif u in ("X", "Y", "Z"):
      pos.setdefault(o, {}).setdefault(u, i)
  objects = [o for o in dict.fromkeys(objs[2:]) if o]

  if rot_cols:
    ri = _resolve(rot_cols, comps, full)
    obj = objs[ri[0]]
  else:
    obj = next((o for o in objects if len(rot.get(o, {})) >= 3 and len(pos.get(o, {})) == 3),
               None)
    if obj is None:
      raise SystemExit(
        f"{path}: no object carries both rotation and position columns "
        f"(found objects {objects or '[]'}). A marker-only 'Trajectories' export has no "
        f"orientation, and this tool needs it for both the frame fit and the body-frame "
        f"rotation. Re-export the rigid body / segment, or name the columns with "
        f"--rot-cols/--pos-cols.")
    order = "WXYZ" if len(rot[obj]) == 4 else "XYZ"
    ri = [rot[obj][k] for k in order]
  pi = _resolve(pos_cols, comps, full) if pos_cols else [pos[obj][k] for k in "XYZ"]
  if len(ri) not in (3, 4) or len(pi) != 3:
    raise SystemExit(f"need 3 or 4 rotation columns and 3 position columns, "
                     f"got {len(ri)} and {len(pi)}")

  fmt_src = "--rot-format"
  if rot_format is None:
    # 4 columns can only be a quaternion. 3 is ambiguous between Vicon's default helical
    # (rotation-vector) export and an Euler triple; helical is the dialog default, so that
    # is the assumption, stated out loud. A wrong pick does not pass silently -- it wrecks
    # omega, and both the |omega| cross-correlation and the Procrustes residual gate on it.
    rot_format = "quat" if len(ri) == 4 else "helical"
    fmt_src = "ASSUMED (4 rotation columns)" if len(ri) == 4 else "ASSUMED (Nexus default)"
  if (rot_format == "quat") != (len(ri) == 4):
    raise SystemExit(f"--rot-format {rot_format} wants {'4' if rot_format == 'quat' else '3'}"
                     f" rotation columns, but {len(ri)} were selected")

  # units: the file's own row wins over the mm default, the CLI wins over both
  u_rot = (units_row[ri[0]] if units_row and ri[0] < len(units_row) else "")
  u_pos = (units_row[pi[0]] if units_row and pi[0] < len(units_row) else "")
  deg = rot_format != "quat" and u_rot != "rad"
  if pos_units:
    units, units_src = pos_units, "--pos-units"
  elif u_pos in ("mm", "m"):
    units, units_src = u_pos, "units row"
  else:
    units, units_src = "mm", "ASSUMED (no units row)"

  raw = np.genfromtxt(path, delimiter=",", skip_header=data0, invalid_raise=False,
                      filling_values=np.nan, usecols=range(len(comps)))
  raw = np.atleast_2d(raw)
  if raw.size == 0 or raw.shape[0] < 10:
    raise SystemExit(f"{path}: only {raw.shape[0]} data rows parsed below the header")
  use = raw[:, [0, 1] + list(ri) + list(pi)]
  ok = np.isfinite(use).all(axis=1)
  dropped = int((~ok).sum())
  bad_frames = use[~ok & np.isfinite(use[:, 0]), 0]
  use = use[ok]
  frame, sub = use[:, 0], use[:, 1]
  rvals, pvals = use[:, 2:2 + len(ri)], use[:, 2 + len(ri):]

  t = (frame - frame[0]) / rate
  bad_t = (bad_frames - frame[0]) / rate
  p = pvals * (1e-3 if units == "mm" else 1.0)
  R = rot_matrices(rvals, rot_format, deg)

  info = {
    "file": Path(path).name, "rate": rate, "rate_src": rate_src, "object": obj or "?",
    "objects": objects, "rot_format": rot_format, "fmt_src": fmt_src,
    "rot_cols": ri, "pos_cols": pi, "rot_units": "deg" if deg else "rad",
    "pos_units": units, "units_src": units_src, "n": len(t), "dropped": dropped,
    "subframe": bool(np.any(sub != 0)),
  }
  if not quiet:
    print("== mocap format ============================================================")
    print(f"  file          {info['file']}  ({info['n']} rows, {t[-1] - t[0]:.1f} s, "
          f"{dropped} gap rows dropped)")
    print(f"  rate          {rate:g} Hz   [{rate_src}]   -> t = (Frame - Frame0) / rate")
    print(f"  object        {info['object']}   (of {len(objects)} found: "
          f"{', '.join(objects) or '-'})")
    ru = "" if rot_format == "quat" else " in " + info["rot_units"]
    print(f"  rotation      {rot_format}{ru}   [{fmt_src}]   "
          f"cols {ri} = {[comps[i] for i in ri]}")
    print(f"  position      {units} -> m   [{units_src}]   cols {pi} = "
          f"{[comps[i] for i in pi]}")
    if info["subframe"]:
      print("  ⚠ non-zero Sub Frame values present; this reader times rows by Frame alone")
    print("  (every line above is overridable: --rate --rot-cols --pos-cols --rot-format "
          "--pos-units)")
  return dict(t=t, R=R, p=p, bad_t=bad_t, info=info)


def rot_matrices(vals, fmt, deg):
  """Vicon rotation columns -> [N,3,3] world<-body rotation matrices."""
  v = np.deg2rad(vals) if (deg and fmt != "quat") else np.asarray(vals, float)
  if fmt == "quat":
    # `read_vicon` selects the four columns by their header letters (W/X/Y/Z), not by their
    # order in the file, so a scalar-first and a scalar-last export both arrive as (w,x,y,z).
    return np.stack([quat_to_mat(q) for q in v])
  if fmt == "helical":
    return np.stack([so3_exp(w) for w in v])           # rotation vector = axis * angle
  if fmt == "euler_xyz":
    # Intrinsic X-then-Y-then-Z: R = Rx(a) Ry(b) Rz(c). Built from so3_exp so there is one
    # rotation primitive in this file, not four.
    return np.stack([so3_exp([a, 0, 0]) @ so3_exp([0, b, 0]) @ so3_exp([0, 0, c])
                     for a, b, c in v])
  raise SystemExit(f"unknown --rot-format {fmt}")


# --------------------------------------------------------------------------- flight reader

def read_flight(path):
  """(t, t_wall, gyro [N,3], psi [N], cmd [N,2], acc [N,3]) from a flight-recorder slice.

  `t_wall`, `cmd` and `acc` are None when the slice has no such columns.

  Every row is kept, including the ~50% that are exact sensor repeats: the spec is to emit
  ground truth on the recorder's OWN row grid, so consumers can join on `t` without an
  interpolation of their own.

  ⚠ `t` is a TICK COUNTER (bundle_hardware_run.py), so it is the output key but never the
  clock the mocap is fitted against. Measured on the 21 Runs of 2026-09-16: against `t_wall`
  it runs 0.27-1% slow AND bends by 36 ms to 1.76 s after the best straight line, which no
  offset-plus-rate map can follow. Fitted on `t_wall` the "skew" to Vicon falls from
  -2800..-9700 ppm to within +-141 ppm, i.e. two real clocks.
  """
  d = np.genfromtxt(path, delimiter=",", names=True)
  have = d.dtype.names
  for c in ("t", "est_gyro_x", "est_gyro_y", "est_gyro_z"):
    if c not in have:
      raise SystemExit(f"{Path(path).name}: column '{c}' missing. The gyro is the only "
                       f"signal both logs share, so a recorder slice without the `est_` "
                       f"block cannot be aligned to mocap at all.")
  t = np.atleast_1d(d["t"]).astype(float)
  t_wall = np.atleast_1d(d["t_wall"]).astype(float) if "t_wall" in have else None
  gyro = np.stack([d[f"est_gyro_{a}"] for a in "xyz"], 1).astype(float)
  psi = np.atleast_1d(d["meas_q12"]).astype(float) if "meas_q12" in have else np.zeros(len(t))
  cmd = (np.stack([d["cmd_vx"], d["cmd_vy"]], 1).astype(float)
         if {"cmd_vx", "cmd_vy"} <= set(have) else None)
  acc = (np.stack([d[f"acc_{a}"] for a in "xyz"], 1).astype(float)
         if {"acc_x", "acc_y", "acc_z"} <= set(have) else None)
  return t, t_wall, gyro, psi, cmd, acc


# --------------------------------------------------------------------------- the pipeline

def body_rates(R, t):
  """Body-frame angular velocity from successive attitudes, CENTRED.

  `so3_log(R[k-1]^T R[k+1]) / (t[k+1] - t[k-1])` rather than the one-sided difference
  `_gyro_from_quat` uses: a one-sided difference is the rate at the MIDPOINT of the
  interval, i.e. it carries a half-sample lag (2.5 ms at 200 Hz), and that lag would be
  fitted out by step 1 as a clock offset and then wrongly applied to the position channel
  as well.
  """
  n = len(t)
  w = np.zeros((n, 3))
  for k in range(1, n - 1):
    dt = t[k + 1] - t[k - 1]
    w[k] = so3_log(R[k - 1].T @ R[k + 1]) / dt if dt > 0 else w[k - 1]
  if n > 2:
    w[0], w[-1] = w[1], w[-2]
  return w


def _boxcar(x, t, win_s=SMOOTH_S):
  dt = np.median(np.diff(t))
  n = int(win_s / dt) if dt > 0 else 0
  return x if n < 2 else np.convolve(x, np.ones(n) / n, mode="same")


def clean_mask(t, bad_t, erode=GAP_ERODE_S):
  """True where a mocap sample sits at least `erode` from every dropout.

  A dropout is either a row the reader refused (`bad_t`: non-finite, or fewer than
  MIN_MARKERS) or a hole in the time axis. Nothing downstream may interpolate across one:
  on 2026-09-16 the robot walked out of the volume for up to 28 s, and every step used to
  draw a straight line through that hole -- fake |omega| for the clock, fake rotation for the
  frame, fake velocity in the written ground truth. The margin exists because the export's
  own filter smears into the rows next to a hole.
  """
  dt = np.diff(t)
  gi = np.flatnonzero(dt > 1.5 * np.median(dt)) if len(dt) else np.zeros(0, int)
  lo = np.r_[np.asarray(bad_t, float), t[gi]] - erode
  hi = np.r_[np.asarray(bad_t, float), t[gi + 1]] + erode
  if lo.size == 0:
    return np.ones(len(t), bool)
  o = np.argsort(lo)
  lo, hi = lo[o], np.maximum.accumulate(hi[o])
  k = np.searchsorted(lo, t, "right") - 1
  return ~((k >= 0) & (t <= hi[np.maximum(k, 0)]))


def _xc_valid(a, k):
  """sum_i a[i + j] k[i] for every j in 0..len(a)-len(k), by FFT."""
  n = len(a)
  r = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(k, n)), n)
  return r[:n - len(k) + 1]


def global_lag(t_c, sig_f, t_m, sig_m, clean, grid=GLOBAL_GRID_S):
  """Whole-take offset L (t_clock = t_mocap + L) and its correlation, or None.

  Masked normalized cross-correlation over EVERY lag, so no seed is needed: it found the two
  3.4-3.8 s tail-of-Run takes of 2026-09-16 (+55.28 s, +18.60 s) that `coarse_lag` could not.
  Both sides are masked: clean mocap only, and only the part of the take that overlaps the
  flight slice at that lag. A take can be LONGER than its slice (TRIAL02 of 2026-09-14 runs
  on into the next entry of the same file); scoring the overhang against padding diluted the
  true peak below a wrong one. An overlap must cover min(20 s, 80% of the clean take, 80%
  of the slice).
  """
  gf = np.arange(t_c[0], t_c[-1], grid)
  gm = np.arange(t_m[0], t_m[-1], grid)
  M = (np.interp(gm, t_m, clean.astype(float)) > 0.999).astype(float)
  if M.sum() < 50:
    return None
  pad = len(gm)
  F = np.r_[np.zeros(pad), np.interp(gf, t_c, sig_f), np.zeros(pad)]
  V = np.r_[np.zeros(pad), np.ones(len(gf)), np.zeros(pad)]
  A = np.interp(gm, t_m, sig_m) * M
  n = _xc_valid(V, M)
  sF, sFF = _xc_valid(F, M), _xc_valid(F * F, M)          # F is already 0 on the padding
  sA, sAA = _xc_valid(V, A), _xc_valid(V, A * A)
  cov = _xc_valid(F, A) - sF * sA / np.maximum(n, 1)
  var = (np.maximum(sFF - sF ** 2 / np.maximum(n, 1), 1e-12)
         * np.maximum(sAA - sA ** 2 / np.maximum(n, 1), 1e-12))
  need = min(20.0 / grid, 0.8 * M.sum(), 0.8 * len(gf))
  c = np.where(n >= need - 0.5, cov / np.sqrt(var), -np.inf)
  j = int(np.argmax(c))
  if not np.isfinite(c[j]):
    return None
  return float(gf[0] - pad * grid + j * grid - gm[0]), float(c[j])


def window_lags(t_c, sig_f, t_m, sig_m, clean, L0, grid=0.005):
  """[(t_mid, length, corr, lag)] for each <= WIN_S window of each clean stretch.

  Each window searches +-LOCAL_S around the whole-take offset on a 5 ms grid with a
  parabolic sub-sample peak. Windows are cut from the part of a stretch that lands inside
  the flight log, so a take starting before the RL state still votes with its later part.
  """
  edges = np.flatnonzero(np.diff(np.r_[0, clean.astype(int), 0]))
  min_len = min(WIN_MIN_S, 0.5 * (t_m[-1] - t_m[0]))
  Ls = np.arange(L0 - LOCAL_S, L0 + LOCAL_S + 1e-9, grid)
  out = []
  for s, e in zip(edges[::2], edges[1::2]):
    ins = np.flatnonzero((t_m[s:e] + L0 >= t_c[0] + LOCAL_S)
                         & (t_m[s:e] + L0 <= t_c[-1] - LOCAL_S)) + s
    if not len(ins):
      continue
    s2, e2 = ins[0], ins[-1] + 1
    cuts = np.linspace(s2, e2, max(int(np.ceil((t_m[e2 - 1] - t_m[s2]) / WIN_S)), 1) + 1)
    for a, b in zip(cuts[:-1].astype(int), cuts[1:].astype(int)):
      span = t_m[b - 1] - t_m[a]
      if span < min_len:
        continue
      g = np.arange(t_m[a], t_m[b - 1], grid)
      A = np.interp(g, t_m, sig_m)
      A = A - A.mean()
      F = np.interp(g[None, :] + Ls[:, None], t_c, sig_f)
      F = F - F.mean(1, keepdims=True)
      c = (F @ A) / (np.linalg.norm(F, axis=1) * np.linalg.norm(A) + 1e-12)
      j = int(np.argmax(c))
      lag = Ls[j]
      if 0 < j < len(c) - 1:
        den = c[j - 1] - 2 * c[j] + c[j + 1]
        if den < 0:
          lag += 0.5 * grid * (c[j - 1] - c[j + 1]) / den
      out.append((t_m[a] + 0.5 * span, span, float(c[j]), float(lag)))
  return np.array(out).reshape(-1, 4)


def fit_windows(win):
  """Offset + rate through the agreeing windows, or None when no window votes.

  A window votes at WIN_CORR_MIN; a quiet stand has no |omega| structure and scores
  0.05-0.2. A voter more than WIN_AGREE_S from the median lag is dropped: on this robot
  the confident wrong answers sit ~0.9 s away, one standing step period (0.905 s), which a
  line through a handful of points would otherwise bend toward. A rate is fitted only when
  at least 3 windows span more than WIN_S; t_wall and Vicon agree to ~100 ppm, so an
  offset alone is the right model for a short take.
  """
  if not len(win):
    return None
  tm, span, corr, lag = win.T
  vote = corr >= WIN_CORR_MIN
  if not vote.any():
    return None
  keep = vote & (np.abs(lag - np.median(lag[vote])) <= WIN_AGREE_S)
  w = np.sqrt(span[keep])
  if keep.sum() >= 3 and np.ptp(tm[keep]) > WIN_S:
    m, b = np.polyfit(tm[keep], lag[keep], 1, w=w)
  else:
    m, b = 0.0, float(np.average(lag[keep], weights=w))
  res = lag[keep] - (b + m * tm[keep])
  return dict(b=float(b), m=float(m), keep=keep, kept=int(keep.sum()), voting=int(vote.sum()),
              total=len(win), spread_ms=float(np.ptp(res) * 1e3),
              xcorr=float(np.average(corr[keep], weights=span[keep])))


def procrustes_rotation(a, b):
  """R minimising ||a - b R^T||, i.e. a ~ R b sample by sample. Orthogonal Procrustes.

  The reflection is forced out with diag(1, 1, det): the unconstrained SVD solution is the
  best ORTHOGONAL matrix, which for noisy near-degenerate data can come back with det = -1,
  a mirror. A mirror is not a frame relationship any two rigid bodies can have, and it would
  flip the sign of one output axis without any other symptom.
  """
  M = np.asarray(a).T @ np.asarray(b)
  U, S, Vt = np.linalg.svd(M)
  d = float(np.sign(np.linalg.det(U @ Vt)))
  return U @ np.diag([1.0, 1.0, d]) @ Vt, S, d


def pivot_lever(t, p_w, R_wt, still, win_s=PIVOT_WIN_S, step_s=0.1):
  """Lever arm r fitted from the mocap alone: pivot calibration over zero-command segments.

  The pelvis origin is rigid with the torso (they share it, h1_2_kinematics.md), so the
  marker cluster sits at p_w = p_pelvis + R_wt r. While the robot is not asked to translate,
  p_pelvis barely moves and every attitude change swings the cluster around it, which is what
  pins r. Each `win_s` window is detrended against [1, t] first (both sides, Frisch-Waugh), so
  a pelvis that drifts at a constant velocity inside a window costs nothing.

  What it cannot see: r_z enters only through roll/pitch, while r_x/r_y also enter through
  yaw. A standing robot does not tilt about its pelvis, it ROCKS ABOUT ITS FEET, so the pelvis
  moves with the tilt exactly as a longer vertical lever would. Measured on the eleven
  aligned 2026-09-16 Runs: r_z = 0.64-0.93 m against a taped 0.32, SE 0.04-0.22.
  r_x/r_y are no better as a check: over the 25 aligned Runs of 2026-09-14/16, r_y ranges
  -0.24..+0.27 m with BOTH signs, which one physical mount cannot do, and a 3-SE warning fired
  on 5 of them. So the whole result is a report, never a warning. A marker on the pelvis
  origin for one take is the way to measure the lever.
  The SE is a leave-one-window-out jackknife, not the textbook formula: consecutive samples
  are strongly correlated (stride-rate sway), and the formula SE would be far too small.

  Returns dict(r, se, windows, seconds, sv) or None when fewer than 3 windows qualify.
  """
  dt = float(np.median(np.diff(t)))
  sub = np.arange(0, len(t), max(int(round(step_s / dt)), 1))
  t, p_w, R_wt, still = t[sub], p_w[sub], R_wt[sub], still[sub]
  still = still & np.isfinite(p_w).all(1) & np.isfinite(R_wt).all((1, 2))
  runs = np.cumsum(np.r_[True, still[1:] != still[:-1]])      # contiguous stretches
  key = runs * 1_000_000 + np.floor((t - t[0]) / win_s).astype(int)
  AtA, Atb, A_all = [], [], []
  for k in np.unique(key[still]):
    i = np.flatnonzero(key == k)
    if len(i) < 5:
      continue
    X = np.column_stack([np.ones(len(i)), t[i] - t[i].mean()])
    detrend = lambda Z: Z - X @ np.linalg.lstsq(X, Z, rcond=None)[0]
    A = detrend(R_wt[i].reshape(len(i), 9)).reshape(-1, 3)   # rows (sample, axis) x cols r_j
    b = detrend(p_w[i]).reshape(-1)
    AtA.append(A.T @ A)
    Atb.append(A.T @ b)
    A_all.append(A)
  G = len(AtA)
  if G < 3:
    return None
  S_AA, S_Ab = sum(AtA), sum(Atb)
  r = np.linalg.solve(S_AA, S_Ab)
  jk = np.stack([np.linalg.solve(S_AA - AtA[g], S_Ab - Atb[g]) for g in range(G)])
  se = np.sqrt((G - 1) / G * ((jk - jk.mean(0)) ** 2).sum(0))
  A_all = np.vstack(A_all)
  return dict(r=r, se=se, windows=G, seconds=len(A_all) / 3 * step_s,
              sv=np.linalg.svd(A_all, compute_uv=False))


def _angles(W, P, mask):
  """Per-sample angle (deg) between rows of W and P where `mask`."""
  a, b = W[mask], P[mask]
  return np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1)
                                                        * np.linalg.norm(b, axis=1) + 1e-12),
                                      -1, 1)))


def gravity_tilt(acc, gyro, up_t, mask):
  """(deg, rows): angle between the accelerometer's mean direction and the torso-frame 'up'
  that mocap attitude + R predict, over quasi-static rows. (nan, rows) if too few.

  An independent check on R from a sensor the fit never saw: at rest the accelerometer reads
  +g along 'up', so R's roll and pitch are pinned by gravity. It says nothing about the mount's
  YAW (rotation about 'up'), which is the component that leaks vx into vy; that part is
  covered by the per-window R spread. Measured 2026-09-14/16: 0.2-1.8 deg on 22 Runs, a floor
  set by accelerometer bias and the chip's own mounting; T15 read 4.1, the same Run the
  window spread flagged. The walking part of the specific force is only 3-8% of g, too little
  to check yaw with (tried: 0.4-10.6 deg, uncorrelated with the gyro spread).
  """
  still = (mask & (np.linalg.norm(gyro, axis=1) < TILT_STILL_GYRO)
           & (np.abs(np.linalg.norm(acc, axis=1) - GRAVITY) < TILT_STILL_ACC))
  n = int(still.sum())
  if n < TILT_MIN_ROWS:
    return float("nan"), n
  a, u = acc[still].mean(0), up_t[still].mean(0)
  return float(np.degrees(np.arccos(np.clip(a @ u / (np.linalg.norm(a) * np.linalg.norm(u)),
                                            -1, 1)))), n


def align(bundle, mocap_path, lever, xcorr_floor=PAIR_CORR_FLOOR,
          proc_angle_max=PROC_ANGLE_MAX, min_excitation=MIN_EXCITATION, smooth=0.0,
          write=True, quiet=False, coarse_seed=None, use_wall=True, erode_s=GAP_ERODE_S,
          **reader_kw):
  """Full pipeline for one bundle. Returns the result dict; writes the csv and run.json.

  `use_wall=False` and `erode_s=0` exist for the self-test, which proves both are needed.
  """
  bundle = Path(bundle)
  run_path = bundle / "run.json"
  if not run_path.exists():
    raise SystemExit(f"{bundle}: no run.json. Bundle the run first "
                     f"(scripts/bundle_hardware_run.py --labels ...).")
  run = json.loads(run_path.read_text())
  flight = bundle / (run.get("flight_recorder") or "")
  if not flight.exists():
    cands = [p for p in sorted(bundle.glob("*.csv"))
             if not p.name.startswith("all_joints") and "_hrl" not in p.name
             and p.name != OUT_NAME]
    if len(cands) != 1:
      raise SystemExit(f"{bundle}: cannot identify the flight recorder slice "
                       f"(run.json says {run.get('flight_recorder')!r}, glob found {cands})")
    flight = cands[0]

  m = read_mocap(mocap_path, quiet=quiet, **reader_kw)
  t_m, R_wm, p_w = m["t"], m["R"], m["p"]
  clean = clean_mask(t_m, m["bad_t"], erode_s)
  t_f, t_wall, gyro, psi, cmd, acc = read_flight(flight)
  flags = []
  if use_wall and t_wall is not None:
    t_c, clock = t_wall, "t_wall"
  else:
    t_c, clock = t_f, "t (tick counter)"
    if use_wall:
      flags.append("no t_wall column: aligned on the tick counter, which bends")
  if not quiet:
    print(f"\n  flight slice  {flight.name}  ({len(t_f)} rows, {t_f[-1] - t_f[0]:.1f} s, "
          f"entry {run.get('entry')});  clock {clock}")
    print(f"  mocap clean   {clean.mean() * 100:.1f}% of kept rows "
          f"(>= {erode_s:.2f} s from every dropout; {len(m['bad_t'])} rows refused)")

  # ---- step 1: clocks, from |omega| alone (invariant to the unknown frame) ----------
  # Prefer the export's own angular-rate triple over differentiating the attitude, rotated
  # WORLD -> BODY through R first (see read_flat_csv). Differentiating 100 Hz attitude turns
  # sub-millimetre marker noise into ~0.7 rad/s of rate; measured frame residual on TRIAL05
  # is 15.8 deg from quaternions against 7.5 deg from the exported triple.
  if m.get("w_world") is not None:
    w_m = np.einsum("nji,nj->ni", R_wm, m["w_world"])   # R^T w : world -> body
    w_src = "exported rates, rotated world->body"
  else:
    w_m = body_rates(R_wm, t_m)
    w_src = "differentiated attitude (no rate columns in the export)"
  if not quiet:
    print(f"  omega source  {w_src}")
  sig_m = _boxcar(np.linalg.norm(w_m, axis=1), t_m)
  sig_f = _boxcar(np.linalg.norm(gyro, axis=1), t_c)
  if coarse_seed is not None:
    gl = (float(coarse_seed), float("nan"))
  else:
    gl = global_lag(t_c, sig_f, t_m, sig_m, clean)
  if gl is None:
    raise SystemExit("TIME ALIGNMENT FAILED: the take has fewer than 1 s of clean samples. "
                     "Nothing written.")
  win = window_lags(t_c, sig_f, t_m, sig_m, clean, gl[0])
  al = fit_windows(win)
  if al is None:
    raise SystemExit(
      f"TIME ALIGNMENT REJECTED: none of {len(win)} clean windows reached |omega| "
      f"correlation {WIN_CORR_MIN} near the whole-take offset {gl[0]:+.2f} s (corr "
      f"{gl[1]:.2f}).\n  The take and this run do not describe the same motion, the robot "
      f"stood still for all of the clean part, or --rot-format/--rate is wrong. "
      f"Nothing written.")
  b, mrate = al["b"], al["m"]
  if not quiet:
    seed = "--coarse-seed" if coarse_seed is not None else f"masked search, corr {gl[1]:.3f}"
    print(f"\n  [1] clock   whole-take offset {gl[0]:+.3f} s ({seed})")
    print(f"              {al['kept']} of {al['total']} windows kept ({al['voting']} reached "
          f"corr {WIN_CORR_MIN}), agreeing to {al['spread_ms']:.1f} ms, mean corr "
          f"{al['xcorr']:.3f}")
    print(f"              {clock} = t_mocap + ({b:+.4f} + {mrate * 1e6:+.1f} ppm * t_mocap)")
  if al["xcorr"] < xcorr_floor:
    raise SystemExit(
      f"TIME ALIGNMENT REJECTED: |omega| cross-correlation {al['xcorr']:.3f} < "
      f"{xcorr_floor:.2f}.\n  Nothing written.")
  if abs(mrate) > MAX_SKEW:
    raise SystemExit(
      f"TIME ALIGNMENT REJECTED: fitted rate {mrate * 1e6:+.0f} ppm exceeds "
      f"{MAX_SKEW * 1e6:.0f} ppm; a percent-level 'rate' means the fit absorbed a bad "
      f"offset into the slope.\n  Nothing written.")
  if use_wall and abs(mrate) > SKEW_WARN:
    flags.append(f"clock rate {mrate * 1e6:+.0f} ppm between t_wall and Vicon")

  # ---- step 2: frame, in closed form, on the clean aligned angular velocities --------
  # Smooth BOTH omega vectors identically before the fit, each on its OWN continuous grid (the
  # clean flight rows are not contiguous any more, so smoothing after selection would mix
  # across holes). The rotation is a property of the mount, not of any one sample, and the
  # per-sample angle between two noisy 3-vectors is noise wherever |omega| is small.
  tf_of_m = t_m + (b + mrate * t_m)                  # mocap samples, on the alignment clock
  cover = np.interp(t_c, tf_of_m, clean.astype(float), left=0.0, right=0.0) > 0.999
  if cover.sum() < 100:
    raise SystemExit(f"OVERLAP TOO SHORT: only {int(cover.sum())} flight rows fall inside a "
                     f"clean stretch of the take after alignment. Nothing written.")
  tc = t_c[cover]
  Wm = np.stack([_boxcar(w_m[:, i], t_m, FRAME_SMOOTH_S) for i in range(3)], 1)
  W = np.stack([np.interp(tc, tf_of_m, Wm[:, i]) for i in range(3)], 1)
  G = np.stack([_boxcar(gyro[:, i], t_c, FRAME_SMOOTH_S) for i in range(3)], 1)[cover]
  R, S, det = procrustes_rotation(W, G)
  pred = G @ R.T
  # Scored only where the robot really rotates. The old "above the median |omega|" rule
  # scored noise on standing-heavy Runs (median 0.003-0.03 rad/s on 2026-09-16) and failed
  # right answers at 16-46 deg; above 0.2 rad/s the same fits read 3.6-8.5 deg.
  wn = np.linalg.norm(W, axis=1)
  fast = wn > OMEGA_RESID_MIN
  scored = "|omega| > %.1f rad/s" % OMEGA_RESID_MIN
  if fast.sum() < 100:
    fast = wn > max(np.median(wn), 1e-6)
    scored = "the upper half of |omega| (too few fast samples)"
  med_ang = float(np.median(_angles(W, pred, fast))) if fast.any() else float("nan")
  rel = float(np.linalg.norm(W - pred) / max(np.linalg.norm(W), 1e-12))
  exc = float(S[2] / max(S[0], 1e-300))
  # Accuracy, not fit: the same Procrustes on each WIN_S stretch of clean samples, compared to
  # the pooled R. The median residual above barely moves for a wrong R (+~5 deg for a 10 deg
  # tilt, measured), while independent windows disagreeing is exactly what a bad take shows
  # (2026-09-16: 0.7-3 deg on 18 Runs, 5.9 on T15, whose marker fit error was 4 mm).
  wid = np.floor((tc - tc[0]) / WIN_S).astype(int)
  spread = [np.degrees(np.linalg.norm(so3_log(R.T @ procrustes_rotation(W[wid == j],
                                                                       G[wid == j])[0])))
            for j in np.unique(wid) if (fast & (wid == j)).sum() >= 200]
  frame_spread = float(np.median(spread)) if len(spread) >= 2 else float("nan")
  if len(spread) < 2:
    flags.append("frame accuracy unchecked: fewer than 2 windows with enough rotation")
  elif frame_spread > FRAME_SPREAD_MAX:
    flags.append(f"frame windows disagree by {frame_spread:.1f} deg (median) > "
                 f"{FRAME_SPREAD_MAX:.0f}: treat this ground truth as low quality")
  # Gravity: R_wm^T z is 'up' in the marker frame (third row of R_wm); R^T takes it to torso.
  tilt, tilt_n = float("nan"), 0
  if acc is not None:
    up_m = np.stack([np.interp(tc, tf_of_m, R_wm[:, 2, i]) for i in range(3)], 1)
    tilt, tilt_n = gravity_tilt(acc[cover], gyro[cover], up_m @ R, np.ones(len(tc), bool))
    if tilt > TILT_FLAG_DEG:
      flags.append(f"accelerometer gravity disagrees with mocap tilt by {tilt:.1f} deg > "
                   f"{TILT_FLAG_DEG:.0f}: roll/pitch of R suspect")
  rpy = np.degrees(so3_log(R))
  if not quiet:
    print(f"  [2] frame   R (torso -> mocap body), rotvec {np.array2string(rpy, precision=2)}"
          f" deg   det {det:+.0f}")
    print(f"              residual {med_ang:.2f} deg median over {scored} "
          f"({int(fast.sum())} samples), {rel * 100:.2f}% rel;  excitation sv "
          f"{S[0]:.3g}/{S[1]:.3g}/{S[2]:.3g} (ratio {exc:.3f})")
    print(f"              per-window R vs pooled: median {frame_spread:.1f} deg over "
          f"{len(spread)} windows" + (f", max {max(spread):.1f}" if spread else ""))
    if acc is None:
      print("              gravity check skipped: no acc_x/y/z columns")
    elif np.isnan(tilt):
      print(f"              gravity check skipped: {tilt_n} quasi-static rows "
            f"(< {TILT_MIN_ROWS})")
    else:
      print(f"              gravity check: accelerometer vs mocap tilt {tilt:.2f} deg over "
            f"{tilt_n} quasi-static rows (roll/pitch only, independent sensor)")
  if det < 0 and not quiet:
    print("              ⚠ the raw SVD solution was a REFLECTION; det forced to +1. Check "
          "the mocap axis convention before trusting this R.")
  if exc < min_excitation:
    raise SystemExit(
      f"FRAME FIT REJECTED: angular excitation ratio {exc:.4f} < {min_excitation}.\n"
      f"  The run rotated about essentially {'one' if S[1] / max(S[0], 1e-30) < 0.1 else 'two'}"
      f" axis/axes, so the rotation about the missing axis is unobservable and R is a guess "
      f"in that direction. Use a take that includes turning AND pitch/roll motion (a few "
      f"seconds of walking with turns is enough).\n  Nothing written.")
  if not np.isfinite(med_ang) or med_ang > proc_angle_max:
    raise SystemExit(
      f"FRAME FIT REJECTED: {med_ang:.1f} deg median residual between R*omega_torso and "
      f"omega_mocap (limit {proc_angle_max:.0f}).\n  No constant rotation maps one onto the "
      f"other, so the two signals are not the same rigid body's angular velocity. Most "
      f"likely causes: --rot-format is wrong (a helical export read as Euler, or vice "
      f"versa), the rigid body's segment was rebuilt mid-session, or the markers moved on "
      f"the robot.\n  Nothing written.")

  # ---- steps 3-4: linear velocity, body frame, lever arm ---------------------------
  # Differentiate on the MOCAP grid, before resampling: the recorder's grid is ~2.5x
  # denser and non-uniform, so differentiating after interpolation would differentiate a
  # staircase and report interpolation noise as acceleration. Samples next to a hole use a
  # one-sided difference across it, which is why `cover` stays GAP_ERODE_S away from them.
  v_w = np.gradient(p_w, t_m, axis=0)
  if smooth > 0:
    v_w = np.stack([_boxcar(v_w[:, i], t_m, smooth) for i in range(3)], 1)
  R_wt = R_wm @ R                                     # robot torso attitude, mocap world
  yaw = np.unwrap(np.arctan2(R_wt[:, 1, 0], R_wt[:, 0, 0]))
  wz = np.gradient(yaw, t_m)
  v_t = np.einsum("nji,nj->ni", R_wt, v_w)            # world -> torso frame
  w_t = w_m @ R                                       # omega, torso frame (= R^T w_m)
  v_t = v_t - np.cross(w_t, lever)                    # marker cluster -> pelvis
  p_pelvis = p_w - np.einsum("nij,j->ni", R_wt, lever)

  # ---- step 5: the measured lever arm, checked against the mocap alone --------------
  piv = None
  if cmd is not None:
    inside = (tf_of_m >= t_c[0]) & (tf_of_m <= t_c[-1])
    cmd_m = np.stack([np.interp(tf_of_m, t_c, cmd[:, i]) for i in range(2)], 1)
    piv = pivot_lever(t_m, p_w, R_wt,
                      clean & inside & (np.abs(cmd_m) < PIVOT_CMD_EPS).all(1))

  def onto_flight(x):
    """Onto the recorder's rows (keyed by its own `t`); NaN outside clean mocap."""
    y = np.full((len(t_f),) + x.shape[1:], np.nan)
    if x.ndim == 1:
      y[cover] = np.interp(tc, tf_of_m, x)
    else:
      for i in range(x.shape[1]):
        y[cover, i] = np.interp(tc, tf_of_m, x[:, i])
    return y

  v = onto_flight(v_t)
  # v_P^P = Rz(psi) (v_I^T - omega_T x r), h1_2_kinematics.md; psi is already on this grid.
  c, s = np.cos(psi), np.sin(psi)
  v = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
  out = dict(t=t_f, v=v, wz=onto_flight(wz), p=onto_flight(p_pelvis), yaw=onto_flight(yaw),
             cover=cover, align=al, global_lag=gl, map=(b, mrate), R=R, flags=flags,
             proc=dict(median_angle_deg=med_ang, rel_resid=rel, excitation=exc, det=det,
                       sv=S.tolist(), spread_deg=frame_spread, tilt_deg=tilt,
                       tilt_rows=tilt_n), pivot=piv)
  n_ok = int(np.isfinite(v).all(axis=1).sum())
  if not quiet:
    p50, p95, pmx = np.percentile(np.abs(psi), [50, 95]).tolist() + [float(np.abs(psi).max())]
    print(f"  [3] motion  |v| median {np.nanmedian(np.linalg.norm(v, axis=1)):.3f} m/s, "
          f"yaw rate p95 {np.nanpercentile(np.abs(out['wz']), 95):.3f} rad/s")
    print(f"  [4] lever   r = {np.array2string(np.asarray(lever), precision=3)} m  "
          f"(pelvis -> marker cluster, body frame); |omega x r| median "
          f"{np.median(np.linalg.norm(np.cross(w_t, lever), axis=1)):.3f} m/s")
    print(f"      waist   |psi| p50 {p50:.4f} p95 {p95:.4f} max {pmx:.4f} rad (meas_q12)")
    if p95 > PSI_WARN:
      # Measured across the 11 bundles of 2026-09-14: p95 0.03-0.25 rad. The waist is NOT
      # rigidly held on A1a runs, so Rz(psi) is load-bearing rather than cosmetic -- and so
      # is the assumption that the marker cluster is rigid with the TORSO (the link the gyro
      # is on). A pelvis-mounted cluster is not silently wrong here: with the waist moving,
      # no constant R maps its rate onto the torso gyro, and the Procrustes residual above
      # is what says so.
      print(f"              ⚠ the waist MOVES (p95 {p95:.3f} rad), so Rz(psi) is doing real "
            f"work and the cluster is assumed rigid with the TORSO. If the residual in [2] "
            f"is large, check whether it is actually mounted on the pelvis.")
    print(f"      coverage {n_ok}/{len(t_f)} flight rows ({100.0 * n_ok / len(t_f):.1f}%) "
          f"have clean ground truth; the rest are NaN")
    if piv is None:
      print("  [5] pivot   skipped: " + ("no cmd_vx/cmd_vy columns" if cmd is None else
            "fewer than 3 zero-command windows inside the take"))
    else:
      gap = piv["r"] - np.asarray(lever)
      print(f"  [5] pivot   r fitted {np.array2string(piv['r'], precision=3)} m "
            f"+- {np.array2string(piv['se'], precision=3)} (jackknife) vs measured "
            f"{np.array2string(np.asarray(lever), precision=3)};  {piv['windows']} windows, "
            f"{piv['seconds']:.0f} s of zero-command data")
      print(f"              report only, never a check (see pivot_lever); gap to measured "
            f"{np.array2string(gap, precision=3)} m")
    for fl in flags:
      print(f"  ⚠ FLAG      {fl}")

  if write:
    cols = np.column_stack([t_f, v, out["wz"], out["p"], out["yaw"]])
    np.savetxt(bundle / OUT_NAME, cols, delimiter=",", fmt="%.6f",
               header="t,gt_vx,gt_vy,gt_vz,gt_wz,gt_px,gt_py,gt_pz,gt_yaw", comments="")
    run["mocap"] = Path(mocap_path).name
    run["mocap_align"] = {
      "clock": clock,
      "xcorr": round(al["xcorr"], 4),                 # length-weighted, kept windows
      "windows": f"{al['kept']}/{al['voting']}/{al['total']}",   # kept/voting/total
      "lag_spread_ms": round(al["spread_ms"], 2),
      "lag_s": round(float(b), 4),
      "skew_ppm": round(float(mrate) * 1e6, 1),
      "global_lag_s": round(gl[0], 3),
      "mocap_clean_frac": round(float(clean.mean()), 4),
      "lever": [float(x) for x in lever],
      "R": [[float(x) for x in row] for row in R],
      "proc_angle_deg": round(med_ang, 3),
      "frame_spread_deg": None if np.isnan(frame_spread) else round(frame_spread, 2),
      "gravity_tilt_deg": None if np.isnan(tilt) else round(tilt, 2),
      "gravity_rows": tilt_n,
      "excitation": round(exc, 4),
      "n_samples": n_ok,
      "coarse_seed_s": coarse_seed,
      "pivot": piv and {"r": [round(float(x), 4) for x in piv["r"]],
                        "se": [round(float(x), 4) for x in piv["se"]],
                        "windows": piv["windows"], "seconds": round(piv["seconds"], 1)},
      "flags": flags,
    }
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    if not quiet:
      print(f"\n  wrote {bundle / OUT_NAME}  and updated run.json")
  return out


# --------------------------------------------------------------------------- self-test

def _write_vicon(path, t, R, p_mm, rate, fmt, obj="H1_2:torso", frames=None):
  """Synthesise a Nexus-shaped ASCII export, so the self-test goes through `read_vicon`.

  `frames` (default 1..N) lets a caller leave holes, which is how a dropout looks on disk.
  """
  if fmt == "quat":
    comps, units = ["RX", "RY", "RZ", "RW"], ["", "", "", ""]
    q = []
    for M in R:
      w = np.sqrt(max(1.0 + np.trace(M), 0.0)) / 2.0
      w = max(w, 1e-9)
      q.append([(M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w),
                (M[1, 0] - M[0, 1]) / (4 * w), w])
    rot = np.array(q)
  elif fmt == "helical":
    comps, units = ["RX", "RY", "RZ"], ["deg"] * 3
    rot = np.degrees(np.stack([so3_log(M) for M in R]))
  else:
    comps, units = ["RX", "RY", "RZ"], ["deg"] * 3
    rot = np.degrees(np.stack([_euler_xyz_of(M) for M in R]))
  names = ["", ""] + [obj] + [""] * (len(comps) + 2)
  with open(path, "w") as fh:
    fh.write("Segments\n")
    fh.write(f"{rate:g}\n")
    fh.write(",".join(names) + "\n")
    fh.write(",".join(["Frame", "Sub Frame"] + comps + ["TX", "TY", "TZ"]) + "\n")
    fh.write(",".join(["", ""] + units + ["mm"] * 3) + "\n")
    frames = np.arange(1, len(t) + 1) if frames is None else frames
    for k in range(len(t)):
      fh.write(",".join([str(int(frames[k])), "0"]
                        + [f"{x:.9f}" for x in rot[k]]
                        + [f"{x:.6f}" for x in p_mm[k]]) + "\n")


def _euler_xyz_of(M):
  """Inverse of the euler_xyz convention in `rot_matrices`: R = Rx(a) Ry(b) Rz(c)."""
  b = np.arcsin(np.clip(M[0, 2], -1, 1))
  a = np.arctan2(-M[1, 2], M[2, 2])
  c = np.arctan2(-M[0, 1], M[0, 0])
  return np.array([a, b, c])


def _orth(M):
  U, _, Vt = np.linalg.svd(M)
  return U @ Vt


def selftest():
  """Plant a clock skew, a frame rotation and a lever arm; check all three come back.

  Everything is generated in FLIGHT time and then observed on a mocap clock that is
  deliberately offset and running at the wrong rate, so the pipeline has to undo exactly
  what a real capture does to it. The synthetic Vicon file is written to disk in the Nexus
  ASCII layout and read back through `read_vicon`, so the format sniffer is under test too,
  not bypassed.
  """
  rng = np.random.default_rng(7)
  fails = 0
  T, dt = 80.0, 0.001
  tt = np.arange(0.0, T, dt)
  # FOUR WALKING BOUTS separated by stands, not one continuous wiggle. This is what a
  # joystick session looks like, and it is what `fit_alignment` is built around: its
  # stride-period alias unwrap seeds the line from the best-correlated windows, which are
  # the ones straddling a start/stop (the aperiodic envelope breaks the alias). Fed a single
  # uninterrupted quasi-periodic bout instead, it locked the offset a whole 8.6 s away with
  # xcorr 0.74 -- a real property of the fitter, reproduced here rather than tuned around.
  env = np.clip(sum(1.0 / (1.0 + np.exp(-(tt - a) / 0.8)) / (1.0 + np.exp((tt - b) / 0.8))
                    for a, b in ((6, 22), (30, 48), (55, 64), (70, 78))), 0.0, 1.0)
  w_true = env[:, None] * np.stack([
    0.45 * np.sin(2 * np.pi * 1.71 * tt),
    0.35 * np.sin(2 * np.pi * 2.33 * tt + 0.9),
    0.80 * np.sin(2 * np.pi * 0.87 * tt + 2.1)], 1)
  v_true = env[:, None] * np.stack([
    0.55 + 0.25 * np.sin(2 * np.pi * 0.11 * tt),
    0.12 * np.sin(2 * np.pi * 0.17 * tt + 0.7),
    0.03 * np.sin(2 * np.pi * 0.29 * tt)], 1)
  R_wt = np.zeros((len(tt), 3, 3))
  R_wt[0] = so3_exp(np.array([0.05, -0.03, 0.4]))
  for k in range(1, len(tt)):
    R_wt[k] = R_wt[k - 1] @ so3_exp(w_true[k - 1] * dt)
  v_world = np.einsum("nij,nj->ni", R_wt, v_true)
  # accelerometer truth: specific force in the torso frame, +g along 'up' at rest
  f_true = np.einsum("nji,nj->ni", R_wt,
                     np.gradient(v_world, tt, axis=0) + np.array([0.0, 0.0, GRAVITY]))
  p_pel = np.cumsum(v_world * dt, axis=0)

  R_plant = _orth(rng.normal(size=(3, 3)))
  if np.linalg.det(R_plant) < 0:
    R_plant[:, 2] *= -1
  lever = np.array([0.20, 0.03, 0.0])                  # yaw -> lateral leak, by construction
  p_mark = p_pel + np.einsum("nij,j->ni", R_wt, lever)
  R_wm = R_wt @ R_plant.T                              # what the mocap rigid body reports

  # The mocap take starts 12.3 s BEFORE the operator enters the RL state, which is the
  # normal order of events and the reason the offset is large. Its own clock starts at its
  # first frame (the reader re-zeros on Frame[0]), so b_plant is the lag of that first frame
  # against the recorder's t_wall = 0. Samples whose flight time is negative are kept: the
  # truth interpolation clamps there, i.e. the robot stands still before the run, which is
  # exactly what the mocap sees and is what the coverage mask then has to exclude.
  # Vicon and t_wall are two real clocks (+40 ppm). The recorder's `t` is a tick counter:
  # 0.45% slow and bent by +-80 ms, the shape measured on 2026-09-16.
  b_plant, m_plant, rate = -12.30, 40e-6, 200.0
  t_mocap = np.arange(0.0, 83.0, 1.0 / rate)
  tf_m = t_mocap + (b_plant + m_plant * t_mocap)       # t_wall of each mocap sample
  keep = tf_m <= tt[-1]
  t_mocap, tf_m = t_mocap[keep], tf_m[keep]
  # element-wise interpolation of the attitude, then re-orthonormalised (the 1 kHz truth
  # grid makes the chord error ~1e-7 rad, far below every tolerance asserted below)
  Rm = _orth(np.transpose(np.array([[np.interp(tf_m, tt, R_wm[:, i, j]) for j in range(3)]
                                    for i in range(3)]), (2, 0, 1)))
  pm = np.stack([np.interp(tf_m, tt, p_mark[:, i]) for i in range(3)], 1)
  # Two dropouts inside walking bouts (the robot leaving the volume), each preceded by 60 ms
  # of garbage the export's filter smeared in: attitude 25 deg off, position 4 cm off. Only
  # the GAP_ERODE_S margin keeps those rows out.
  gaps = [(25.0, 28.0), (58.0, 59.0)]
  frames = np.arange(1, len(t_mocap) + 1)
  hole = np.zeros(len(t_mocap), bool)
  for g0, g1 in gaps:
    hole |= (t_mocap >= g0) & (t_mocap < g1)
    smear = (t_mocap >= g0 - 0.06) & (t_mocap < g0)
    Rm[smear] = Rm[smear] @ so3_exp(np.deg2rad([25.0, 0.0, 0.0]))
    pm[smear] += 0.04
  t_mocap, tf_m, Rm, pm, frames = (t_mocap[~hole], tf_m[~hole], Rm[~hole], pm[~hole],
                                   frames[~hole])

  t_f = np.cumsum(rng.uniform(0.0018, 0.0022, 50000))  # ~500 Hz, non-uniform, like the real log
  t_f = t_f - t_f[0]
  t_f = t_f[t_f <= tt[-1]]                              # this is t_wall
  t_tick = t_f * (1 - 4500e-6) + 0.08 * np.sin(2 * np.pi * t_f / 50.0)
  gyro = np.stack([np.interp(t_f, tt, w_true[:, i]) for i in range(3)], 1)
  acc = np.stack([np.interp(t_f, tt, f_true[:, i]) for i in range(3)], 1)

  print("planted: b = %.3f s, skew = %+.0f ppm (t_wall), tick counter -4500 ppm +-80 ms, "
        "dropouts %s s, lever = %s m, R = rotvec %s deg"
        % (b_plant, m_plant * 1e6, gaps, lever.tolist(),
           np.array2string(np.degrees(so3_log(R_plant)), precision=2)))

  with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # (0) the reader itself: all three rotation formats must round-trip to the same R
    # euler_xyz is the one variant the sniffer CANNOT detect -- it is three degree columns
    # named exactly like a helical export -- so it is checked with the override that exists
    # for precisely that case, and the other two are checked with no override at all.
    print("\n-- (0) format sniffer round-trip")
    for fmt, override in (("helical", None), ("quat", None), ("euler_xyz", "euler_xyz")):
      f = td / f"vicon_{fmt}.csv"
      _write_vicon(f, t_mocap[:500], Rm[:500], pm[:500] * 1e3, rate, fmt)
      got = read_vicon(f, quiet=True, rot_format=override)
      e = max(np.degrees(np.linalg.norm(so3_log(A.T @ B)))
              for A, B in zip(Rm[:500], got["R"]))
      ep = np.abs(got["p"] - pm[:500]).max()
      dt_e = np.abs(np.diff(got["t"]) - 1.0 / rate).max()
      print(f"  {fmt:10s} {'sniffed' if override is None else 'override'} -> "
            f"{got['info']['rot_format']:10s} {got['info']['pos_units']}->m   "
            f"max attitude err {e:.2e} deg, max position err {ep:.2e} m, "
            f"dt err {dt_e:.1e} s")
      # tolerances sit just below the synthetic file's own print precision (1e-9 deg /
      # 1e-6 mm); they check that the FORMAT is decoded, not the writer's rounding
      fails += e > 1e-5 or ep > 1e-8 or dt_e > 1e-12
      fails += got["info"]["rot_format"] != fmt

    # the bundle: a flight-recorder slice with only the columns this tool reads
    bundle = td / "2026_09_14-_00_00_SELFTEST"
    bundle.mkdir()
    fcsv = bundle / "selftest_flight.csv"
    with open(fcsv, "w") as fh:
      fh.write("t,t_wall,est_gyro_x,est_gyro_y,est_gyro_z,meas_q12,acc_x,acc_y,acc_z\n")
      for k in range(len(t_f)):
        fh.write(f"{t_tick[k]:.4f},{t_f[k]:.4f},{gyro[k, 0]:.6f},{gyro[k, 1]:.6f},"
                 f"{gyro[k, 2]:.6f},0.0,{acc[k, 0]:.5f},{acc[k, 1]:.5f},{acc[k, 2]:.5f}\n")
    (bundle / "run.json").write_text(json.dumps(
      {"policy_tag": "SELFTEST", "flight_recorder": fcsv.name, "entry": 0,
       "mocap": None}, indent=2))
    vfile = td / "vicon_selftest.csv"
    _write_vicon(vfile, t_mocap, Rm, pm * 1e3, rate, "helical", frames=frames)

    print("\n-- (1..4) full pipeline")
    out = align(bundle, vfile, lever, quiet=False)
    al, R = out["align"], out["R"]
    b_got, m_got = out["map"][0], out["map"][1]

    print("\n-- recovery")
    e_b = abs(b_got - b_plant)
    e_m = abs(m_got - m_plant) * 1e6
    e_R = float(np.degrees(np.linalg.norm(so3_log(R.T @ R_plant))))
    print(f"  clock offset   {b_got:+.4f} s  vs planted {b_plant:+.4f}  -> err {e_b * 1e3:.2f} ms "
          f"  (whole-take search {out['global_lag'][0]:+.3f}; windows {al['kept']}/"
          f"{al['voting']}/{al['total']}, agreeing to {al['spread_ms']:.2f} ms)")
    print(f"  clock skew     {m_got * 1e6:+.1f} ppm vs planted {m_plant * 1e6:+.1f} -> "
          f"err {e_m:.1f} ppm")
    print(f"  frame R        angle to planted {e_R:.4f} deg (residual "
          f"{out['proc']['median_angle_deg']:.3f} deg median, windows agree to "
          f"{out['proc']['spread_deg']:.3f} deg)")
    fails += e_b > 0.005
    fails += e_m > 100.0
    fails += e_R > 1.0
    fails += len(out["flags"]) > 0
    tilt = out["proc"]["tilt_deg"]
    print(f"  gravity check  accelerometer vs mocap tilt {tilt:.3f} deg over "
          f"{out['proc']['tilt_rows']} quasi-static rows")
    fails += not (tilt < 0.5)

    # (5d) the gravity check bites: the same rows with the accelerometer tilted 5 deg
    k = out["cover"]
    R_wt_f = _orth(np.transpose(np.array([[np.interp(t_f[k], tt, R_wt[:, i, j])
                                           for j in range(3)] for i in range(3)]), (2, 0, 1)))
    up_t = R_wt_f[:, 2, :]                                # true torso-frame 'up'
    tilted = acc[k] @ so3_exp(np.deg2rad([5.0, 0.0, 0.0])).T
    t5, n5 = gravity_tilt(tilted, gyro[k], up_t, np.ones(k.sum(), bool))
    t0, _ = gravity_tilt(acc[k], gyro[k], up_t, np.ones(k.sum(), bool))
    print(f"\n-- (5d) gravity check on a 5 deg tilted accelerometer: {t5:.2f} deg "
          f"(untilted {t0:.3f}, {n5} rows; flag above {TILT_FLAG_DEG:.0f})")
    fails += not (4.5 < t5 < 5.5 and t5 > TILT_FLAG_DEG and t0 < 0.5)

    # velocity vs truth, read back from the csv this run actually wrote. Its `t` column is
    # the recorder's tick counter; the truth lives on t_wall, which is row-aligned (t_f).
    got = np.genfromtxt(bundle / OUT_NAME, delimiter=",", names=True)
    key_ok = np.allclose(got["t"], np.round(t_tick, 4), atol=1e-6)
    print(f"  output keyed by the recorder's own `t` rows (tick counter): {key_ok}")
    fails += not key_ok
    ok = np.isfinite(got["gt_vx"])
    tw_ok = t_f[ok]
    ref = np.stack([np.interp(tw_ok, tt, v_true[:, i]) for i in range(3)], 1)
    err = np.stack([got[f"gt_v{a}"][ok] for a in "xyz"], 1) - ref
    rms = float(np.sqrt((err ** 2).sum(1).mean()))
    print(f"  body velocity  rms |err| {rms * 1e3:.2f} mm/s, max {np.abs(err).max() * 1e3:.2f}"
          f" mm/s over {int(ok.sum())} rows  (|v| ~ {np.linalg.norm(ref, axis=1).mean():.2f} m/s)")
    fails += rms > 0.010
    # position and yaw come from the same map; a sign error in either shows up here
    ref_p = np.stack([np.interp(tw_ok, tt, p_pel[:, i]) for i in range(3)], 1)
    e_p = float(np.abs(np.stack([got[f"gt_p{a}"][ok] for a in "xyz"], 1) - ref_p).max())
    ref_wz = np.interp(tw_ok, tt, w_true[:, 2])   # yaw rate, level-ish flight
    print(f"  pelvis position max err {e_p * 1e3:.2f} mm;  gt_wz vs body yaw rate "
          f"corr {np.corrcoef(got['gt_wz'][ok], ref_wz)[0, 1]:.4f}")
    fails += e_p > 0.005
    # a dropout is NaN in the output, never a straight line drawn through it
    in_gap = np.zeros(len(t_f), bool)
    for g0, g1 in gaps:
      in_gap |= (t_f >= g0 + b_plant) & (t_f < g1 + b_plant)
    print(f"  dropout rows   {int(in_gap.sum())} flight rows fall in a mocap hole, "
          f"{int(np.isfinite(got['gt_vx'][in_gap]).sum())} of them carry a value")
    fails += np.isfinite(got["gt_vx"][in_gap]).any() or not in_gap.any()

    # (5b) each new piece is load-bearing: the tick counter, and the margin around a dropout
    print("\n-- (5b) without the fixes")
    for name, kw in (("tick-counter clock", dict(use_wall=False)),
                     ("no dropout margin", dict(erode_s=0.0))):
      try:
        o = align(bundle, vfile, lever, quiet=True, write=False, **kw)
        k = np.isfinite(o["v"]).all(1)
        rr = np.stack([np.interp(t_f[k], tt, v_true[:, i]) for i in range(3)], 1)
        r2 = float(np.sqrt(((o["v"][k] - rr) ** 2).sum(1).mean()))
        print(f"  {name:20s} rms |err| {r2 * 1e3:.1f} mm/s ({r2 / rms:.0f}x)")
        fails += r2 < 3 * rms
      except SystemExit as e:
        print(f"  {name:20s} refused: {str(e).splitlines()[0]}")

    # (5c) a take LONGER than its slice (the mocap ran on into the next entry): only the
    # overlap may be scored, or the overhang dilutes the true offset below a wrong one
    short = td / "2026_09_14-_00_01_SHORT"
    short.mkdir()
    sel = (t_f >= 30.0) & (t_f < 50.0)
    with open(short / "short_flight.csv", "w") as fh:
      fh.write("t,t_wall,est_gyro_x,est_gyro_y,est_gyro_z,meas_q12,acc_x,acc_y,acc_z\n")
      for k in np.flatnonzero(sel):
        fh.write(f"{t_tick[k]:.4f},{t_f[k]:.4f},{gyro[k, 0]:.6f},{gyro[k, 1]:.6f},"
                 f"{gyro[k, 2]:.6f},0.0,{acc[k, 0]:.5f},{acc[k, 1]:.5f},{acc[k, 2]:.5f}\n")
    (short / "run.json").write_text(json.dumps(
      {"policy_tag": "SELFTEST", "flight_recorder": "short_flight.csv", "entry": 0}))
    print("\n-- (5c) take 4x longer than its flight slice")
    try:
      o = align(short, vfile, lever, quiet=True, write=False)
      e_s = abs(o["map"][0] - b_plant)
      print(f"  offset {o['map'][0]:+.4f} s (whole-take search {o['global_lag'][0]:+.3f}), "
            f"err {e_s * 1e3:.1f} ms")
      fails += e_s > 0.005
    except SystemExit as e:
      print(f"  REFUSED: {str(e).splitlines()[0]}")
      fails += 1

    # (5) the lever arm: re-run with r = 0 and check the contamination is exactly omega x r
    print("\n-- (5) lever-arm correction")
    out0 = align(bundle, vfile, np.zeros(3), quiet=True, write=False)
    err0 = out0["v"][ok] - ref
    rms0 = float(np.sqrt((err0 ** 2).sum(1).mean()))
    w_body = np.stack([np.interp(tw_ok, tt, w_true[:, i]) for i in range(3)], 1)
    pred = np.cross(w_body, lever)
    slope = float(np.polyfit(w_body[:, 2], err0[:, 1], 1)[0])
    corr = float(np.corrcoef(err0[:, 1], w_body[:, 2])[0, 1])
    print(f"  uncorrected    rms |err| {rms0 * 1e3:.1f} mm/s  ({rms0 / max(rms, 1e-9):.0f}x "
          f"the corrected {rms * 1e3:.2f} mm/s)")
    print(f"  the residue IS omega x r: max|err0 - omega x r| "
          f"{np.abs(err0 - pred).max() * 1e3:.2f} mm/s")
    print(f"  yaw leaks into vy: slope d(err_vy)/d(wz) = {slope:+.4f} m  vs r_x = "
          f"{lever[0]:+.3f} m   (corr {corr:+.4f})")
    fails += not (rms0 > 20 * rms)
    fails += abs(slope - lever[0]) > 0.005
    fails += abs(corr) < 0.99
    fails += np.abs(err0 - pred).max() > 0.010

    # (6) the gates actually bite
    print("\n-- (6) fail-closed gates")
    for name, kw, needle in (
        ("xcorr floor", dict(xcorr_floor=1.01), "TIME ALIGNMENT REJECTED"),
        ("procrustes residual", dict(proc_angle_max=-1.0), "FRAME FIT REJECTED"),
        ("angular excitation", dict(min_excitation=1.01), "FRAME FIT REJECTED")):
      try:
        align(bundle, vfile, lever, quiet=True, write=False, **kw)
        print(f"  {name:22s} DID NOT FIRE")
        fails += 1
      except SystemExit as e:
        hit = needle in str(e)
        print(f"  {name:22s} {'fires' if hit else 'WRONG MESSAGE'}: {str(e).splitlines()[0]}")
        fails += not hit

  # (7) pivot calibration, on its own trajectory: the pipeline one above never turns while
  # standing, so it has nothing to fit. Turning in place at 0.3 rad/s with roll/pitch sway,
  # while the pelvis creeps 3 cm/s FORWARD in the body frame (a slow circle) and wobbles 5 mm,
  # sampled at 100 Hz like the real takes. The creep is what makes the per-window [1, t]
  # detrend load-bearing: with a mean-only detrend r_y comes back ~10 cm off, since the creep
  # direction turns with the same yaw that carries the r_x/r_y information.
  print("\n-- (7) pivot calibration (lever arm from mocap alone)")
  tp = np.arange(0.0, 60.0, 0.01)
  yaw = 0.3 * tp
  roll = 0.04 * np.sin(2 * np.pi * 1.1 * tp)
  pitch = 0.03 * np.sin(2 * np.pi * 0.7 * tp + 1.0)
  R_p = np.stack([so3_exp([0, 0, a]) @ so3_exp([0, b, 0]) @ so3_exp([c, 0, 0])
                  for a, b, c in zip(yaw, pitch, roll)])
  r_plant = np.array([0.05, -0.02, 0.32])
  creep = 0.03 * np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)
  pel = np.cumsum(creep * 0.01, axis=0) + 0.005 * rng.standard_normal((len(tp), 3))
  p_m = pel + np.einsum("nij,j->ni", R_p, r_plant)
  pv = pivot_lever(tp, p_m, R_p, np.ones(len(tp), bool))
  e_piv = np.abs(pv["r"] - r_plant)
  print(f"  fitted {np.array2string(pv['r'], precision=4)} +- "
        f"{np.array2string(pv['se'], precision=4)} vs planted {r_plant.tolist()}  "
        f"({pv['windows']} windows)")
  fails += (e_piv > 0.03).any()                     # synthetic recovery, 3 cm
  fails += (e_piv > 4 * pv["se"] + 1e-3).any()      # the jackknife SE covers the actual error
  zero_ok = (np.abs(pv["r"][2]) < 3 * pv["se"][2])  # r = 0 must be ruled out on the big axis
  print(f"  r_z = 0 ruled out by the SE: {not zero_ok}")
  fails += zero_ok
  none = pivot_lever(tp, p_m, R_p, tp < 3.0)         # one window's worth only
  print(f"  under 3 windows -> {none}")
  fails += none is not None

  print("\nSELFTEST:", "PASS" if fails == 0 else f"FAIL ({fails})")
  return int(fails > 0)


# --------------------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bundle", nargs="?", type=Path, help="logs/robot_logs/<run bundle>/")
  ap.add_argument("mocap", nargs="?", type=Path, help="the Vicon ASCII export for that run")
  ap.add_argument("--lever", nargs=3, type=float, metavar=("X", "Y", "Z"),
                  help="r, the vector FROM the pelvis origin TO the mocap marker-cluster "
                       "origin, in the body frame, metres. Sign convention: "
                       "v_pelvis = v_marker - omega x r. REQUIRED and deliberately without "
                       "a default -- at 1 rad/s and 0.2 m this term is 0.2 m/s, the size of "
                       "the signal being measured, and it leaks yaw rate straight into "
                       "apparent lateral velocity.")
  ap.add_argument("--selftest", action="store_true", help="synthetic end-to-end proof; exits")
  ap.add_argument("--xcorr-floor", type=float, default=PAIR_CORR_FLOOR)
  ap.add_argument("--proc-angle-max", type=float, default=PROC_ANGLE_MAX,
                  help="max median angle (deg) between R*omega_torso and omega_mocap")
  ap.add_argument("--min-excitation", type=float, default=MIN_EXCITATION,
                  help="min singular-value ratio of the Procrustes matrix; below it the "
                       "run did not rotate about enough axes to observe R")
  ap.add_argument("--coarse-seed", type=float, metavar="SECONDS",
                  help="replace the whole-take masked search with this t_wall-minus-t_mocap "
                       "offset (the per-window search then looks +-1 s around it). Rarely "
                       "needed: the masked search found every 2026-09-16 take on its own. "
                       "Every window and frame gate still runs, so a wrong seed is caught.")
  ap.add_argument("--smooth", type=float, default=0.0, metavar="S",
                  help="boxcar (s) on the differentiated mocap velocity; 0 = off")
  ap.add_argument("--rot-cols", help="override: 'RX,RY,RZ' or column indices '2,3,4'")
  ap.add_argument("--pos-cols", help="override: 'TX,TY,TZ' or column indices '5,6,7'")
  ap.add_argument("--rot-format", choices=("quat", "euler_xyz", "helical"))
  ap.add_argument("--pos-units", choices=("mm", "m"))
  ap.add_argument("--rate", type=float, help="mocap sample rate (Hz) if the file has no rate line")
  args = ap.parse_args()

  if args.selftest:
    return selftest()
  if not args.bundle or not args.mocap:
    ap.error("give a bundle directory and a mocap csv (or --selftest)")
  if args.lever is None:
    ap.error("--lever X Y Z is required: the marker cluster is not the pelvis, and an "
             "omega x r term of the same magnitude as the measurement cannot be guessed")
  align(args.bundle, args.mocap, np.asarray(args.lever, float),
        xcorr_floor=args.xcorr_floor, proc_angle_max=args.proc_angle_max,
        min_excitation=args.min_excitation, smooth=args.smooth,
        coarse_seed=args.coarse_seed,
        rot_cols=args.rot_cols, pos_cols=args.pos_cols, rot_format=args.rot_format,
        pos_units=args.pos_units, rate=args.rate)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
