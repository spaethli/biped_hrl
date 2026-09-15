#!/usr/bin/env python3
"""Ground-truth base velocity for one hardware RUN, from a Vicon export.

A Run is one FSM `entry` (CONTEXT.md), already sliced into a bundle by
`bundle_hardware_run.py`. This tool adds the one thing no onboard log carries: the
robot's ACTUAL base velocity, against which `leg_odom`, the complementary filter and
every EKF arm can finally be scored rather than compared to each other.

TWO UNKNOWNS, FITTED SEPARATELY. The Vicon system and the robot's controller run on
different clocks (offset AND rate -- measured -2462 to -6613 ppm between the robot's own
two loggers), and the Vicon rigid-body frame is some arbitrary constant rotation R away
from the robot's torso frame. Fitting both together is ill-conditioned: a clock error and
a frame error both show up as "the traces do not match", and the optimiser trades one
against the other. They are decoupled here by choosing signals that see only one unknown
at a time:

  1. TIME, from |omega|.  The MAGNITUDE of the angular velocity is invariant to R, so the
     mocap's |omega| and the flight recorder's |est_gyro| are the same scalar function of
     time whatever the frame relationship is. `fit_alignment` (bench_flight_recorder, the
     same affine fitter that pairs the joint-telemetry log) recovers offset and rate from
     it. No frame knowledge is used.
  2. FRAME, from omega.  With the clocks aligned, omega_mocap(t) = R omega_torso(t) holds
     sample by sample, which is orthogonal Procrustes: one SVD, in closed form, with the
     reflection forced out (det = +1). No calibration pose, no T-pose capture, no manual
     alignment of the rigid body in Nexus.

Only then is the LINEAR velocity computed, by differentiating the mocap position and
rotating it through the now-known frames.

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

NO MOCAP DATA EXISTED WHEN THIS WAS WRITTEN. The Vicon reader is therefore written to a
documented, sniffed format assumption (printed prominently before any work is done, with a
CLI override for every field), and the whole pipeline is proven end-to-end by `--selftest`
against a synthetic trajectory with a planted clock skew, a planted frame rotation and a
planted lever arm. What the self-test cannot prove is that a real Nexus export matches the
sniffer -- check the printed mapping on the first real file.

Usage:
  python scripts/mocap_align.py --selftest
  python scripts/mocap_align.py logs/robot_logs/2026_09_14-_14_07_A1a vicon_run1.csv \\
      --lever 0.0 0.0 0.30
  # ... and when the sniffer guesses wrong:
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

from bench_flight_recorder import fit_alignment  # noqa: E402
from bundle_hardware_run import MAX_SKEW, PAIR_CORR_FLOOR  # noqa: E402
from replay_base_estimators import quat_to_mat, so3_exp, so3_log  # noqa: E402

OUT_NAME = "mocap_aligned.csv"
PROC_ANGLE_MAX = 15.0   # deg; median angle between R*omega_torso and omega_mocap
MIN_EXCITATION = 0.02   # smallest singular value / largest, of the Procrustes matrix
SMOOTH_S = 0.02         # boxcar on |omega| before alignment; fit_alignment resamples to 50 Hz
FRAME_SMOOTH_S = 0.10   # boxcar on the omega VECTORS before the frame fit (see below)
PSI_WARN = 0.05         # rad; above this the waist is not "held" and the mount matters


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
  dropped = int((~ok).sum())
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
          f"{dropped} non-finite rows dropped)")
    print(f"  layout        PRE-PROCESSED flat CSV (single header row), {info['rate']:.1f} Hz "
          f"from time_s")
    print("  rotation      quaternion qw/qx/qy/qz, taken as world<-body")
    print("  position      x/y/z_mm -> m")
  return dict(t=t, R=R, p=p, w_world=w_world, info=info)


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
  use = use[ok]
  frame, sub = use[:, 0], use[:, 1]
  rvals, pvals = use[:, 2:2 + len(ri)], use[:, 2 + len(ri):]

  t = (frame - frame[0]) / rate
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
  return dict(t=t, R=R, p=p, info=info)


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
  """(t, gyro [N,3], psi [N]) from a bundled flight-recorder slice.

  Every row is kept, including the ~50% that are exact sensor repeats: the spec is to emit
  ground truth on the recorder's OWN row grid, so consumers can join on `t` without an
  interpolation of their own.
  """
  d = np.genfromtxt(path, delimiter=",", names=True)
  have = d.dtype.names
  for c in ("t", "est_gyro_x", "est_gyro_y", "est_gyro_z"):
    if c not in have:
      raise SystemExit(f"{Path(path).name}: column '{c}' missing. The gyro is the only "
                       f"signal both logs share, so a recorder slice without the `est_` "
                       f"block cannot be aligned to mocap at all.")
  t = np.atleast_1d(d["t"]).astype(float)
  gyro = np.stack([d[f"est_gyro_{a}"] for a in "xyz"], 1).astype(float)
  psi = np.atleast_1d(d["meas_q12"]).astype(float) if "meas_q12" in have else np.zeros(len(t))
  return t, gyro, psi


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


def _map_corr(t_f, sig_f, t_m, sig_m, b, m, grid=0.002):
  """Normalized correlation of the two |omega| traces under the map t_f = t_m + b + m t_m."""
  tfm = t_m + b + m * t_m
  lo, hi = max(t_f[0], tfm[0]), min(t_f[-1], tfm[-1])
  if hi - lo < 1.0:
    return -1.0
  g = np.arange(lo, hi, grid)
  A, F = np.interp(g, tfm, sig_m), np.interp(g, t_f, sig_f)
  A = (A - A.mean()) / (A.std() + 1e-12)
  F = (F - F.mean()) / (F.std() + 1e-12)
  return float((A * F).mean())


def _golden(f, lo, hi, n=60):
  g = (np.sqrt(5.0) - 1.0) / 2.0
  c, d = hi - g * (hi - lo), lo + g * (hi - lo)
  fc, fd = f(c), f(d)
  for _ in range(n):
    if fc > fd:
      hi, d, fd = d, c, fc
      c = hi - g * (hi - lo)
      fc = f(c)
    else:
      lo, c, fc = c, d, fd
      d = lo + g * (hi - lo)
      fd = f(d)
    if hi - lo < 1e-7:
      break
  return 0.5 * (lo + hi)


def refine_affine(t_f, sig_f, t_m, sig_m, al, coarse=None, bound=0.25):
  """Sub-sample polish of `fit_alignment`'s map, on the SAME |omega| traces.

  Two reasons this stage exists, both measured:

  * PRECISION. `fit_alignment` searches lags on a 20 ms grid (`CTRL_DT`) and fits a line
    through a handful of 30 s windows, so its rate is quantisation-limited -- a planted
    -4500 ppm comes back as -4258 with a 17 ms offset error in `--selftest`. That matters
    here in a way it does not for the joint-telemetry pairing it was written for, because
    the lever-arm term differentiates: at 5 rad/s^2 and r = 0.2 m, 30 ms of residual timing
    error is 30 mm/s of fabricated body velocity -- error of the same size as the estimator
    error the ground truth exists to measure.
  * SEARCH. On a short or quiet run its per-window line can simply mis-fit. Replaying
    2026-09-14_15-16-46 (73 s) with a planted -3300 ppm it reported -8880 ppm at xcorr 0.964
    from 3 of 5 windows: a confident wrong answer, scoring 0.679 where the true map scores
    0.993. Seeding the polish from `coarse_lag` alone (offset only, no rate) as well as from
    the windowed fit recovers it to 4.9 ms / 54 ppm.

  It stays a POLISH, not a second aligner: both seeds come from `bench_flight_recorder`, each
  anchor is clamped to +-`bound` around its own seed, and an answer is kept only if it beats
  the seed's correlation. The bound is under half a stride period (~0.5 s on this robot), so
  no start can slide onto the neighbouring alias that `fit_alignment`'s unwrap exists to
  prevent.

  Parameterised by the lag at each END of the overlap rather than by (offset, rate): those
  two are nearly independent, while (b, m) lie in a long diagonal valley that coordinate
  descent crawls along (12 ms after 8 rounds, vs 0.2 ms after 5 with the anchors).
  """
  t0, t1 = float(t_m[0]), float(t_m[-1])
  if t1 - t0 < 1.0:
    return al["b"], al["m"], float("nan"), False

  def bm(L0, L1):
    """the two end lags -> (offset, rate)"""
    m = (L1 - L0) / (t1 - t0)
    return L0 - m * t0, m

  seeds = [(al["b"], al["m"])]
  if coarse is not None and np.isfinite(coarse):
    seeds.append((float(coarse), 0.0))
  best = (al["b"], al["m"], _map_corr(t_f, sig_f, t_m, sig_m, al["b"], al["m"]), False)
  for sb, sm in seeds:
    s0, s1 = sb + sm * t0, sb + sm * t1
    L0, L1 = s0, s1
    for _ in range(6):
      L0 = _golden(lambda x: _map_corr(t_f, sig_f, t_m, sig_m, *bm(x, L1)),
                   s0 - bound, s0 + bound)
      L1 = _golden(lambda x: _map_corr(t_f, sig_f, t_m, sig_m, *bm(L0, x)),
                   s1 - bound, s1 + bound)
    b, m = bm(L0, L1)
    c = _map_corr(t_f, sig_f, t_m, sig_m, b, m)
    if c > best[2]:
      best = (b, m, c, True)
  return best


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


def align(bundle, mocap_path, lever, xcorr_floor=PAIR_CORR_FLOOR,
          proc_angle_max=PROC_ANGLE_MAX, min_excitation=MIN_EXCITATION, smooth=0.0,
          refine=True, write=True, quiet=False, **reader_kw):
  """Full pipeline for one bundle. Returns the result dict; writes the csv and run.json."""
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
  t_f, gyro, psi = read_flight(flight)
  if not quiet:
    print(f"\n  flight slice  {flight.name}  ({len(t_f)} rows, {t_f[-1] - t_f[0]:.1f} s, "
          f"entry {run.get('entry')})")

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
  sig_f = _boxcar(np.linalg.norm(gyro, axis=1), t_f)
  al = fit_alignment(t_f, sig_f, t_m, sig_m)
  if al is None:
    raise SystemExit(
      "TIME ALIGNMENT FAILED: the |omega| traces have no usable overlap. `fit_alignment` "
      "needs ~20 s of simultaneous recording to fit an offset and a rate; check that the "
      "mocap take really covers this entry's wall window.")
  if not quiet:
    print(f"\n  [1] clock   t_flight = t_mocap + ({al['b']:+.3f} + {al['m'] * 1e6:+.1f} ppm "
          f"* t_mocap)   xcorr {al['xcorr']:.3f}  resid {al['resid_ms']:.1f} ms  "
          f"windows {al['windows']}/{al['windows_total']}")
  if not np.isfinite(al["xcorr"]) or al["xcorr"] < xcorr_floor:
    raise SystemExit(
      f"TIME ALIGNMENT REJECTED: |omega| cross-correlation {al['xcorr']:.3f} < "
      f"{xcorr_floor:.2f}.\n  The mocap take and this run do not describe the same motion. "
      f"Either the take covers a different entry (a flight-recorder file holds several "
      f"runs), the rigid body was occluded for most of it, or --rot-format/--rate is "
      f"wrong, which corrupts omega before anything else sees it.\n  Nothing written.")
  b, mrate, rcorr, moved = (
    refine_affine(t_f, sig_f, t_m, sig_m, al, coarse=al.get("coarse_lag_s"))
    if refine else (al["b"], al["m"], float("nan"), False))
  if not quiet:
    how = (f"polished to {b:+.4f} s, {mrate * 1e6:+.1f} ppm" if moved
           else "kept the windowed fit (polish found nothing better)")
    print(f"      polish  {how}   |omega| correlation {rcorr:.5f}")
  # The rate is gated AFTER the polish, on the map actually used: the windowed fit's own
  # rate can be badly off on a short run and still be recoverable (measured -8880 ppm
  # against a true -3300, polished back to -3246).
  if abs(mrate) > MAX_SKEW:
    raise SystemExit(
      f"TIME ALIGNMENT REJECTED: fitted rate {mrate * 1e6:+.0f} ppm exceeds "
      f"{MAX_SKEW * 1e6:.0f} ppm. Real clock skew on this rig is thousands of ppm; a "
      f"percent-level 'rate' means the fit absorbed a bad offset into the slope.\n"
      f"  Nothing written.")

  # ---- step 2: frame, in closed form, on the now-aligned angular velocities ---------
  tf_of_m = t_m + (b + mrate * t_m)                  # mocap samples, on the flight clock
  cover = (t_f >= tf_of_m[0]) & (t_f <= tf_of_m[-1])
  if cover.sum() < 100:
    raise SystemExit(f"OVERLAP TOO SHORT: only {int(cover.sum())} flight rows fall inside "
                     f"the mocap take after alignment. Nothing written.")
  W = np.stack([np.interp(t_f[cover], tf_of_m, w_m[:, i]) for i in range(3)], 1)
  G = gyro[cover]
  # Smooth BOTH omega vectors identically before the frame fit. The rotation is a property of
  # the mount, not of any one sample, and the per-sample angle between two noisy 3-vectors is
  # dominated by noise wherever |omega| is small. Applied to both channels so no relative lag
  # is introduced. Measured: TRIAL05 10.8 -> 7.5 deg, TRIAL03 12.6 -> 7.5 deg.
  tc = t_f[cover]
  Ws = np.stack([_boxcar(W[:, i], tc, FRAME_SMOOTH_S) for i in range(3)], 1)
  Gs = np.stack([_boxcar(G[:, i], tc, FRAME_SMOOTH_S) for i in range(3)], 1)
  R, S, det = procrustes_rotation(Ws, Gs)
  W, G = Ws, Gs
  pred = G @ R.T
  big = np.linalg.norm(W, axis=1) > max(np.median(np.linalg.norm(W, axis=1)), 1e-6)
  ang = np.degrees(np.arccos(np.clip(
    (pred[big] * W[big]).sum(1) / (np.linalg.norm(pred[big], axis=1)
                                   * np.linalg.norm(W[big], axis=1) + 1e-12), -1, 1)))
  med_ang = float(np.median(ang)) if big.any() else float("nan")
  rel = float(np.linalg.norm(W - pred) / max(np.linalg.norm(W), 1e-12))
  exc = float(S[2] / max(S[0], 1e-300))
  rpy = np.degrees(so3_log(R))
  if not quiet:
    print(f"  [2] frame   R (torso -> mocap body), rotvec {np.array2string(rpy, precision=2)}"
          f" deg   det {det:+.0f}")
    print(f"              residual {med_ang:.2f} deg median, {rel * 100:.2f}% rel;  "
          f"excitation sv {S[0]:.3g}/{S[1]:.3g}/{S[2]:.3g} (ratio {exc:.3f})")
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
  # staircase and report interpolation noise as acceleration.
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

  def onto_flight(x):
    y = np.full((len(t_f),) + x.shape[1:], np.nan)
    if x.ndim == 1:
      y[cover] = np.interp(t_f[cover], tf_of_m, x)
    else:
      for i in range(x.shape[1]):
        y[cover, i] = np.interp(t_f[cover], tf_of_m, x[:, i])
    return y

  v = onto_flight(v_t)
  # v_P^P = Rz(psi) (v_I^T - omega_T x r), h1_2_kinematics.md; psi is already on this grid.
  c, s = np.cos(psi), np.sin(psi)
  v = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
  out = dict(t=t_f, v=v, wz=onto_flight(wz), p=onto_flight(p_pelvis), yaw=onto_flight(yaw),
             cover=cover, align=al, map=(b, mrate, rcorr, moved), R=R,
             proc=dict(median_angle_deg=med_ang, rel_resid=rel, excitation=exc, det=det,
                       sv=S.tolist()))
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
          f"fall inside the mocap take")

  if write:
    cols = np.column_stack([t_f, v, out["wz"], out["p"], out["yaw"]])
    np.savetxt(bundle / OUT_NAME, cols, delimiter=",", fmt="%.6f",
               header="t,gt_vx,gt_vy,gt_vz,gt_wz,gt_px,gt_py,gt_pz,gt_yaw", comments="")
    run["mocap"] = Path(mocap_path).name
    run["mocap_align"] = {
      "xcorr": round(float(al["xcorr"]), 4),          # the gated quantity: fit_alignment's
      "resid_ms": round(float(al["resid_ms"]), 2),
      "skew_ppm": round(float(mrate) * 1e6, 1),       # after the sub-sample polish
      "lag_s": round(float(b), 4),
      "polish_corr": round(float(rcorr), 5),
      "polished": bool(moved),
      "lever": [float(x) for x in lever],
      "R": [[float(x) for x in row] for row in R],
      "proc_angle_deg": round(med_ang, 3),
      "excitation": round(exc, 4),
      "n_samples": n_ok,
    }
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    if not quiet:
      print(f"\n  wrote {bundle / OUT_NAME}  and updated run.json")
  return out


# --------------------------------------------------------------------------- self-test

def _write_vicon(path, t, R, p_mm, rate, fmt, obj="H1_2:torso"):
  """Synthesise a Nexus-shaped ASCII export, so the self-test goes through `read_vicon`."""
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
    for k in range(len(t)):
      fh.write(",".join([str(k + 1), "0"]
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
  # against the flight recorder's t = 0. Samples whose flight time is negative are kept: the
  # truth interpolation clamps there, i.e. the robot stands still before the run, which is
  # exactly what the mocap sees and is what the coverage mask then has to exclude.
  b_plant, m_plant, rate = -12.30, -4500e-6, 200.0
  t_mocap = np.arange(0.0, 83.0, 1.0 / rate)
  tf_m = t_mocap + (b_plant + m_plant * t_mocap)       # flight time of each mocap sample
  keep = tf_m <= tt[-1]
  t_mocap, tf_m = t_mocap[keep], tf_m[keep]
  # element-wise interpolation of the attitude, then re-orthonormalised (the 1 kHz truth
  # grid makes the chord error ~1e-7 rad, far below every tolerance asserted below)
  Rm = _orth(np.transpose(np.array([[np.interp(tf_m, tt, R_wm[:, i, j]) for j in range(3)]
                                    for i in range(3)]), (2, 0, 1)))
  pm = np.stack([np.interp(tf_m, tt, p_mark[:, i]) for i in range(3)], 1)

  t_f = np.cumsum(rng.uniform(0.0018, 0.0022, 50000))  # ~500 Hz, non-uniform, like the real log
  t_f = t_f - t_f[0]
  t_f = t_f[t_f <= tt[-1]]
  gyro = np.stack([np.interp(t_f, tt, w_true[:, i]) for i in range(3)], 1)

  print("planted: b = %.3f s, skew = %+.0f ppm, lever = %s m, R = rotvec %s deg"
        % (b_plant, m_plant * 1e6, lever.tolist(),
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
      fh.write("t,t_wall,est_gyro_x,est_gyro_y,est_gyro_z,meas_q12\n")
      for k in range(len(t_f)):
        fh.write(f"{t_f[k]:.4f},{t_f[k]:.4f},{gyro[k, 0]:.6f},{gyro[k, 1]:.6f},"
                 f"{gyro[k, 2]:.6f},0.0\n")
    (bundle / "run.json").write_text(json.dumps(
      {"policy_tag": "SELFTEST", "flight_recorder": fcsv.name, "entry": 0,
       "mocap": None}, indent=2))
    vfile = td / "vicon_selftest.csv"
    _write_vicon(vfile, t_mocap, Rm, pm * 1e3, rate, "helical")

    print("\n-- (1..4) full pipeline")
    out = align(bundle, vfile, lever, quiet=False)
    al, R = out["align"], out["R"]
    b_got, m_got = out["map"][0], out["map"][1]

    print("\n-- recovery")
    e_b = abs(b_got - b_plant)
    e_m = abs(m_got - m_plant) * 1e6
    e_R = float(np.degrees(np.linalg.norm(so3_log(R.T @ R_plant))))
    print(f"  clock offset   {b_got:+.4f} s  vs planted {b_plant:+.4f}  -> "
          f"err {e_b * 1e3:.1f} ms   (windowed fit alone: {al['b']:+.4f}, "
          f"err {abs(al['b'] - b_plant) * 1e3:.1f} ms)")
    print(f"  clock skew     {m_got * 1e6:+.1f} ppm vs planted {m_plant * 1e6:+.1f} -> "
          f"err {e_m:.1f} ppm   (windowed fit alone: {al['m'] * 1e6:+.1f}, "
          f"err {abs(al['m'] - m_plant) * 1e6:.0f} ppm)")
    print(f"  frame R        angle to planted {e_R:.4f} deg "
          f"(residual {out['proc']['median_angle_deg']:.3f} deg median)")
    fails += e_b > 0.030
    fails += e_m > 60.0
    fails += e_R > 1.0

    # velocity vs truth, read back from the csv this run actually wrote
    got = np.genfromtxt(bundle / OUT_NAME, delimiter=",", names=True)
    ok = np.isfinite(got["gt_vx"])
    ref = np.stack([np.interp(got["t"][ok], tt, v_true[:, i]) for i in range(3)], 1)
    err = np.stack([got[f"gt_v{a}"][ok] for a in "xyz"], 1) - ref
    rms = float(np.sqrt((err ** 2).sum(1).mean()))
    print(f"  body velocity  rms |err| {rms * 1e3:.2f} mm/s, max {np.abs(err).max() * 1e3:.2f}"
          f" mm/s over {int(ok.sum())} rows  (|v| ~ {np.linalg.norm(ref, axis=1).mean():.2f} m/s)")
    fails += rms > 0.010
    # position and yaw come from the same map; a sign error in either shows up here
    ref_p = np.stack([np.interp(got["t"][ok], tt, p_pel[:, i]) for i in range(3)], 1)
    e_p = float(np.abs(np.stack([got[f"gt_p{a}"][ok] for a in "xyz"], 1) - ref_p).max())
    ref_wz = np.interp(got["t"][ok], tt, w_true[:, 2])   # yaw rate, level-ish flight
    print(f"  pelvis position max err {e_p * 1e3:.2f} mm;  gt_wz vs body yaw rate "
          f"corr {np.corrcoef(got['gt_wz'][ok], ref_wz)[0, 1]:.4f}")
    fails += e_p > 0.005

    # (5) the lever arm: re-run with r = 0 and check the contamination is exactly omega x r
    print("\n-- (5) lever-arm correction")
    out0 = align(bundle, vfile, np.zeros(3), quiet=True, write=False)
    err0 = out0["v"][ok] - ref
    rms0 = float(np.sqrt((err0 ** 2).sum(1).mean()))
    w_body = np.stack([np.interp(got["t"][ok], tt, w_true[:, i]) for i in range(3)], 1)
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
  ap.add_argument("--no-refine", action="store_true",
                  help="skip the sub-sample polish and use fit_alignment's windowed map "
                       "as-is (its lag grid is 20 ms; see refine_affine)")
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
        min_excitation=args.min_excitation, smooth=args.smooth, refine=not args.no_refine,
        rot_cols=args.rot_cols, pos_cols=args.pos_cols, rot_format=args.rot_format,
        pos_units=args.pos_units, rate=args.rate)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
