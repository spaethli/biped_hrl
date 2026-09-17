#!/usr/bin/env python3
"""Torso-frame marker template for `mocap_align.py --marker-template`, from tape measurements.

The geometric frame is the IMU-free way to express mocap in a robot-fixed frame: the marker
layout, measured on the torso, defines the frame, so no gyro enters it. Measurement protocol
and the maths: doc/hrl/mocap_alignment.md, "Geometric frame".

Input (JSON, metres): per marker its FACE (front/back), `u` = horizontal distance of the marker
CENTRE from that face's reference edge, `v` = height above the face's bottom edge, `offset` =
distance of the marker centre in front of the face surface; per face its `width`; and the
`convention` the operator used for "left edge" (`viewer`: the edge on the left of a person
looking at that face; `robot`: the robot's own left). Torso frame: x forward, y left, z up.

Three quantities a tape cannot give are fitted on the Vicon inter-marker distances (which are
exact to about a millimetre): the face separation `depth`, and the back face's height and
lateral offset against the front (`back_dz`, `back_dy`), since both faces are measured from
their own edges. The marker labelling is found at the same time (Vicon's labels change
between takes), and the other convention is scored too: distances cannot tell a mirror image
apart, the rigid fit can.

Output (JSON): marker coordinates in the torso frame plus the diagnostics that say how far to
trust it: distance and fit rms, where each marker really is relative to its tape position,
any extra check distances, and a Monte Carlo frame uncertainty with the per-number tape error
implied by that gap.

Usage:
  python scripts/mocap_marker_template.py doc/hrl/mocap/h1_2_torso_markers_tape.json \\
      logs/robot_logs/mocap_session_20260916/T1.c3d [more.c3d ...] \\
      -o doc/hrl/mocap/h1_2_torso_marker_template.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mocap_align import kabsch, match_markers, read_c3d_markers, so3_log  # noqa: E402

FACES = ("front", "back")
MC_DRAWS = 300


def layout(meas, depth, back_dz, back_dy, convention):
  """Torso-frame positions (x fwd, y left, z up) of the measured markers.

  Front face at x = +depth/2, back face at x = -depth/2; y = 0 is the face centre; z = 0 the
  front bottom edge. `u` is from the face's reference edge:
    front, viewer: the viewer's left edge is the robot's RIGHT (-y)  -> y = u - W/2
    front, robot : the robot's left edge (+y)                        -> y = W/2 - u
    back, either : behind the robot both lefts coincide (+y)         -> y = W/2 - u
  """
  out = []
  for m in meas["markers"]:
    W = meas["faces"][m["face"]]["width"]
    if m["face"] == "front":
      x = depth / 2 + m["offset"]
      y = (m["u"] - W / 2) if convention == "viewer" else (W / 2 - m["u"])
      z = m["v"]
    elif m["face"] == "back":
      x = -depth / 2 - m["offset"]
      y = W / 2 - m["u"] + back_dy
      z = m["v"] + back_dz
    else:
      raise SystemExit(f"marker {m['name']}: face {m['face']!r} not supported ({FACES})")
    out.append([x, y, z])
  return np.array(out)


def dmat(X):
  return np.linalg.norm(X[:, None] - X[None], axis=-1)


BOUNDS = [(0.05, 0.5), (-0.15, 0.15), (-0.1, 0.1)]      # depth, back_dz, back_dy (m)
STARTS = [(d, z, y) for d in (0.1, 0.2, 0.3) for z in (-0.05, 0.0, 0.05) for y in (-0.05, 0.0, 0.05)]


def fit_free(meas, D_obs, order, convention, starts=((0.2, 0.0, 0.0),)):
  """(depth, back_dz, back_dy), distance rms: the free geometry that best reproduces the Vicon
  inter-marker distances. The cost has local minima (a single start found depth 0.087 m at
  34 mm rms where 0.165 m fits at 15 mm), so the final fits use a grid of starts."""
  iu = np.triu_indices(len(order), 1)
  Dm = D_obs[np.ix_(order, order)][iu]

  def cost(q):
    return np.sum((dmat(layout(meas, *q, convention))[iu] - Dm) ** 2)
  r = min((minimize(cost, x0, bounds=BOUNDS) for x0 in starts), key=lambda r: r.fun)
  return r.x, float(np.sqrt(r.fun / len(iu[0])))


def fit_rms(T, X):
  res = []
  for x in X:
    R, t = kabsch(T, x)
    res.append(np.sqrt(np.mean(np.sum((T @ R.T + t - x) ** 2, 1))))
  return float(np.median(res))


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("measurements", type=Path)
  ap.add_argument("c3d", type=Path, nargs="+", help="one or more takes of the same marker set")
  ap.add_argument("-o", "--out", type=Path, required=True)
  args = ap.parse_args()
  meas = json.loads(args.measurements.read_text())
  names = [m["name"] for m in meas["markers"]]
  n = len(names)

  # observed geometry, pooled over takes (each take may label the trajectories differently,
  # so every take is matched on its own and re-ordered before pooling)
  takes = []
  for p in args.c3d:
    xyz, vis, _, labels = read_c3d_markers(p)
    ok = vis.all(1)
    if ok.sum() < 50:
      raise SystemExit(f"{p.name}: only {int(ok.sum())} frames see every labelled marker")
    X = xyz[ok][:: max(ok.sum() // 400, 1)]
    takes.append((p.name, X, np.median(dmat_frames(X), 0), xyz, vis))

  D0 = takes[0][2]
  # rank the declared convention against the mirror by the rigid fit, not by distances
  report = {}
  for conv in ("viewer", "robot"):
    cands = []
    for order in itertools.permutations(range(takes[0][1].shape[1]), n):
      q, d_rms = fit_free(meas, D0, list(order), conv)
      cands.append((d_rms, list(order), q))
    cands.sort(key=lambda c: c[0])
    ranked = []
    for _, order, _ in cands[:6]:
      q, d_rms = fit_free(meas, D0, order, conv, STARTS)
      T = layout(meas, *q, conv)
      ranked.append((fit_rms(T, takes[0][1][:, order][:: max(len(takes[0][1]) // 60, 1)]), d_rms, order, q))
    ranked.sort(key=lambda c: c[0])
    report[conv] = ranked[0]
  declared = meas["convention"]
  other = "robot" if declared == "viewer" else "viewer"
  f_rms, d_rms, order, q = report[declared]
  print(f"convention {declared}: fit rms {f_rms * 1e3:.1f} mm, distance rms {d_rms * 1e3:.1f} mm")
  print(f"convention {other}: fit rms {report[other][0] * 1e3:.1f} mm, distance rms "
        f"{report[other][1] * 1e3:.1f} mm")
  if report[other][0] < f_rms:
    print(f"⚠ the {other} convention fits the markers better than the declared {declared}: "
          f"check how left and right were measured")

  # refit on all takes pooled (each take re-labelled by its own best order)
  Ds = []
  T_first = layout(meas, *q, declared)
  for i, (name, X, D, xyz, vis) in enumerate(takes):
    o = order if i == 0 else match_markers(xyz, vis, T_first)[0]
    Ds.append(D[np.ix_(o, o)])
  D_pool = np.median(Ds, 0)
  q, d_rms = fit_free(meas, D_pool, list(range(n)), declared, [tuple(q)] + STARTS)
  T = layout(meas, *q, declared)

  # where the markers really are, in the frame the tape defines
  X0 = takes[0][1][:, order]
  act = np.median(np.stack([(x - kabsch(T, x)[1]) @ kabsch(T, x)[0] for x in X0]), 0)
  gap = act - T
  gap_rms = float(np.sqrt(np.mean(np.sum(gap ** 2, 1))))
  sigma = gap_rms / np.sqrt(3.0)                     # per tape number
  print(f"fitted depth {q[0]:.3f} m, back_dz {q[1]:+.3f} m, back_dy {q[2]:+.3f} m; "
        f"distance rms {d_rms * 1e3:.1f} mm over {len(takes)} take(s)")
  for nm, t_, a_, g_ in zip(names, T, act, gap):
    print(f"  {nm:18s} tape {np.round(t_ * 1e3)}  actual {np.round(a_ * 1e3)}  gap {np.round(g_ * 1e3)} mm")
  print(f"  rms gap {gap_rms * 1e3:.1f} mm -> about {sigma * 1e3:.1f} mm per tape number")

  checks = []
  for c in meas.get("check_distances", []):
    ia = [names.index(k) for k in c["a"]]
    ib = [names.index(k) for k in c["b"]]
    ax = "xyz".index(c["axis"])
    got = abs(act[ia, ax].mean() - act[ib, ax].mean())
    tape = abs(T[ia, ax].mean() - T[ib, ax].mean())
    checks.append({**c, "actual": round(float(got), 4), "template": round(float(tape), 4)})
    print(f"  check {c['what']}: measured {c['value']:.3f}, markers {got:.3f}, template {tape:.3f} m")

  # frame uncertainty: every tape number perturbed by sigma, refit, frame rotation vs nominal
  rng = np.random.default_rng(0)
  keys = [(i, k) for i in range(n) for k in ("u", "v", "offset")]
  sample = X0[:: max(len(X0) // 25, 1)]
  R0 = [kabsch(T, x)[0] for x in sample]
  angles = []
  for _ in range(MC_DRAWS):
    m2 = json.loads(json.dumps(meas))
    for i, k in keys:
      m2["markers"][i][k] += rng.normal(0, sigma)
    for f in m2["faces"].values():
      f["width"] += rng.normal(0, sigma)
    q2, _ = fit_free(m2, D_pool, list(range(n)), declared, [tuple(q)])
    T2 = layout(m2, *q2, declared)
    angles.append(np.median([np.degrees(np.linalg.norm(so3_log(a.T @ kabsch(T2, x)[0])))
                             for a, x in zip(R0, sample)]))
  angles = np.array(angles)
  unc = {"sigma_mm": round(sigma * 1e3, 1), "median": round(float(np.median(angles)), 2),
         "p90": round(float(np.percentile(angles, 90)), 2)}
  print(f"frame uncertainty from the tape: median {unc['median']} deg, p90 {unc['p90']} deg "
        f"(sigma {unc['sigma_mm']} mm per number)")

  out = {
    "_comment": "Torso-frame marker template (x forward, y left, z up; metres) for "
                "mocap_align.py --marker-template. Built by scripts/mocap_marker_template.py.",
    "measurements": args.measurements.name, "c3d": [p.name for p in args.c3d],
    "convention": declared,
    "markers": {nm: [round(float(v), 5) for v in t_] for nm, t_ in zip(names, T)},
    "fitted": {"depth": round(float(q[0]), 4), "back_dz": round(float(q[1]), 4),
               "back_dy": round(float(q[2]), 4)},
    "distance_rms_mm": round(d_rms * 1e3, 1), "fit_rms_mm": round(f_rms * 1e3, 1),
    "mirror_fit_rms_mm": round(report[other][0] * 1e3, 1),
    "tape_gap_mm": {nm: [round(float(v) * 1e3, 1) for v in g_] for nm, g_ in zip(names, gap)},
    "checks": checks,
    "frame_uncertainty_deg": unc,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(out, indent=2) + "\n")
  print(f"wrote {args.out}")


def dmat_frames(X):
  return np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)


if __name__ == "__main__":
  main()
