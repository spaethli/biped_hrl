"""Measure the per-joint |tau_est| / |dq| headroom a policy actually needs, and emit the
`h1_2_joint_trips` table for deploy/robots/h1_2/include/h1_2_limits.h.

WHY THIS EXISTS. The trip thresholds have to clear normal WALKING with margin and still
catch a runaway. Sizing them off the stand is badly wrong (36 Nm / 0.06 rad/s, which every
stride beats by 5-350x), and sizing them off ONE policy's walking is wrong in a subtler
way: the first table was cut at 1.3x the pooled HARDWARE walking maximum, which is
A0-dominated, and it then held a leg joint during the A1 candidate's own normal bridge
gait. A1 is measurably twitchier than A0, so "measured walking" has to mean the walking of
every policy the table will be applied to.

Two sources, unioned:
  hardware  robot trajectory logs (`trajectories/all_joints_*.csv`), leg-active samples with
            the IMU tilt envelope under 12 deg for +-1 s so stumbles and falls are excluded
  bridge    SafetyLogger flight recordings (`logs/deploy_safety/*.csv`), any session that
            did NOT fall, using meas_dq for velocity and `trip_ratio` for torque

The bridge CSV has no per-joint tau column (27 more at 500 Hz), so torque headroom comes
from `trip_ratio` -- the per-tick max over joints of max(|tau|/tau_trip, |dq|/dq_trip),
logged on every tick precisely so a clean session reports its margin instead of only a
boolean. Where trip_ratio exceeds the dq ratio computable from meas_dq, the excess is
torque on `trip_jid`.

Usage:
  python scripts/measure_joint_trip_thresholds.py [--headroom 1.3]
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NAMES = ["left_hip_yaw", "left_hip_pitch", "left_hip_roll", "left_knee",
         "left_ankle_pitch", "left_ankle_roll",
         "right_hip_yaw", "right_hip_pitch", "right_hip_roll", "right_knee",
         "right_ankle_pitch", "right_ankle_roll", "waist_yaw",
         "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
         "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
         "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
         "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
LEG = list(range(12))
WALK_DQ, TILT_OK, GUARD_S = 1.0, np.radians(12.0), 1.0
# Joint groups share a threshold: left/right are the same hardware and the arm joints are
# too sparsely exercised to justify per-side numbers.
GROUPS = {'hip_yaw': [0, 6], 'hip_pitch': [1, 7], 'hip_roll': [2, 8], 'knee': [3, 9],
          'ank_pitch': [4, 10], 'ank_roll': [5, 11], 'waist': [12],
          'shoulder': [13, 14, 15, 20, 21, 22],
          'elbow_wrist': [16, 17, 18, 19, 23, 24, 25, 26]}


def _rows(path):
  with open(path) as f:
    r = csv.reader(f)
    hdr = [h.strip() for h in next(r)]
    rows = [row for row in r if len(row) == len(hdr)]
  return np.array([[float(x) for x in row] for row in rows]), {h: k for k, h in enumerate(hdr)}


def hardware_walking_max():
  """(tau, dq) per joint over healthy hardware walking."""
  tau_mx, dq_mx = np.zeros(27), np.zeros(27)
  secs = 0.0
  for p in sorted(glob.glob(str(Path.home() / 'ramlab_ws/trajectories/all_joints_*.csv'))):
    try:
      d, i = _rows(p)
    except Exception:
      continue
    # Older recorder versions lack the IMU rpy columns; without tilt there is no way to
    # exclude falls, and a fall would raise the walking bar. Skip rather than guess.
    if d.size == 0 or 'tau_est[0]' not in i or 'imu_rpy_r' not in i:
      continue
    hs = [i[c] for c in ('vel_x', 'vel_y', 'body_height', 'pos_x') if c in i]
    if hs and not np.all(d[:, hs] == 0.0):
      continue  # sim/bridge capture, not the robot
    t = d[:, 0]
    dt = float(np.median(np.diff(t)))
    dq = d[:, [i[f'dq[{j}]'] for j in range(27)]]
    tau = d[:, [i[f'tau_est[{j}]'] for j in range(27)]]
    tilt = np.maximum(np.abs(d[:, i['imu_rpy_r']]), np.abs(d[:, i['imu_rpy_p']]))
    g = max(1, int(round(GUARD_S / dt)))
    bad = np.convolve((tilt > TILT_OK).astype(float), np.ones(2 * g + 1), 'same') > 0
    walk = (np.abs(dq[:, LEG]).max(1) > WALK_DQ) & ~bad
    if walk.sum() < 200:
      continue
    tau_mx = np.maximum(tau_mx, np.abs(tau[walk]).max(0))
    dq_mx = np.maximum(dq_mx, np.abs(dq[walk]).max(0))
    secs += walk.sum() * dt
  return tau_mx, dq_mx, secs


def bridge_walking_max():
  """(dq per joint, torque headroom ratio per joint) over non-falling bridge sessions."""
  dq_mx = np.zeros(27)
  tau_ratio_mx = np.zeros(27)
  secs, files = 0.0, 0
  for p in sorted(glob.glob(str(REPO / 'logs/deploy_safety/*.csv'))):
    if p.endswith('_hrl.csv'):
      continue
    hdr = open(p).readline().strip().split(',')
    if 'trip_ratio' not in hdr or 'meas_dq0' not in hdr:
      continue
    meta = Path(p.replace('.csv', '_meta.json'))
    if not meta.exists():
      continue
    trips = {j['hw_id']: (j['tau_trip'], j['dq_trip'])
             for j in json.loads(meta.read_text())['joints']}
    dqc = sorted((c for c in hdr if c.startswith('meas_dq')), key=lambda c: int(c[7:]))
    use = ['t', 'trig_tilt', 'trig_fall', 'trip_ratio', 'trip_jid'] + dqc
    d = np.genfromtxt(p, delimiter=',', names=True, usecols=use, dtype=float)
    if ((d['trig_tilt'] != 0) | (d['trig_fall'] != 0)).any():
      continue  # a session that fell says nothing about the walking bar
    dq = np.stack([d[c] for c in dqc], axis=1)
    walk = np.abs(dq[:, LEG]).max(1) > WALK_DQ
    if walk.sum() < 200:
      continue
    dq_mx = np.maximum(dq_mx, np.abs(dq[walk]).max(0))
    # Torque headroom: on ticks where the logged max ratio exceeds anything dq can explain,
    # the binding quantity is torque on trip_jid. Recover it as a RATIO of that joint's
    # current threshold, which is all that is needed to rescale it.
    dq_ratio = np.abs(dq) / np.array([trips[j][1] for j in range(27)])
    excess = d['trip_ratio'] - dq_ratio.max(1)
    for k in np.where(walk & (excess > 1e-6))[0]:
      j = int(d['trip_jid'][k])
      if 0 <= j < 27:
        tau_ratio_mx[j] = max(tau_ratio_mx[j], float(d['trip_ratio'][k]))
    secs += walk.sum() * float(np.median(np.diff(d['t'])))
    files += 1
  return dq_mx, tau_ratio_mx, secs, files


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--headroom', type=float, default=1.3)
  args = ap.parse_args()

  h_tau, h_dq, h_s = hardware_walking_max()
  b_dq, b_tau_ratio, b_s, b_n = bridge_walking_max()
  print(f'hardware healthy walking: {h_s:.0f} s')
  print(f'bridge non-falling walking: {b_s:.0f} s over {b_n} sessions\n')

  # Current table, needed to turn the bridge torque RATIOS back into Nm.
  lim = (REPO / 'deploy/robots/h1_2/include/h1_2_limits.h').read_text()
  import re
  blk = re.search(r'h1_2_joint_trips\s*=\s*\{\{(.*?)\}\};', lim, re.S).group(1)
  cur = [(float(a), float(b)) for a, b in
         re.findall(r'\{\s*([\d.]+)f\s*,\s*([\d.]+)f\s*\}', blk)]
  b_tau = np.array([b_tau_ratio[j] * cur[j][0] for j in range(27)])

  print(f'{"group":<13}{"hw_tau":>8}{"br_tau":>8}{"hw_dq":>7}{"br_dq":>7}'
        f'{"TAU":>7}{"DQ":>7}{"binding":>10}')
  out = {}
  for name, ids in GROUPS.items():
    ht, bt = h_tau[ids].max(), b_tau[ids].max()
    hd, bd = h_dq[ids].max(), b_dq[ids].max()
    tau = float(np.ceil(max(ht, bt) * args.headroom / 5) * 5)
    dq = float(np.ceil(max(hd, bd) * args.headroom * 2) / 2)
    binding = 'bridge' if (bt > ht or bd > hd) else 'hardware'
    for j in ids:
      out[j] = (tau, dq)
    print(f'{name:<13}{ht:8.1f}{bt:8.1f}{hd:7.2f}{bd:7.2f}{tau:7.0f}{dq:7.1f}{binding:>10}')

  print(f'\n// generated by scripts/measure_joint_trip_thresholds.py --headroom {args.headroom}')
  print('static const std::array<H12JointTrip, 27> h1_2_joint_trips = {{')
  for j in range(27):
    print(f'    {{{out[j][0]:6.1f}f, {out[j][1]:5.1f}f}},  // {j:2d} {NAMES[j].upper()}')
  print('}};')


if __name__ == '__main__':
  main()
