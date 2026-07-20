#!/usr/bin/env python3
"""Offline analysis for `read_all_joints` CSV output: onboard velocity/height estimator
quality (bias, noise, base-vs-other-frame check) and IMU specific-force convention.

Generic sensor validation, not tied to any one gate plan -- point it at a log from
`ros2 run h1_2_low_level_controller read_all_joints` (writes to
~/ramlab_ws/trajectories/all_joints_<ts>.csv) and mark the time windows you want checked.
No network access needed; only numpy.

Usage:
  python3 robot_estimator_check.py ~/ramlab_ws/trajectories/all_joints_<ts>.csv \
      --static START END --walk START END --rotate START END --distance 2.5

All *_START/_END are seconds into the log (the "time" column), marking windows you
noted during the session (e.g. "static standing" or "walked a measured line"). Omit
any window you didn't record.

Checks:
  --static  window: velocity/height bias + noise floor at true-zero velocity, AND the
            IMU specific-force convention (a_world_z = R*a_imu.z - 9.81, should be ~0
            while standing still -- this is the assumption behind a downward-accel
            fall-trigger; a real IMU that reports net acceleration instead of specific
            force would show a ~+-9.81 offset here instead).
  --rotate  window: pure-yaw rotation in place, no translation. Checks whether the
            velocity estimate is correlated with yaw rate -- a red flag that it's
            computed off the true rotation center (e.g. torso/IMU-mounted with an
            offset) rather than genuinely base-frame.
  --walk    window (+ --distance): integrates vel_x over a measured walk to bias-check
            the estimate against ground truth (tape measure / stopwatch).
"""
import argparse
import numpy as np


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def window(d, lo, hi):
    m = (d["time"] >= lo) & (d["time"] <= hi)
    return d[m]


def specific_force_check(d, lo, hi):
    w = window(d, lo, hi)
    if len(w) == 0:
        print("[specific-force] no rows in window"); return
    qw, qx, qy, qz = w["imu_quat_w"], w["imu_quat_x"], w["imu_quat_y"], w["imu_quat_z"]
    ax, ay, az = w["imu_acc_x"], w["imu_acc_y"], w["imu_acc_z"]

    def rot_z(qw, qx, qy, qz, ax, ay, az):
        # world-frame z component of R(q) @ [ax,ay,az]
        return (2 * (qx * qz + qw * qy) * ax + 2 * (qy * qz - qw * qx) * ay
                + (1 - 2 * (qx * qx + qy * qy)) * az)

    world_z = rot_z(qw, qx, qy, qz, ax, ay, az) - 9.81
    print(f"[specific-force] window {lo}-{hi}s, n={len(w)}")
    print(f"    a_world_z: mean={world_z.mean():.3f}  std={world_z.std():.3f}  "
          f"min={world_z.min():.3f}  max={world_z.max():.3f}")
    print("    PASS if mean is close to 0 (a few m/s^2 tolerance) and min never drops "
          "below -7 sustained -- that's the fall-trigger threshold. If mean sits near "
          "-9.81 or +9.81 instead of 0, the real IMU does NOT report specific force the "
          "way a sim-derived fall-trigger assumed -- the trigger formula needs the "
          "opposite sign convention before it's trusted on hardware.")


def velocity_bias_check(d, lo, hi):
    w = window(d, lo, hi)
    if len(w) == 0:
        print("[velocity-bias] no rows in window"); return
    for k in ("vel_x", "vel_y", "vel_z"):
        print(f"[velocity-bias] {k}: mean(bias)={w[k].mean():.4f}  std(noise)={w[k].std():.4f}")
    print(f"[velocity-bias] body_height: mean={w['body_height'].mean():.4f}  "
          f"std={w['body_height'].std():.4f}")
    print("    PASS if |bias| is small relative to whatever noise envelope the "
          "consuming policy was trained/tested against, and height sits near the "
          "robot's true standing height.")


def rotation_frame_check(d, lo, hi):
    w = window(d, lo, hi)
    if len(w) == 0:
        print("[rotation-frame] no rows in window"); return
    vx, vy, wz = w["vel_x"], w["vel_y"], w["yaw_speed"]
    corr_x = np.corrcoef(vx, wz)[0, 1] if wz.std() > 1e-6 else float("nan")
    corr_y = np.corrcoef(vy, wz)[0, 1] if wz.std() > 1e-6 else float("nan")
    print(f"[rotation-frame] pure-yaw window {lo}-{hi}s, n={len(w)}")
    print(f"    vel_x: mean={vx.mean():.4f} std={vx.std():.4f} | "
          f"vel_y: mean={vy.mean():.4f} std={vy.std():.4f} | yaw_speed mean={wz.mean():.4f}")
    print(f"    corr(vel_x, yaw_speed)={corr_x:.3f}  corr(vel_y, yaw_speed)={corr_y:.3f}")
    print("    PASS if vel_x/vel_y stay small and roughly UNcorrelated with yaw_speed. "
          "A clear correlation (|corr| notably above ~0.3-0.4) means the estimate is "
          "off-center from the true rotation axis -- the base-vs-other-frame trap. That "
          "matters more than a small nonzero static bias alone.")


def walk_distance_check(d, lo, hi, distance_m):
    w = window(d, lo, hi)
    if len(w) < 2:
        print("[walk-distance] not enough rows in window"); return
    t = w["time"]
    vx = w["vel_x"]
    dist_est = np.sum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))  # manual trapezoid rule
    # (avoids depending on np.trapz vs np.trapezoid across numpy versions)
    duration = t[-1] - t[0]
    print(f"[walk-distance] window {lo}-{hi}s, duration={duration:.2f}s")
    print(f"    integrated distance from vel_x = {dist_est:.3f} m")
    if distance_m is not None:
        err = dist_est - distance_m
        print(f"    measured (tape) distance = {distance_m:.3f} m  ->  error = {err:+.3f} m "
              f"({100 * err / distance_m:+.1f}%)")
    print("    PASS if the integrated estimate is reasonably close to the tape-measured "
          "distance (systematic drift here is a bias problem, not just noise).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--static", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--rotate", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--walk", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--distance", type=float, default=None, help="tape-measured walk distance (m)")
    args = p.parse_args()

    d = load(args.csv)
    print(f"Loaded {len(d)} rows, time range {d['time'][0]:.1f}-{d['time'][-1]:.1f}s\n")

    if args.static:
        specific_force_check(d, *args.static)
        print()
        velocity_bias_check(d, *args.static)
        print()
    if args.rotate:
        rotation_frame_check(d, *args.rotate)
        print()
    if args.walk:
        walk_distance_check(d, *args.walk, args.distance)
        print()


if __name__ == "__main__":
    main()
