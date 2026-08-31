#!/usr/bin/env python3
"""WP1b sweep -- FINAL, single continuous run on an uncontended GPU.

Grid : T in {0.40,0.45,0.50,0.55,0.60,0.65,0.70} (uniform 0.05; 0.65/0.70 added
       so the copper-loss interior optimum for 8/12 kg -- ~0.64-0.66 in WP1 --
       is bracketed, per step 5). vx=1.0 excluded (edge-degenerate).
Cell : payload {0,4,8,12} kg x vx {0.5,0.6,0.7}, R=8 independent play.py
       PROCESSES (seed 42+r), --eval-steps 1200, --num-envs 64.
Drift: the reference cell (T=0.50, p=0, vx=0.5) is re-run every DRIFT_EVERY
       cells and logged to wp1b_drift.csv, so a slow between-hours shift like the
       GPU-contention one (cot +/-6%) cannot hide inside the sweep.
Tree : run with cwd = the d077902 worktree (the commit the keeper trained on).

Guard: aborts on [SHIM]/structure-restore, on a nonzero exit, or if play.py's
       metric code changes mid-run (fingerprint of the cot/copper/err_vx lines).
"""
import csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/home/iams/ramlab_ws/src/unitree_rl_mjlab")            # for the checkpoint + data dir
WT = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/wt_v3"                # cwd for play.py (d077902)
CKPT = (REPO / "logs/rsl_rl/h1_2_velocity_a1_v2/"
        "2026-08-26_10-02-56_a1a_fullmirror_jacc1e-7_standing15_s42/model_10000.pt")
OUT_DIR = REPO / "data/2026-08-28-wp1b-payload-repeat"
CSV_PATH = OUT_DIR / "wp1b_sweep.csv"
DRIFT_PATH = OUT_DIR / "wp1b_drift.csv"
LOG_DIR = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/wp1b/final_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "WANDB_MODE": "disabled"}

T_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
PAYLOAD_GRID = [0, 4, 8, 12]
VX_GRID = [0.5, 0.6, 0.7]
REPEATS = 8
EVAL_STEPS = 1200
DRIFT_EVERY = 12
REF = (0.50, 0, 0.5)

COLUMNS = ["label", "hl_algorithm", "goal_components", "c", "seed", "err_vx", "err_vy",
           "err_yaw", "ss_err_vx", "ss_err_vy", "t90_s", "fall_rate", "mean_ep_len",
           "action_rate", "act_legs", "jacc", "jacc_legs", "jacc_arms", "jacc_legs_p95",
           "jacc_arms_p95", "ajit_legs", "ajit_legs_p95", "qjit_legs", "qjit_legs_p95",
           "act_legs_rad", "orient_dev", "height_dev", "ub_pose_dev", "ub_arm_vel",
           "mech_power_w", "gait_match", "cot", "mech_power_copper_w", "cot_copper",
           "omega_xy", "ss_vx_var", "stride_period_s",
           "T_pinned", "payload_kg", "vx_pinned", "repeat", "eval_steps"]
PLAY = WT / "scripts/play.py"
FP_TOKS = ("cot", "copper", "energy_eng", "powers", "achieved_vx_var", "omega_xy",
           "err_vx", "errs_vx", "qfrc_actuator", "joint_vel", "COPPER_LOSS_K",
           "dist_eng", "ss_vx_var", "MASS_G")


def play_fp():
  src = PLAY.read_text().splitlines()
  lo = next(i for i, l in enumerate(src) if "Headless deterministic benchmark" in l)
  hi = next(i for i, l in enumerate(src) if "[BENCH] {json.dumps(bench_out)}" in l)
  keep = [l for l in src[lo:hi + 1] if any(t in l for t in FP_TOKS)]
  return hashlib.md5("\n".join(keep).encode()).hexdigest()


def one(T, payload, vx, r, fp0):
  if play_fp() != fp0:
    print(f"ABORT: play.py metric code changed mid-run. Stopping.", flush=True)
    sys.exit(2)
  seed = 42 + r
  cmd = [sys.executable, "scripts/play.py", "Unitree-H1_2-Flat-A1",
         "--checkpoint-file", str(CKPT),
         "--num-envs", "64", "--eval-steps", str(EVAL_STEPS), "--eval-seeds", "1",
         "--eval-cmd-vx", str(vx), "--eval-cadence-period", str(T),
         "--eval-payload-kg", str(payload)]
  c0 = time.time()
  p = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env=ENV)
  dt = time.time() - c0
  tag = f"T{T}_p{payload}_vx{vx}_r{r}"
  (LOG_DIR / f"{tag}.log").write_text(p.stdout + "\n--STDERR--\n" + p.stderr)
  if p.returncode != 0 or "[SHIM]" in p.stdout or "structure-restore" in p.stdout.lower():
    print(f"ABORT at {tag}: rc={p.returncode}. See {LOG_DIR/tag}.log", flush=True)
    sys.exit(1)
  sl = [l for l in p.stdout.splitlines() if l.startswith("[BENCH_SEED]")]
  if len(sl) != 1:
    print(f"ABORT at {tag}: {len(sl)} [BENCH_SEED] lines. See {LOG_DIR/tag}.log", flush=True)
    sys.exit(1)
  rec = json.loads(sl[0][len("[BENCH_SEED] "):])
  rec.update(T_pinned=T, payload_kg=payload, vx_pinned=vx, repeat=r, eval_steps=EVAL_STEPS)
  miss = [c for c in COLUMNS if c not in rec]
  if miss:
    print(f"ABORT at {tag}: [BENCH_SEED] missing {miss}", flush=True)
    sys.exit(1)
  return rec, dt


def main():
  fp0 = play_fp()
  print(f"worktree HEAD: ", end="", flush=True)
  subprocess.run(["git", "-C", str(WT), "log", "--oneline", "-1"])
  print(f"play.py metric fingerprint {fp0}", flush=True)

  cells = [(T, p, vx) for vx in VX_GRID for p in PAYLOAD_GRID for T in T_GRID]
  total = len(cells) * REPEATS
  print(f"{len(cells)} cells x {REPEATS} = {total} processes + drift probes every "
        f"{DRIFT_EVERY} cells", flush=True)

  fcsv = open(CSV_PATH, "w", newline="")
  w = csv.DictWriter(fcsv, fieldnames=COLUMNS, extrasaction="ignore")
  w.writeheader()
  fdr = open(DRIFT_PATH, "w", newline="")
  wdr = csv.DictWriter(fdr, fieldnames=["after_cell", "wallclock_min", "cot", "cot_copper",
                                        "err_vx", "ss_vx_var", "mech_power_w"])
  wdr.writeheader()

  t0 = time.time()
  done = 0

  def drift(after_cell):
    rec, _ = one(*REF, 0, fp0)
    wdr.writerow(dict(after_cell=after_cell, wallclock_min=round((time.time()-t0)/60, 1),
                      cot=rec["cot"], cot_copper=rec["cot_copper"], err_vx=rec["err_vx"],
                      ss_vx_var=rec["ss_vx_var"], mech_power_w=rec["mech_power_w"]))
    fdr.flush()
    print(f"  [drift@{after_cell}] cot={rec['cot']:.5f} cot_copper={rec['cot_copper']:.3f} "
          f"err_vx={rec['err_vx']:.5f}", flush=True)

  drift(0)
  for ci, (T, payload, vx) in enumerate(cells):
    for r in range(REPEATS):
      rec, dt = one(T, payload, vx, r, fp0)
      w.writerow(rec)
      fcsv.flush()
      done += 1
      el = time.time() - t0
      eta = (total - done) * el / done
      print(f"[{done}/{total}] T{T} p{payload} vx{vx} r{r}  {dt:.1f}s  "
            f"elapsed={el/60:.1f}min ETA={eta/60:.1f}min", flush=True)
    if (ci + 1) % DRIFT_EVERY == 0:
      drift(ci + 1)
  drift(len(cells))

  fcsv.close()
  fdr.close()
  print(f"\nDONE. {done} rows -> {CSV_PATH}. wall {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
  main()
