#!/usr/bin/env python3
"""WP1b step 1 -- MEASURE the noise floor (do not assume +/-2.6%).

Same cell everywhere: T=0.50, payload=0 kg, vx=0.5. Three probes:

  A. one process, --eval-seeds 6  (seeds 42..47): internal-seed spread, the
     cheapest floor and the one that matches how each sweep cell will run.
  B. 6 independent processes, --eval-seeds 1 (seed 42 each): process-to-process
     spread. THIS is the floor that matters for T* extraction, because the 5
     T-points a bowl is fit through are 5 separate play.py processes, each with
     its own GPU-nondeterminism draw.
  C. one process, --eval-seeds 2 --eval-steps {600,1200,2400}: does a longer
     rollout tighten cot / ss_vx_var (the eval-steps lever for step 2).

Reports the across-run coefficient of variation (sample std / mean, ddof=1) of
cot, cot_copper, err_vx, ss_vx_var -- the four objectives step 6 adjudicates.
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/iams/ramlab_ws/src/unitree_rl_mjlab")
CKPT = (REPO / "logs/rsl_rl/h1_2_velocity_a1_v2/"
        "2026-08-26_10-02-56_a1a_fullmirror_jacc1e-7_standing15_s42/model_10000.pt")
LOG_DIR = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/wp1b/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp/wp1b/step1_result.json"
ENV = {**os.environ, "WANDB_MODE": "disabled"}

T, PAYLOAD, VX = 0.50, 0, 0.5
METRICS = ["cot", "cot_copper", "err_vx", "ss_vx_var"]


def run(tag, seeds, steps):
  cmd = [
    sys.executable, "scripts/play.py", "Unitree-H1_2-Flat-A1",
    "--checkpoint-file", str(CKPT),
    "--num-envs", "64", "--eval-steps", str(steps), "--eval-seeds", str(seeds),
    "--eval-cmd-vx", str(VX), "--eval-cadence-period", str(T),
    "--eval-payload-kg", str(PAYLOAD),
  ]
  t0 = time.time()
  proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=ENV)
  dt = time.time() - t0
  (LOG_DIR / f"{tag}.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
  if proc.returncode != 0:
    print(f"ABORT {tag}: play.py exited {proc.returncode}. See {LOG_DIR/tag}.log")
    sys.exit(1)
  if "[SHIM]" in proc.stdout or "structure-restore" in proc.stdout.lower():
    print(f"ABORT {tag}: [SHIM]/structure-restore warning. See {LOG_DIR/tag}.log")
    sys.exit(1)
  lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("[BENCH_SEED]")]
  recs = [json.loads(ln[len("[BENCH_SEED] "):]) for ln in lines]
  print(f"  {tag}: {len(recs)} seed-rows in {dt:.1f}s")
  return recs


def cv_table(label, recs):
  print(f"\n=== {label}  (n={len(recs)}) ===")
  out = {}
  for m in METRICS:
    xs = [r[m] for r in recs]
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    cv = sd / mean if mean else float("nan")
    out[m] = {"mean": mean, "std": sd, "cv_pct": 100 * cv,
              "min": min(xs), "max": max(xs),
              "range_pct_of_min": 100 * (max(xs) - min(xs)) / min(xs) if min(xs) else float("nan")}
    print(f"  {m:12s} mean={mean:.6g}  std={sd:.3g}  CV={100*cv:.3f}%  "
          f"range={100*(max(xs)-min(xs))/min(xs):.3f}% of min")
  return out


def main():
  results = {"cell": {"T": T, "payload_kg": PAYLOAD, "vx": VX}, "probes": {}}

  print("PROBE A: one process, --eval-seeds 6 (internal-seed spread)")
  recs_a = run("A_seeds6_steps600", seeds=6, steps=600)
  results["probes"]["A_internal_seeds"] = cv_table("A  internal seeds 42..47, 600 steps", recs_a)

  print("\nPROBE B: 6 independent processes, seed 42 each (process-to-process spread)")
  recs_b = []
  for i in range(6):
    recs_b += run(f"B_proc{i}_seed42_steps600", seeds=1, steps=600)
  results["probes"]["B_process_to_process"] = cv_table(
    "B  6 independent processes, seed 42, 600 steps", recs_b)

  print("\nPROBE C: eval-steps lever (600 / 1200 / 2400, 2 internal seeds)")
  c_tab = {}
  for steps in (600, 1200, 2400):
    recs_c = run(f"C_seeds2_steps{steps}", seeds=2, steps=steps)
    c_tab[str(steps)] = cv_table(f"C  2 seeds, {steps} steps", recs_c)
  results["probes"]["C_eval_steps"] = c_tab

  OUT.write_text(json.dumps(results, indent=2))
  print(f"\nwrote {OUT}")


if __name__ == "__main__":
  main()
