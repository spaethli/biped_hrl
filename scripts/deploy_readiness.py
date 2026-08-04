"""Hardware-readiness pipeline: one command, one verdict.

Chains the existing tools -- it computes NO metrics of its own, every number comes from a
marker another tool already prints. Replaces ad-hoc bridge sessions, which is where the
2026-08-03 failures came from: a policy nobody had checked, a rate read off a column that
was never a clock, and a session called good from its FSM transition log.

  provenance -> sim -> replica -> bridge (candidate + A0 control) -> analyze

EXIT CODES        0 GO | 3 GO-WITH-CAVEAT | 1 NO-GO | 2 INFRASTRUCTURE
FAIL-CLOSED RULE  never report GO from a stage that could not be scored. An unrunnable or
                  unscoreable stage is exit 2, never a pass. A fall is read from CSV
                  evidence (height / trig_fall / trig_tilt), never from FSM transitions.
FAILURE HANDLING  provenance aborts immediately, before any process starts -- that is the
                  point of the gate. Every later stage runs to completion even after a
                  failure, so one invocation yields the whole evidence set, including the
                  A0 control arm. Stopping at the candidate's fall would throw away the
                  attribution step that actually resolved 2026-08-03.

GATES (blocking)          fall_rate == 0 on the bench and both holds; zero safety-filter
                          engagement anywhere in the bridge; no fall in any bridge phase
                          INCLUDING transition windows; estimator inside the pre-registered
                          bands on BOTH instruments.
REPORTED, NEVER GATED     leg action rate, transition overshoot, trig_joint rate/magnitude,
                          whole-body act_rate, stride period, cmd-0 drift, loop rate.
EXPLICITLY UNKNOWN        what leg action rate is unsafe; what trig_joint rate is acceptable.

Usage:
  python scripts/deploy_readiness.py Unitree-H1_2-Flat-A1 --checkpoint-file <pt>
  python scripts/deploy_readiness.py ... --stages provenance,sim   # subset
  python scripts/deploy_readiness.py ... --refresh-control         # force the A0 arm
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_gate_analyzer as dga  # noqa: E402
import deploy_provenance as prov  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OPT_CFG = Path("/opt/unitree_mujoco/simulate/config.yaml")
EXPECTED_SCENE = "scene_stress.xml"  # the shipped Unitree plant; see F6 / deployment.md
CONTROL_CACHE = REPO / "logs" / "deploy_safety" / ".a0_control_cache.json"

# Estimator bands, metric max(vx, vy). Sim bands pre-registered 2026-08-01. Bridge bands
# were pre-registered at 2x sim on 2026-08-04 BEFORE the first corrected measurement, then
# tightened to 1.5x after it came in at 0.0139/0.0288 -- tightening on a pass is not
# fitting the gate to the candidate, and the incumbent still clears it 2.6x.
BANDS = {"sim": (0.05, 0.15), "bridge": (0.075, 0.225)}
# Leg action rate reference points. NOT a gate: what is unsafe is unknown. And every one of
# these was measured on the RETIRED training-proximate plant, so they do not transfer to
# the shipped scene until re-baselined -- reported with that caveat attached, never compared
# silently.
LEG_AR_REF = {"A0 (hardware-proven)": 0.5908, "keeper arm4d (bridge-passed)": 0.7552,
              "arm1 (never bridged)": 0.8125}


class Result:
  """One gate outcome. `blocking` False means it is reported and never affects the verdict."""

  def __init__(self, stage, name, status, detail, blocking=True):
    self.stage, self.name, self.status = stage, name, status  # PASS|FAIL|CAVEAT|INFRA|INFO
    self.detail, self.blocking = detail, blocking

  def as_dict(self):
    return {"stage": self.stage, "gate": self.name, "status": self.status,
            "detail": self.detail, "blocking": self.blocking}


def marker(text: str, tag: str) -> dict | None:
  """Last `[TAG] {json}` line from a tool's stdout. Last, not first: multi-seed runs print
  per-seed lines then the aggregate."""
  hits = re.findall(rf"^\[{tag}\]\s*(\{{.*\}})\s*$", text, re.M)
  return json.loads(hits[-1]) if hits else None


def run(cmd, **kw) -> tuple[int, str]:
  p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, **kw)
  return p.returncode, p.stdout + p.stderr


def band_verdict(value: float, which: str) -> str:
  go, caveat = BANDS[which]
  return "PASS" if value <= go else "CAVEAT" if value <= caveat else "FAIL"


# ---------------------------------------------------------------- stages


def stage_provenance(args, results):
  """P1-P6. The only stage that aborts the chain: running a bridge against a stranger is
  exactly what wasted 2026-08-03, and every downstream number would be attributed wrong."""
  if OPT_CFG.exists():
    scene = re.search(r'^robot_scene:\s*"([^"]+)"', OPT_CFG.read_text(), re.M)
    live = scene.group(1) if scene else "<unreadable>"
    results.append(Result("provenance", "P6 scene", "PASS" if live == EXPECTED_SCENE
                          else "INFRA", f"robot_scene={live} (expected {EXPECTED_SCENE})"))
  else:
    results.append(Result("provenance", "P6 scene", "INFRA", f"{OPT_CFG} not found"))

  if args.stage_policy:
    try:
      m = prov.stage(args.checkpoint_file.resolve(), args.policy_dir)
      results.append(Result("provenance", "staging", "INFO",
                            f"staged {m['source_run']}", blocking=False))
    except SystemExit as e:
      results.append(Result("provenance", "staging", "INFRA", str(e)))
      return

  rep = prov.check(args.policy_dir, expect_checkpoint=args.checkpoint_file)
  results.append(Result("provenance", "P1-P4 identity/dims/metadata",
                        "PASS" if rep["ok"] else "FAIL",
                        "; ".join(rep["problems"]) or f"source_run={rep['source_run']}"))

  rc, out = run(["python", "scripts/onnx_parity.py", "--task", args.task,
                 "--checkpoint-file", str(args.checkpoint_file),
                 "--onnx-dir", str(args.policy_dir / "exported")])
  # Read the tool's own [PARITY] marker rather than scraping numbers off the table: a
  # regex over trailing floats picks up unrelated columns and reports a confident wrong
  # magnitude, which is worse than reporting none.
  p = marker(out, "PARITY")
  worst = max((v for v in _flatten_numbers(p)), default=None) if p else None
  results.append(Result("provenance", "P5 parity vs DEPLOYED onnx",
                        "PASS" if rc == 0 else "FAIL",
                        f"onnx_parity rc={rc}"
                        + (f", max|d|={worst:.2g}" if worst is not None else "")))


def _flatten_numbers(obj):
  """Every numeric leaf of the [PARITY] json, whatever its nesting/naming."""
  if isinstance(obj, (int, float)) and not isinstance(obj, bool):
    yield float(obj)
  elif isinstance(obj, dict):
    for k, v in obj.items():
      if "max" in k.lower() or isinstance(v, (dict, list)):
        yield from _flatten_numbers(v)
  elif isinstance(obj, list):
    for v in obj:
      yield from _flatten_numbers(v)


def stage_sim(args, results):
  base = ["python", "scripts/play.py", args.task, "--checkpoint-file",
          str(args.checkpoint_file), "--num-envs", "64", "--eval-seeds", "2"]

  rc, out = run(base + ["--eval-steps", "600"])
  b = marker(out, "BENCH")
  if b is None:
    results.append(Result("sim", "S2.1 bench", "INFRA", f"no [BENCH] marker (rc={rc})"))
  else:
    fr = float(b.get("fall_rate", b.get("fall_rate_mean", 1.0)))
    results.append(Result("sim", "S2.1 bench fall_rate", "PASS" if fr == 0 else "FAIL",
                          f"fall_rate={fr}"))

  rc, out = run(base + ["--diagnose-action-rate", "600"])
  a = marker(out, "ARDIAG")
  refs = ", ".join(f"{k} {v}" for k, v in LEG_AR_REF.items())
  results.append(Result("sim", "S2.2 leg action rate", "INFO",
                        f"{a if a else 'no [ARDIAG]'} | refs (RETIRED plant, do not compare "
                        f"without re-baselining): {refs}", blocking=False))

  for vx in ("0.5", "1.0"):
    rc, out = run(base + ["--eval-steps", "600", "--eval-cmd-vx", vx])
    b = marker(out, "BENCH")
    if b is None:
      results.append(Result("sim", f"S2.3 hold vx={vx}", "INFRA", "no [BENCH] marker"))
      continue
    fr = float(b.get("fall_rate", b.get("fall_rate_mean", 1.0)))
    results.append(Result("sim", f"S2.3 hold vx={vx} fall_rate",
                          "PASS" if fr == 0 else "FAIL", f"fall_rate={fr}"))

  rc, out = run(base + ["--check-vel-increment", "480"])
  v = marker(out, "VELINC")
  if v is None:
    results.append(Result("sim", "S2.4 estimator (sim)", "INFRA", "no [VELINC] marker"))
  else:
    g = _velinc_gate(v)
    results.append(Result("sim", "S2.4 estimator (sim)", band_verdict(g, "sim")
                          if g is not None else "INFRA",
                          f"max(vx,vy)={g} vs {BANDS['sim']}"))


def _velinc_gate(v: dict):
  """The DEPLOYABLE-rung max(vx,vy) from [VELINC] — i.e. `gate_rms`, which play.py already
  computes from the waist-corrected @200 Hz rung (a conservative stand-in for run()'s
  1 kHz). The other rungs in that json are diagnostics and two of them use privileged
  state no hardware has.

  Returns None (-> INFRA) if the key is absent. It must NOT guess: an earlier version fell
  back to max() over every vx/vy-looking key and scored the RAW rung at 0.2476 against a
  true 0.0255, which would have failed a policy that ships. Refusing to score beats scoring
  the wrong thing -- the same fail-closed rule this pipeline applies everywhere else.
  """
  g = v.get("gate_rms")
  return round(float(g), 4) if isinstance(g, (int, float)) else None


def stage_replica(args, results):
  rc, out = run(["python", "scripts/bridge_replica.py", "--policy",
                 "hrl" if "A1" in args.task else "a0",
                 "--onnx-dir", str(args.policy_dir / "exported"), "--delay-ms", "4"])
  fell = "fall" in out.lower() and "no fall" not in out.lower()
  results.append(Result("replica", "S3 replica @4ms",
                        "INFRA" if rc != 0 else ("FAIL" if fell else "PASS"),
                        f"rc={rc}"))


def control_fingerprint(args) -> str:
  """What the cached A0 control arm is valid for. A rebuilt binary is a DIFFERENT shared
  layer -- articulation, joint_offset, safety filter -- which is exactly what the control
  vouches for, so it must invalidate. No time expiry (the user's call): staleness that a
  fingerprint cannot see is not what this guards against."""
  parts = []
  for p in [args.a0_dir / "exported" / "policy.onnx",
            REPO / "deploy/robots/h1_2/build/h1_2_ctrl",
            args.a0_dir / "params" / "deploy.yaml",
            Path("/opt/unitree_mujoco/unitree_robots/h1_2") / EXPECTED_SCENE]:
    parts.append(prov.md5(p) if p.exists() else f"<missing:{p.name}>")
  parts.append(args.seq)
  return hashlib.md5("|".join(parts).encode()).hexdigest()


def run_bridge_arm(policy, tag, args) -> Path | None:
  rc, out = run(["python", "scripts/bridge_session.py", "--policy", policy,
                 "--seq", args.seq, "--tag", tag, "--no-analyze"])
  m = re.search(r"\[log\]\s*(\S+)\.csv", out)
  return Path(m.group(1) + ".csv") if m else None


def stage_bridge(args, results):
  csvs = {}
  csvs["candidate"] = run_bridge_arm("hrl" if "A1" in args.task else "a0",
                                     f"{args.tag}_candidate", args)
  fp = control_fingerprint(args)
  cache = json.loads(CONTROL_CACHE.read_text()) if CONTROL_CACHE.exists() else {}
  if not args.refresh_control and cache.get("fingerprint") == fp \
          and Path(cache.get("csv", "")).exists():
    csvs["a0_control"] = Path(cache["csv"])
    results.append(Result("bridge", "A0 control arm", "INFO",
                          f"CACHED from {cache['captured']} (fingerprint {fp[:8]})",
                          blocking=False))
  else:
    csvs["a0_control"] = run_bridge_arm("a0", f"{args.tag}_a0control", args)
    if csvs["a0_control"]:
      CONTROL_CACHE.parent.mkdir(parents=True, exist_ok=True)
      CONTROL_CACHE.write_text(json.dumps(
        {"fingerprint": fp, "csv": str(csvs["a0_control"]),
         "captured": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
  return csvs


def stage_analyze(csvs, results):
  """Score both arms. The A0 column is what makes a candidate failure attributable: A0
  exercises the shared layer without the HRL path, so A0-clean + candidate-fails isolates
  the fault to the HRL path or the policy itself."""
  for arm, path in csvs.items():
    if path is None or not Path(path).exists():
      results.append(Result("analyze", f"{arm}", "INFRA", "no CSV produced"))
      continue
    out = dga.analyze_base_csv(str(path))
    blocking = arm == "candidate"

    if out.get("band_released") is False:
      results.append(Result("analyze", f"{arm} band release", "INFRA",
                            "robot never translated under a held command -- harness "
                            "artifact, tells you nothing about the policy"))
      continue
    alpha = max((s["alpha_max"] for s in out["segments"]), default=0.0)
    results.append(Result("analyze", f"{arm} safety filter",
                          "PASS" if alpha == 0 else "FAIL",
                          f"alpha_max={alpha}", blocking=blocking))
    results.append(Result("analyze", f"{arm} falls",
                          "PASS" if not out["any_fall"] else "FAIL",
                          f"any_fall={out['any_fall']}", blocking=blocking))
    results.append(Result("analyze", f"{arm} transition falls",
                          "PASS" if not out["any_transition_fall"] else "FAIL",
                          f"{len(out['transitions'])} transitions scored, "
                          f"any_fall={out['any_transition_fall']}", blocking=blocking))
    worst = max((w["peak_overshoot"] for w in out["transitions"]), default=0.0)
    results.append(Result("analyze", f"{arm} transition overshoot", "INFO",
                          f"peak {worst} (no grounded threshold)", blocking=False))
    jc = out.get("joint_clamp", {})
    if jc.get("available"):
      top = jc["per_joint"][0] if jc["per_joint"] else None
      results.append(Result("analyze", f"{arm} trig_joint", "INFO",
                            f"any-joint {jc['any_joint_rate']:.1%}"
                            + (f", worst {top['joint']} {top['rate']:.1%} "
                               f"max {top['max_past_stop_rad']:+.3f} rad past stop"
                               if top else "")
                            + f" | acceptable rate {jc['acceptable_rate']}", blocking=False))
    for d in out.get("stand_drift", []):
      results.append(Result("analyze", f"{arm} cmd-0 drift", "INFO",
                            f"{d['drift_m']} m at {d['heading_deg']}deg over {d['dur_s']}s",
                            blocking=False))
    if out.get("loop_rate_hz"):
      results.append(Result("analyze", f"{arm} loop rate", "INFO",
                            f"{out['loop_rate_hz']} Hz measured", blocking=False))

    hrl = Path(str(path).replace(".csv", "_hrl.csv"))
    if hrl.exists():
      est = _bridge_estimator(hrl)
      if est is not None:
        results.append(Result("analyze", f"{arm} estimator (bridge)",
                              band_verdict(est, "bridge"),
                              f"max(vx,vy)={est} vs {BANDS['bridge']}", blocking=blocking))


def _bridge_estimator(hrl_csv: Path):
  import numpy as np
  d = np.genfromtxt(str(hrl_csv), delimiter=",", names=True)
  if not {"est_vx", "gt_vx"} <= set(d.dtype.names or ()):
    return None
  return round(float(max(np.sqrt(((d["est_vx"] - d["gt_vx"]) ** 2).mean()),
                         np.sqrt(((d["est_vy"] - d["gt_vy"]) ** 2).mean()))), 4)


# ---------------------------------------------------------------- verdict


def verdict(results) -> tuple[str, int]:
  blocking = [r for r in results if r.blocking]
  if any(r.status == "INFRA" for r in blocking):
    return "INFRASTRUCTURE — a stage could not be run or scored; this is NOT a pass", 2
  if any(r.status == "FAIL" for r in blocking):
    return "NO-GO", 1
  if any(r.status == "CAVEAT" for r in blocking):
    return "GO-WITH-CAVEAT — a gate landed in its middle band; your call", 3
  return "GO", 0


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("task")
  ap.add_argument("--checkpoint-file", type=Path, required=True)
  ap.add_argument("--policy-dir", type=Path, default=prov.HRL_DIR)
  ap.add_argument("--a0-dir", type=Path, default=prov.A0_DIR)
  ap.add_argument("--stages", default="provenance,sim,replica,bridge,analyze")
  ap.add_argument("--tag", default="readiness")
  ap.add_argument("--seq", default=None, help="bridge sequence (default: the full battery)")
  ap.add_argument("--refresh-control", action="store_true")
  ap.add_argument("--no-stage-policy", dest="stage_policy", action="store_false",
                  help="validate what is already in exported/ instead of staging the "
                       "checkpoint into it")
  args = ap.parse_args()
  if args.seq is None:
    sys.path.insert(0, str(REPO / "scripts"))
    import bridge_session
    args.seq = bridge_session.DEFAULT_SEQ

  stages = args.stages.split(",")
  results: list[Result] = []
  print(f"[READINESS] task={args.task} checkpoint={args.checkpoint_file}")

  if "provenance" in stages:
    stage_provenance(args, results)
    if any(r.status in ("FAIL", "INFRA") and r.blocking
           for r in results if r.stage == "provenance"):
      _report(results)
      print("[READINESS] ABORTED at provenance — no process was started. Fix what is "
            "deployed before spending a bridge run on it.")
      msg, code = verdict(results)
      print(f"[READINESS] {msg}")
      return code

  if "sim" in stages:
    stage_sim(args, results)
  if "replica" in stages:
    stage_replica(args, results)
  csvs = stage_bridge(args, results) if "bridge" in stages else {}
  if "analyze" in stages and csvs:
    stage_analyze(csvs, results)

  _report(results)
  msg, code = verdict(results)
  print(f"[READINESS] {msg}")
  print("[READINESS] UNKNOWN, surfaced not gated: what leg action rate is actually unsafe; "
        "what trig_joint rate is acceptable.")
  summary = {"verdict": msg, "exit": code, "gates": [r.as_dict() for r in results]}
  print(f"[READINESS] {json.dumps(summary, default=str)}")
  return code


def _report(results):
  print(f"\n{'stage':<12}{'gate':<34}{'status':<9}detail")
  for r in results:
    flag = "" if r.blocking else " (report-only)"
    print(f"{r.stage:<12}{r.name:<34}{r.status:<9}{r.detail}{flag}")


if __name__ == "__main__":
  sys.exit(main())
