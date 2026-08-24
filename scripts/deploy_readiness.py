"""Hardware-readiness pipeline: one command, one verdict.

Chains the existing tools -- it computes NO metrics of its own, every number comes from a
marker another tool already prints. Replaces ad-hoc bridge sessions, which is where the
2026-08-03 failures came from: a policy nobody had checked, a rate read off a column that
was never a clock, and a session called good from its FSM transition log.

  provenance -> sim -> bridge (candidate + A0 control) -> analyze

  (`replica` exists as an opt-in stage but is NOT in the default chain -- see stage_replica.)

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
BANDS = {"sim": (0.05, 0.15), "bridge": (0.075, 0.225),
         # Leg odometry (the HL's ABSOLUTE velocity) is a different instrument from the IMU
         # increment above and cannot share its band: it is ~5x less accurate BY DESIGN, and
         # that is fine, because the HL filters it over a whole window at 6.25 Hz while the
         # LL sees the increment every step. Bounds are the measured bridge values, not
         # aspirations -- retrospective scoring of the three 2026-08-05 gt-arm captures
         # (arm4d, shipped plant, ~400 moving windows each) gave vx 0.067-0.071 and
         # vy 0.099-0.104. PASS is set just above that, CAVEAT at 2x.
         # ⚠ The vy figure is 2.1x the sim bench's 0.0486 and that is NOT an implementation
         # gap: bridge vy TRACKING is 0.055 against sim's 0.041, and the whole excess is the
         # c-averaging lag (sim's lateral velocity barely moves inside 0.16 s, the bridge's
         # moves a lot). Sim and bridge measure different plants; the bridge sets the bar.
         "legodom": (0.115, 0.230)}
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


def select_scene(name: str) -> str | None:
  """Point the bridge at `name` by rewriting /opt's robot_scene, and return what it reads
  back. The pipeline SETS the plant rather than only asserting it, because the two ways of
  getting this wrong are symmetric: a gate that expects the shipped scene while /opt still
  holds the retired one scores the wrong plant, and a plant A/B driven by hand edits drifts
  out of step with the gate on the first forgotten edit. Setting then re-reading makes the
  two impossible to disagree. Returns None if the file is unwritable/unparseable."""
  if not OPT_CFG.exists():
    return None
  text = OPT_CFG.read_text()
  new, n = re.subn(r'^(robot_scene:\s*)"[^"]+"', rf'\g<1>"{name}"', text, count=1, flags=re.M)
  if n and new != text:
    OPT_CFG.write_text(new)
  m = re.search(r'^robot_scene:\s*"([^"]+)"', OPT_CFG.read_text(), re.M)
  return m.group(1) if m else None


def stage_provenance(args, results):
  """P1-P6. The only stage that aborts the chain: running a bridge against a stranger is
  exactly what wasted 2026-08-03, and every downstream number would be attributed wrong."""
  if OPT_CFG.exists():
    live = select_scene(args.scene) if args.scene else None
    if live is None:
      scene = re.search(r'^robot_scene:\s*"([^"]+)"', OPT_CFG.read_text(), re.M)
      live = scene.group(1) if scene else "<unreadable>"
    want = args.scene or EXPECTED_SCENE
    note = "" if args.scene is None else " (set by --scene)"
    results.append(Result("provenance", "P6 scene", "PASS" if live == want
                          else "INFRA", f"robot_scene={live} (expected {want}){note}"))
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
  """Headless python re-implementation of the bridge loop. Cheap plant/latency screen; it
  CANNOT see C++ defects (it does not run `h1_2_ctrl` at all), so it screens, never clears.

  `--scene` must be passed explicitly: bridge_replica defaults to `scene.xml`, which is the
  RETIRED training-proximate plant, so omitting it silently screens against a plant we no
  longer gate on.
  """
  rc, out = run(["python", "scripts/bridge_replica.py", "--policy",
                 "hrl" if "A1" in args.task else "a0",
                 "--onnx-dir", str(args.policy_dir / "exported"),
                 "--delay-ms", "4", "--scene", args.scene or EXPECTED_SCENE])
  # Read the tool's own verdict. A previous version tested `"fall" in out.lower()`, but the
  # replica prints "FELL"/"ok" -- and "fell" does not contain "fall", so that gate was
  # silently ALWAYS-PASS. Never scrape prose for a verdict a tool already publishes.
  rep = marker(out, "REPLICA")
  if rep is None or "pass" not in rep:
    results.append(Result("replica", "S3 replica @4ms", "INFRA",
                          f"no [REPLICA] verdict (rc={rc})"))
    return
  bad = [p.get("phase", "?") for p in rep.get("phases", []) if p.get("fell")]
  results.append(Result("replica", f"S3 replica @4ms ({EXPECTED_SCENE})",
                        "PASS" if rep["pass"] else "FAIL",
                        "no falls" if rep["pass"] else f"fell in: {', '.join(bad)}"))


def control_fingerprint(args) -> str:
  """What the cached A0 control arm is valid for. A rebuilt binary is a DIFFERENT shared
  layer -- articulation, joint_offset, safety filter -- which is exactly what the control
  vouches for, so it must invalidate. No time expiry (the user's call): staleness that a
  fingerprint cannot see is not what this guards against."""
  parts = []
  for p in [args.a0_dir / "exported" / "policy.onnx",
            REPO / "deploy/robots/h1_2/build/h1_2_ctrl",
            args.a0_dir / "params" / "deploy.yaml",
            Path("/opt/unitree_mujoco/unitree_robots/h1_2") / (args.scene
                                                                 or EXPECTED_SCENE)]:
    parts.append(prov.md5(p) if p.exists() else f"<missing:{p.name}>")
  parts.append(args.seq)
  parts.append(args.bridge_cfg)  # gt vs estimator is a different control arm
  return hashlib.md5("|".join(parts).encode()).hexdigest()


def run_bridge_arm(policy, tag, args, deploy_cfg=None, results=None, arm=None) -> Path | None:
  cmd = ["python", "scripts/bridge_session.py", "--policy", policy,
         "--seq", args.seq, "--tag", tag, "--no-analyze"]
  if deploy_cfg:
    cmd += ["--deploy-cfg", deploy_cfg]
  if getattr(args, "auto_restand", False) and arm == "candidate":
    cmd += ["--auto-restand"]
  rc, out = run(cmd)
  # Surface the elastic-band release (2026-08-06). bridge_session prints exactly one
  # `[band] ...` line and this function used to keep only the `[log]` path, so the single
  # fact that decides whether a session means anything was discarded. After the 9-run
  # leg-odometry battery it was impossible to audit which runs had been on the harness --
  # the only reason we knew one was is that a human happened to be watching the window.
  if results is not None:
    band = re.search(r"^\[band\] (.*)$", out, re.M)
    ok = bool(band) and band.group(1).startswith("released")
    results.append(Result("bridge", f"{arm} band release",
                          "INFO" if ok else "INFRA",
                          band.group(1) if band else "no [band] line — release not attempted",
                          blocking=(arm == "candidate" and not ok)))
  m = re.search(r"\[log\]\s*(\S+)\.csv", out)
  return Path(m.group(1) + ".csv") if m else None


def stage_bridge(args, results):
  csvs = {}
  results.append(Result("bridge", "base-state source", "INFO",
                        f"candidate arm runs {args.bridge_cfg} "
                        + ("(deployable IMU/FK estimator — what the robot runs)"
                           if "est" in args.bridge_cfg
                           else "(privileged rt/sportmodestate — SIM ONLY, the real robot "
                                "publishes zeros here)"), blocking=False))
  csvs["candidate"] = run_bridge_arm("hrl" if "A1" in args.task else "a0",
                                     f"{args.tag}_candidate", args, args.bridge_cfg,
                                     results, "candidate")
  fp = control_fingerprint(args)
  cache = json.loads(CONTROL_CACHE.read_text()) if CONTROL_CACHE.exists() else {}
  if not args.refresh_control and cache.get("fingerprint") == fp \
          and Path(cache.get("csv", "")).exists():
    csvs["a0_control"] = Path(cache["csv"])
    results.append(Result("bridge", "A0 control arm", "INFO",
                          f"CACHED from {cache['captured']} (fingerprint {fp[:8]})",
                          blocking=False))
  else:
    csvs["a0_control"] = run_bridge_arm("a0", f"{args.tag}_a0control", args,
                                        args.bridge_cfg, results, "a0_control")
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
      # `blocking` (2026-08-06): every other a0_control gate is report-only, and this one
      # was not -- so a harness artifact on the CONTROL arm sank the whole verdict to
      # INFRASTRUCTURE while the candidate had been scored perfectly well (gt r1 of the
      # leg-odometry A/B). A control arm on the harness costs ATTRIBUTION, not the
      # candidate's result; say so and let the candidate's own gates decide.
      results.append(Result("analyze", f"{arm} band release", "INFRA",
                            "robot never translated under a held command -- harness "
                            "artifact, tells you nothing about the policy"
                            + ("" if blocking else " (control arm: attribution lost, "
                                                   "candidate result stands)"),
                            blocking=blocking))
      continue
    alpha = max((s["alpha_max"] for s in out["segments"]), default=0.0)
    results.append(Result("analyze", f"{arm} safety filter",
                          "PASS" if alpha == 0 else "FAIL",
                          f"alpha_max={alpha}", blocking=blocking))

    # Per-joint torque/joint-velocity trip (2026-08-05). Gated separately from `alpha`
    # because the two triggers have different reach -- tilt/fall hold all 27 joints, this
    # holds only the joints that tripped -- so one number cannot represent both. Fail-closed:
    # a CSV without the columns is UNSCORED, not quiet, and must not read as a pass.
    jt = out.get("joint_trip", {})
    if not jt.get("available"):
      results.append(Result("analyze", f"{arm} joint trip", "INFRA",
                            "no trig_torque/trig_dq columns — this CSV predates the "
                            "per-joint trip (2026-08-05); rebuild h1_2_ctrl and re-capture",
                            blocking=blocking))
    else:
      worst = ", ".join(f"{k} {v:.1%}" for k, v in jt["worst_joint_share"].items())
      # REPORT-ONLY while the trip is detection-only (h1_2_limits.h H1_2_TRIP_HOLD 0). The
      # thresholds were derived FROM the walking they are being scored against, so a small
      # residual firing rate means the bound is slightly under the true distribution, not
      # that the policy is unsafe -- gating on it would block every run on a channel that
      # no longer touches the command. Make this blocking again when the hold is enabled.
      results.append(Result("analyze", f"{arm} joint trip (detect-only)", "INFO",
                            f"alpha_trip_max={jt['alpha_trip_max']} "
                            f"(torque {jt['torque_rate']:.2%}, dq {jt['dq_rate']:.2%})"
                            + (f" | worst: {worst}" if worst else ""), blocking=False))
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
      lo = _bridge_leg_odom(hrl)
      if lo is not None:
        # Reported, never blocking. The open-loop number is a sanity screen on the
        # instrument; the verdict on leg odometry is the closed-loop fall count, and the two
        # are known not to track each other -- the IMU estimator scored 0.0285-0.0299 passive
        # and 0.054-0.065 once it was in the loop. Never quote the passive number as the
        # estimator's accuracy.
        results.append(Result("analyze", f"{arm} leg odometry (HL abs vel)",
                              band_verdict(lo, "legodom"),
                              f"max(vx,vy)={lo} vs {BANDS['legodom']} (open loop, at-fire)",
                              blocking=False))


def _bridge_estimator(hrl_csv: Path):
  import numpy as np
  d = np.genfromtxt(str(hrl_csv), delimiter=",", names=True)
  if not {"est_vx", "gt_vx"} <= set(d.dtype.names or ()):
    return None
  return round(float(max(np.sqrt(((d["est_vx"] - d["gt_vx"]) ** 2).mean()),
                         np.sqrt(((d["est_vy"] - d["gt_vy"]) ** 2).mean()))), 4)


def _bridge_leg_odom(hrl_csv: Path):
  """Leg odometry (absolute, HL-consumed) vs absolute ground truth, scored ON FIRE ROWS.

  Two things here are deliberate and easy to get wrong:

  1. `gt_avx/gt_avy`, not `gt_vx/gt_vy`. The latter pair are within-window INCREMENTS and
     cannot score an absolute estimator -- comparing against them would silently measure the
     wrong quantity and read far too good.
  2. Fire rows only. `lo_*` is the window-latched c-average, held constant across the window,
     and the HL consumes it exactly at the fire. Averaging over all rows would weight each
     window by its length and score a value the HL never acted on at a time it never acted.
     Fire rows are identified the proven way -- `est_vx` is EXACTLY 0 there by construction
     (`base_vel_increment(psi, w, 0, lev0) = 0` at a window start) -- not by index
     arithmetic. That identity holds whichever base-state source is configured, because
     est_vel is computed unconditionally.

  This reproduces the bench's `fasthl_*` definition (window-mean estimate vs truth AT the
  fire), lag included, so the two numbers are directly comparable."""
  import numpy as np
  d = np.genfromtxt(str(hrl_csv), delimiter=",", names=True)
  if not {"lo_vx", "gt_avx", "est_vx"} <= set(d.dtype.names or ()):
    return None
  fire = d["est_vx"] == 0.0
  fire[0] = False  # first window never latched a value
  if fire.sum() < 10:
    return None
  return round(float(max(np.sqrt(((d["lo_vx"][fire] - d["gt_avx"][fire]) ** 2).mean()),
                         np.sqrt(((d["lo_vy"][fire] - d["gt_avy"][fire]) ** 2).mean()))), 4)


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
  ap.add_argument("task", metavar="TASK_ID",
                  help="task id the checkpoint was trained on, e.g. Unitree-H1_2-Flat-A1 "
                       "(A1/HRL) or Unitree-H1_2-Flat (A0). Decides which policy the bridge "
                       "arms drive: 'A1' in the name -> the two-ONNX hierarchy, else A0.")
  ap.add_argument("--checkpoint-file", type=Path, required=True, metavar="MODEL.pt",
                  help="the .pt being cleared for hardware. Everything is scored against "
                       "THIS file: it is staged into --policy-dir/exported (unless "
                       "--no-stage-policy), md5-matched back to its source run, and used "
                       "for every sim bench.")
  ap.add_argument("--policy-dir", type=Path, default=prov.HRL_DIR, metavar="DIR",
                  help="deploy config dir of the CANDIDATE, holding exported/ and params/ "
                       f"(default: {prov.HRL_DIR.relative_to(REPO)}).")
  ap.add_argument("--a0-dir", type=Path, default=prov.A0_DIR, metavar="DIR",
                  help="deploy config dir of the A0 CONTROL arm — the attribution baseline, "
                       "not a second candidate. A0 exercises the shared layer (binary, "
                       "articulation, safety filter, plant) without the HRL path, so "
                       "A0-clean + candidate-fails isolates the fault "
                       f"(default: {prov.A0_DIR.relative_to(REPO)}).")
  # `replica` is NOT in the default chain (2026-08-05). It runs no C++ and no DDS, models
  # neither the safety filter nor `hold_joint_ids` (the actual hardware config), and it
  # passed every phase -- including all four stops -- on the same keeper/plant where the
  # real bridge fell on the 0.5->0 decel. Opt in with --stages ...,replica for a plant or
  # latency A/B, which is the one thing it is still good at.
  ap.add_argument("--stages", default="provenance,sim,bridge,analyze", metavar="A,B,C",
                  help="comma-separated subset of provenance,sim,replica,bridge,analyze "
                       "(default: provenance,sim,bridge,analyze — everything but replica). "
                       "'analyze' scores the CSVs 'bridge' captured, so it is a no-op "
                       "without it. 'replica' is RETIRED as a gate (no C++, no DDS, models "
                       "neither the safety filter nor hold_joint_ids); opt in only for a "
                       "plant or latency A/B. Dropping stages narrows what the verdict "
                       "covers — a GO from --stages provenance,sim says nothing about "
                       "hardware.")
  # Explicit plant selection for A/B work (2026-08-05). Omitted = do not touch /opt, just
  # assert it already holds the shipped scene, which is the shipping default. Named =
  # SET /opt's robot_scene to it and gate on the read-back, so the gate and the plant it
  # scored can never disagree. `scene_stress.xml` is the shipped Unitree plant,
  # `scene.xml` the retired training-proximate one.
  ap.add_argument("--scene", default=None, metavar="XML",
                  help="MuJoCo plant for the bridge. Named = WRITE it into /opt's "
                       "robot_scene and gate on the read-back, so the gate and the plant it "
                       f"scored cannot disagree. Omitted = assert /opt already holds "
                       f"{EXPECTED_SCENE} and touch nothing (the shipping default). "
                       f"{EXPECTED_SCENE} is the shipped Unitree plant; scene.xml is the "
                       "RETIRED training-proximate one — reference numbers do not transfer "
                       "between them.")
  # Which base-state source the bridge feeds the goal space (2026-08-05). Default is the
  # ESTIMATOR, because that is what the real robot runs: rt/sportmodestate publishes
  # identical zeros once our own controller has command, so gating on it validates a
  # configuration that only exists in sim. Pass `deploy.yaml` for the ground-truth arm,
  # which is the attribution tool when the estimator arm fails -- est fails + gt passes
  # isolates the fault to the estimator rather than the policy.
  # Not an allow-list: the A/B arms are configs that exist for one experiment (e.g. the
  # 2026-08-06 leg-odometry control arm), and a hardcoded pair silently makes those
  # unrunnable through the pipeline, pushing them onto hand-driven bridge sessions -- the
  # thing this tool exists to replace. Validated by EXISTENCE in both params dirs instead,
  # which is the failure that actually bites (a missing A0 twin kills the binary at startup).
  ap.add_argument("--bridge-cfg", default="deploy_est.yaml", metavar="YAML",
                  help="H1_2_DEPLOY_CFG the bridge arms run, i.e. WHERE THE GOAL SPACE GETS "
                       "ITS BASE STATE. Default deploy_est.yaml = the deployable IMU/FK "
                       "estimator, what the real robot runs. deploy.yaml = privileged "
                       "rt/sportmodestate, SIM-ONLY (the real robot publishes zeros there) "
                       "and useful as the attribution arm: est fails + gt passes blames the "
                       "estimator, not the policy. Must exist in BOTH --policy-dir/params "
                       "and --a0-dir/params or the controller will not start.")
  ap.add_argument("--tag", default="readiness", metavar="NAME",
                  help="label for this invocation's bridge CSVs, written to "
                       "logs/deploy_safety/<timestamp>_<tag>_candidate and "
                       "..._<tag>_a0control (default: readiness). Timestamped, so runs never "
                       "overwrite — the tag is purely how you find and attribute them later, "
                       "so name it after what you are testing.")
  ap.add_argument("--seq", default=None, metavar="SPEC",
                  help="bridge command sequence (default: bridge_session.DEFAULT_SEQ, the "
                       "full G2.2+G2.3 battery — 20 segments / ~253 s, including the "
                       "0.5->0 decel that produced the 2026-08-03 fall). Shortening it "
                       "shrinks what 'no fall in any phase' actually covers.")
  ap.add_argument("--refresh-control", action="store_true",
                  help="force a fresh A0 control run instead of reusing the cached one. The "
                       "cache already invalidates on any fingerprint change (A0 onnx, "
                       "h1_2_ctrl binary, deploy.yaml, scene, --seq, --bridge-cfg) and "
                       "never expires by time, so use this only when something the "
                       "fingerprint cannot see has moved.")
  ap.add_argument("--auto-restand", action="store_true",
                  help="pass through to bridge_session.py for the CANDIDATE arm only "
                       "(never the A0 control). Default off, matching bridge_session.py's "
                       "own default. On a fall, holds zero command then Backspace-resets "
                       "the sim (mj_resetData -- FixStand cannot recover a robot actually "
                       "lying down) before the next phase, so one fall does not cost "
                       "every later phase's evidence -- does not touch stage_analyze's "
                       "fall gate, which stays blocking.")
  ap.add_argument("--no-stage-policy", dest="stage_policy", action="store_false",
                  help="score whatever is ALREADY in --policy-dir/exported instead of "
                       "staging --checkpoint-file into it. Use to validate an export you "
                       "did not just produce; provenance still md5-identifies it and still "
                       "fails if it is not the checkpoint you named.")
  args = ap.parse_args()
  # Fail before starting anything, not 20 minutes in: BOTH FSM states are constructed at
  # startup and each loads this filename from its OWN params dir, so a config present for the
  # hierarchy but missing its A0 twin takes the whole binary down (found the hard way on
  # 2026-08-05 while testing the T1 refusal path).
  for d in (args.policy_dir, args.a0_dir):
    if not (Path(d) / "params" / args.bridge_cfg).exists():
      print(f"[readiness] --bridge-cfg {args.bridge_cfg} not found in {d}/params — both the "
            f"hierarchy and the A0 dir need it, or the controller will not start.")
      return 2
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
