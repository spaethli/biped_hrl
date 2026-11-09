"""Golden-baseline gate for scripts/play.py: run a fixed matrix of probe invocations, then judge a
candidate run against repeated baseline runs of the SAME matrix.

  python scripts/play_golden.py capture --out <dir> --rep a [--only name,...]
  python scripts/play_golden.py compare --out <dir> --baseline a,b,c --candidate r [--only ...]

The contract under test is what downstream parsers read: the exit code (and exception type), every
``[TAG]`` line (JSON flattened per field, text lines as a number-masked skeleton + the numbers), the
exported ONNX bytes, and the schedule-replay trace as bench_flight_recorder --sim scores it.

Field band (repeated runs differ by GPU non-determinism, measured 2026-09-18 at a 2 % median but
25-85 % on some ``_std`` fields, so no single tolerance fits):
  |R - median(B)| <= max(2 * range(B), q_tag * |median(B)|, 1e-6)
where q_tag is the 75th-percentile relative range over that tag's numeric fields. Strings, bools,
key sets and exit codes match exactly. ``per_env`` arrays (one value per env: per-env randomization
draws, up to 199 % apart between identical runs) and the trace's per-env score files are compared as
distributions (n, mean, p10, p50, p90), never element by element.

Row verdict (see `compare`): no hard mismatch, and a failing-field fraction within 1.5x the
baselines' own worst leave-one-out fraction. Validated 2026-11-09: a seed +1 and an off-by-one HL
window each fail their rows at 1.6-3.5x the null (data/2026-11-09-play-golden/validation.md).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2, A1 = "logs/rsl_rl/h1_2_velocity_v2", "logs/rsl_rl/h1_2_velocity_a1_v2"
A1A = f"{A1}/2026-09-02_12-57-09_a1a_fullmirror_cot5_standing12_s42/model_10000.pt"
A0 = f"{V2}/2026-09-14_16-33-58_a0_standing12_s42/model_10000.pt"
HADAPT = f"{A1}/2026-09-08_07-48-18_a1a_hadapt_e_cot5_keeperLL_standing12_s42/model_10000.pt"
LEGACY = f"{A1}/2026-07-14_16-03-57_a1_td3_pose0p5shw16-4_ar0p02_cadhl_cot0p2_kl0p01_rs20_s42/model_10000.pt"
T_A1, T_A0 = "Unitree-H1_2-Flat-A1", "Unitree-H1_2-Flat"
PHI = ["--hl-obs-e-source", "phi", "--phi-dir", "data/2026-09-08-wp5-phase2-keeper/models"]

# Every probe on both architectures (a flat checkpoint that errors must keep erroring the same way),
# plus the modes the 2026-09-18 capture missed. kind: tags | export | trace.
PROBES = {
  "bench": ["--eval-steps", "600"],
  "bench_limits": ["--eval-steps", "600", "--check-joint-limits", "True"],
  "bench_vx05": ["--eval-steps", "600", "--eval-cmd-vx", "0.5"],
  "bench_vx10": ["--eval-steps", "600", "--eval-cmd-vx", "1.0"],
  "bench_reward": ["--eval-steps", "600", "--reward-terms", "True"],
  "goals": ["--diagnose-goals", "600"],
  "goals_vx05": ["--diagnose-goals", "600", "--eval-cmd-vx", "0.5"],
  "goals_vx10": ["--diagnose-goals", "600", "--eval-cmd-vx", "1.0"],
  "velinc": ["--check-vel-increment", "600"],
  "legodom": ["--check-leg-odometry", "600"],
  "symmetry": ["--diagnose-symmetry", "600"],
  "cadence_vx05": ["--diagnose-cadence", "600", "--eval-cmd-vx", "0.5"],
  "actrate": ["--diagnose-action-rate", "600"],
  "lean": ["--eval-cmd-vx", "0", "--probe-lean", "400"],
}
MATRIX = {}
for arch, task, ckpt in (("a1a", T_A1, A1A), ("a0", T_A0, A0)):
  for p, args in PROBES.items():
    MATRIX[f"{arch}_{p}"] = (task, ckpt, args, "tags")
  MATRIX[f"{arch}_trace"] = (task, ckpt, ["--eval-seeds", "1", "--eval-cmd-seq", "default",
                                          "--reward-terms", "True"], "trace")
  MATRIX[f"{arch}_export"] = (task, ckpt, ["--export-onnx"], "export")
MATRIX["a1a_lean_grav"] = (T_A1, A1A, ["--eval-cmd-vx", "0", "--probe-lean", "400",
                                       "--probe-gravity-noise", "0.05,0.2"], "tags")
MATRIX["a1a_payload"] = (T_A1 + "-Payload", A1A, ["--eval-steps", "600", "--eval-payload-kg", "7.5"],
                         "tags")
MATRIX["hadapt_phi"] = (T_A1, HADAPT, ["--eval-steps", "600", *PHI], "tags")
# A pre-2026-07-16 checkpoint: no baked goal_scale and an absence-shimmed agent.yaml key, so the
# [SHIM] fix-up path runs (every other row carries a baked scale).
MATRIX["legacy_bench"] = (T_A1, LEGACY, ["--eval-steps", "600"], "tags")

SKIP_TAGS = {"INFO", "WARN"}   # human-facing, carry paths and timings; no parser reads them
TAG_RE = re.compile(r"^\s*\[([A-Z_]+)\]\s?(.*)$")
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
ERR_RE = re.compile(r"^(\w+(?:Error|Exception))\b")
# File paths only (scratch dirs differ per rep); 'ach_vx/win' is a label, not a path.
PATH_RE = re.compile(r"^['\"(]?(?:/|~|\./|logs/|data/)|\.(?:pt|onnx|npz|json|yaml|csv)\W*$")


def capture(out: Path, rep: str, only):
  d = out / rep
  d.mkdir(parents=True, exist_ok=True)
  env = {**os.environ, "WANDB_MODE": "disabled"}
  for name, (task, ckpt, args, kind) in MATRIX.items():
    if only and name not in only:
      continue
    cmd_ckpt = ckpt
    if kind == "export":
      # play.py exports NEXT TO the checkpoint; the real run dirs hold the exports deploy_provenance
      # md5-matches against, so export from a scratch copy (params/ + a symlink to the weights).
      scratch = d / f"{name}_run"
      shutil.rmtree(scratch, ignore_errors=True)
      shutil.copytree(ROOT / Path(ckpt).parent / "params", scratch / "params")
      (scratch / Path(ckpt).name).symlink_to(ROOT / ckpt)
      cmd_ckpt = str(scratch / Path(ckpt).name)
    extra = ["--eval-trace", str(d / f"{name}_trace")] if kind == "trace" else []
    cmd = [sys.executable, "scripts/play.py", task, "--checkpoint-file", cmd_ckpt, "--num-envs", "64",
           *(["--eval-seeds", "2"] if "--eval-seeds" not in args else []), *args, *extra]
    t0 = time.time()
    with open(d / f"{name}.log", "w") as f:
      rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    status = {"rc": rc, "seconds": round(time.time() - t0, 1), "cmd": cmd[1:]}
    if kind == "export":
      status["onnx"] = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
                        for p in sorted(scratch.glob("*.onnx"))}
    if kind == "trace":
      npz = d / f"{name}_trace_s42.npz"
      sc = subprocess.run([sys.executable, "scripts/bench_flight_recorder.py", "--sim", "--trace",
                           str(npz), "--out-dir", str(d / f"{name}_score")], cwd=ROOT, env=env,
                          capture_output=True, text=True)
      status["score_rc"] = sc.returncode
      npz.unlink(missing_ok=True)   # ~100s of MB; the per-env score files are what figures consume
    (d / f"{name}.status.json").write_text(json.dumps(status, indent=1))
    print(f"[golden] {rep} {name}: rc={rc} {status['seconds']}s", flush=True)


def _dist(vals):
  v = sorted(x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool))
  if not v:
    return {"n": 0}
  q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
  return {"n": len(v), "mean": sum(v) / len(v), "p10": q(.1), "p50": q(.5), "p90": q(.9)}


def _flatten(obj, path, out):
  if isinstance(obj, dict):
    for k, v in obj.items():
      _flatten(v, f"{path}.{k}", out)
  elif isinstance(obj, list):
    if "per_env" in path and all(isinstance(x, (int, float)) for x in obj):
      for k, v in _dist(obj).items():
        out[f"{path}~{k}"] = v
    else:
      for i, v in enumerate(obj):
        _flatten(v, f"{path}[{i}]", out)
  else:
    out[path] = obj


def extract(d: Path, name: str) -> dict:
  """Every contract field of one invocation -> {key: value}. A key's tag is its prefix before '#'."""
  st = json.loads((d / f"{name}.status.json").read_text())
  out = {"exit#.rc": st["rc"]}
  lines = (d / f"{name}.log").read_text(errors="replace").splitlines()
  if st["rc"]:
    errs = [m.group(1) for ln in lines if (m := ERR_RE.match(ln))]
    out["exit#.error"] = errs[-1] if errs else None
  seen = {}
  for ln in lines:
    m = TAG_RE.match(ln)
    if not m or m.group(1) in SKIP_TAGS:
      continue
    tag, rest = m.group(1), m.group(2).strip()
    i = seen[tag] = seen.get(tag, -1) + 1
    key = f"{tag}#{i}"
    try:
      obj = json.loads(rest) if rest.startswith("{") else None
    except json.JSONDecodeError:
      obj = None
    if obj is not None:
      _flatten(obj, key, out)
      continue
    text = " ".join("<path>" if PATH_RE.match(tok) else tok for tok in rest.split())
    out[f"{key}.text"] = NUM_RE.sub("#", text)
    for j, n in enumerate(NUM_RE.findall(text)):
      out[f"{key}.n{j}"] = float(n)
  for fn, md5 in st.get("onnx", {}).items():
    out[f"onnx#.{fn}"] = md5
  if "score_rc" in st:
    out["score#.rc"] = st["score_rc"]
    by_regime = {}
    for p in sorted((d / f"{name}_score").glob("*.json")):
      flat = {}
      _flatten(json.loads(p.read_text()), "", flat)
      by_regime.setdefault(p.stem.rsplit(".", 1)[-1], []).append(flat)
    for regime, rows in by_regime.items():
      out[f"score_{regime}#.n_files"] = len(rows)
      for k in sorted({k for r in rows for k in r}):
        vals = [r.get(k) for r in rows]
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
          for s, v in _dist(vals).items():
            out[f"score_{regime}#{k}~{s}"] = v
  return out


def _median(vals):
  s = sorted(vals)
  m = len(s) // 2
  return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def _num(v):
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def judge(base: list, cand: dict):
  """-> (n_checked, failures, notes). base: one extract() per baseline rep."""
  keys_all = set.intersection(*(set(b) for b in base))
  keys_any = set.union(*(set(b) for b in base))
  fails, notes = [], []
  for k in sorted(keys_any - keys_all):
    notes.append(f"unstable key (in some baselines only): {k}")
  for k in sorted(keys_all - set(cand)):
    fails.append((k, "missing in candidate", None, None))
  for k in sorted(set(cand) - keys_any):
    fails.append((k, "extra in candidate", cand[k], None))
  # q_tag: the tag's typical relative run-to-run spread, the floor for fields whose baselines
  # happened to land close together.
  rel = {}
  for k in keys_all:
    vals = [b[k] for b in base]
    if all(_num(v) and math.isfinite(v) for v in vals):
      med = _median(vals)
      rel.setdefault(k.split("#")[0], []).append((max(vals) - min(vals)) / max(abs(med), 1e-9))
  q_tag = {t: sorted(r)[int(0.75 * (len(r) - 1))] for t, r in rel.items()}
  n = 0
  for k in sorted(keys_all & set(cand)):
    vals, r = [b[k] for b in base], cand[k]
    n += 1
    if all(_num(v) for v in vals) and _num(r):
      nan_b = [math.isnan(v) for v in vals]
      if any(nan_b) or math.isnan(r):
        if not (all(nan_b) and math.isnan(r)):
          if all(nan_b) or not any(nan_b):
            fails.append((k, "NaN mismatch", r, vals))
          else:
            notes.append(f"NaN in some baselines only: {k}")
        continue
      med = _median(vals)
      tol = max(2 * (max(vals) - min(vals)), q_tag.get(k.split("#")[0], 0.0) * abs(med), 1e-6)
      if abs(r - med) > tol:
        fails.append((k, f"|R-med|={abs(r - med):.4g} > tol={tol:.4g}", r, vals))
    elif len({json.dumps(v, sort_keys=True) for v in vals}) > 1:
      notes.append(f"non-numeric field differs across baselines: {k} {vals}")
    elif r != vals[0]:
      fails.append((k, "exact mismatch", r, vals[0]))
  return n, fails, notes


PROFILE_MARGIN, PROFILE_FLOOR = 1.5, 0.02


def compare(out: Path, base_reps, cand_rep, only, report: Path | None):
  """Row verdict. The per-field band alone cannot be the gate: the noise is heavy-tailed (a 2-seed
  `_std`, a rare limit touch, a per-env quantile), so an identical-code rerun fails ~8 % of its
  ~10k fields (measured 2026-11-09). A moved line shifts MANY fields of the rows it touches;
  noise scatters a few. So a row passes when it has no hard mismatch (exit code, exception,
  key set, string) and its failing-field fraction, judged against every baseline pair, is at most
  PROFILE_MARGIN x the worst leave-one-out fraction among the baselines themselves (+ a floor)."""
  if len(base_reps) < 3:
    sys.exit("compare needs >= 3 baseline reps (the null is each one judged against a pair)")
  pairs = [[r for r in base_reps if r != x] for x in base_reps]
  lines, n_bad = [], 0
  for name in MATRIX:
    if only and name not in only:
      continue
    if not all((out / r / f"{name}.status.json").exists() for r in [*base_reps, cand_rep]):
      lines.append(f"## {name}: SKIPPED (not captured in every rep)")
      continue
    ext = {r: extract(out / r, name) for r in [*base_reps, cand_rep]}
    null = []
    for x, pair in zip(base_reps, pairs):
      n, f, _ = judge([ext[r] for r in pair], ext[x])
      null.append(len(f) / max(n, 1))
    cand, hard, fails, notes = [], [], [], []
    for pair in pairs:
      n, f, nt = judge([ext[r] for r in pair], ext[cand_rep])
      cand.append(len(f) / max(n, 1))
      hard += [x for x in f if "> tol" not in x[1]]
    n, fails, notes = judge([ext[r] for r in base_reps], ext[cand_rep])
    frac, limit = sum(cand) / len(cand), PROFILE_MARGIN * max(null) + PROFILE_FLOOR
    ok = not hard and frac <= limit
    n_bad += not ok
    lines.append(f"## {name}: {'PASS' if ok else 'FAIL'}  failing fraction {frac:.3f} vs limit "
                 f"{limit:.3f} (baseline leave-one-out {', '.join(f'{v:.3f}' for v in null)}); "
                 f"{len({k for k, *_ in hard})} hard mismatch(es)")
    seen = set()
    for k, why, r, b in hard:
      if k not in seen:
        seen.add(k)
        lines.append(f"- HARD {k}: {why}  cand={r!r} base={b!r}")
    lines += [f"- field outside the 3-baseline band: {k}: {why}  cand={r!r} base={b!r}"
              for k, why, r, b in fails[:15] if k not in seen]
    lines += [f"- note: {s}" for s in notes[:5]]
  head = (f"# play_golden compare: candidate `{cand_rep}` vs baselines {base_reps}\n\n"
          f"**{'PASS' if not n_bad else 'FAIL'}**: {n_bad} row(s) failing\n")
  text = head + "\n".join(lines) + "\n"
  print(text)
  if report:
    report.write_text(text)
  return 1 if n_bad else 0


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("mode", choices=["capture", "compare", "list"])
  ap.add_argument("--out", type=Path, required=False)
  ap.add_argument("--rep", help="capture: the rep label (a subdirectory of --out)")
  ap.add_argument("--baseline", help="compare: comma-separated baseline reps")
  ap.add_argument("--candidate", help="compare: the rep to judge")
  ap.add_argument("--only", default="", help="comma-separated matrix entries (default: all)")
  ap.add_argument("--report", type=Path, help="compare: also write the report here")
  a = ap.parse_args()
  only = set(filter(None, a.only.split(",")))
  if unknown := only - set(MATRIX):
    sys.exit(f"unknown matrix entries: {sorted(unknown)}")
  if a.mode == "list":
    for k, (task, ckpt, args, kind) in MATRIX.items():
      print(f"{k:<20} {kind:<6} {task:<26} {' '.join(args)}")
  elif a.mode == "capture":
    capture(a.out, a.rep, only)
  else:
    sys.exit(compare(a.out, a.baseline.split(","), a.candidate, only, a.report))


if __name__ == "__main__":
  main()
