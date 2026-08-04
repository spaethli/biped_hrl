"""What policy is actually deployed, and is it the one you think?

THE FAILURE THIS EXISTS TO PREVENT (2026-08-03). The deploy directory is a single mutable
slot: `State_RLHRL.cpp:206-207` hardcodes `exported/high_level.onnx` and
`exported/low_level.onnx` with no YAML redirect, so testing a different checkpoint means
copying files in under those fixed names. On 2026-07-29 a WL-D arm-6 experiment
(`rolloverfix0p5`) was copied there and left behind. Four days later three bridge runs and
an estimator measurement were spent on it before anyone checked, and it fell reproducibly.
Nothing in the pipeline could have caught it: the ONNX `run_path` metadata says `local`
for every training-time export, and `onnx_parity.py` (the W4 gate) compared a checkpoint
against its own fresh temp export and never opened the deployed file at all.

TWO ROUTES TO AN ANSWER, in order of cost:

1. IDENTIFY BY md5. Training auto-exports and `play.py --export-onnx` both write the ONNX
   into the run directory, and the deployed copies are byte-identical to them (verified
   2026-08-04: deployed high_level c83da701 / low_level c11d591f both match
   `2026-07-17_21-46-14_a1a_cot0p2_cad0p5_energy0p05_s42`). So hashing every
   `logs/rsl_rl/**/*.onnx` gives a reverse index that NAMES an unknown deployed file. This
   is the capability that was missing: on 2026-08-03 the file "matched no run export I
   checked by md5" -- checked by hand, a few at a time.

2. VERIFY NUMERICALLY. An md5 miss is not proof of a wrong policy: a re-export on a
   different torch build, or an export from a checkpoint whose run directory has since
   been re-exported from a DIFFERENT checkpoint, both break byte-equality while the
   policy is still correct. `onnx_parity.py --onnx-dir` settles it by running random obs
   through the deployed file and the checkpoint's torch actor. Slower (needs the env), and
   authoritative.

Note the ambiguity route 2 also closes: a run directory holds ONE exported set with no
record of WHICH checkpoint produced it, so "stage from the run dir" is an assumption until
something checks it numerically.

Usage:
  python scripts/deploy_provenance.py --check                       # what is deployed now?
  python scripts/deploy_provenance.py --check --policy-dir <dir>
  python scripts/deploy_provenance.py --stage --checkpoint-file <pt>  # put it there, recorded
"""

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnx
import yaml

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs" / "rsl_rl"
HRL_DIR = REPO / "deploy/robots/h1_2/config/policy/velocity_hrl/v0"
A0_DIR = REPO / "deploy/robots/h1_2/config/policy/velocity/v0"
MANIFEST = "PROVENANCE.json"
# The two deploy layouts. A1 runs a two-net hierarchy, A0 a single policy.
HRL_NETS = ("high_level.onnx", "low_level.onnx")
A0_NETS = ("policy.onnx",)


def md5(path: Path) -> str:
  return hashlib.md5(path.read_bytes()).hexdigest()


def build_index(logs_root: Path = LOGS) -> dict[str, list[Path]]:
  """md5 -> every run-dir ONNX with that hash. One hash can map to several paths (the
  same export copied across runs), so the value is a list and the caller reports all."""
  index: dict[str, list[Path]] = {}
  for onnx_path in logs_root.glob("**/*.onnx"):
    index.setdefault(md5(onnx_path), []).append(onnx_path)
  return index


def net_names(policy_dir: Path) -> tuple[str, ...]:
  return HRL_NETS if (policy_dir / "exported" / "high_level.onnx").exists() or \
      "velocity_hrl" in str(policy_dir) else A0_NETS


def read_meta(path: Path) -> dict[str, str]:
  return {p.key: p.value for p in onnx.load(str(path)).metadata_props}


def input_dim(path: Path) -> int:
  g = onnx.load(str(path)).graph
  return g.input[0].type.tensor_type.shape.dim[1].dim_value


def expected_dims(policy_dir: Path) -> dict[str, int]:
  """HL input = 92 proprio+command, +2 when the HL also reads the (vx,vy) estimate.
  LL input = 89 (proprio minus the 3 command columns) + goal_dim. The C++ throws on a
  mismatch, so catching it here turns a runtime abort into a pre-flight message."""
  cfg_path = policy_dir / "params" / "deploy.yaml"
  if not cfg_path.exists():
    return {}
  cfg = yaml.safe_load(cfg_path.read_text())
  hrl = cfg.get("hrl") or {}
  if not hrl:
    return {}
  goal_dim = sum({"velocity": 3, "orientation": 3, "height": 1}[c]
                 for c in hrl.get("goal_components", []))
  return {"high_level.onnx": 92 + (2 if hrl.get("hl_obs_vel") else 0),
          "low_level.onnx": 89 + goal_dim}


def check(policy_dir: Path, index: dict[str, list[Path]] | None = None,
          expect_checkpoint: Path | None = None) -> dict:
  """P1 identify / P1c manifest / P2 dims / P3 goal_scale / P4 structure. Returns a report;
  the caller decides the verdict. Never raises on a bad policy -- only on a missing one.

  `expect_checkpoint` is what gives P1 teeth. Identifying the deployed file is necessary
  but not sufficient: on 2026-08-03 the parked `rolloverfix0p5` pair WAS a valid export
  from a real run, so "matches a known run" would have passed it. The gate is that the
  deployed file matches the run of the checkpoint you asked to validate.
  """
  exported = policy_dir / "exported"
  index = build_index() if index is None else index
  names = net_names(policy_dir)
  dims = expected_dims(policy_dir)
  manifest = json.loads((exported / MANIFEST).read_text()) \
      if (exported / MANIFEST).exists() else None

  report: dict = {"policy_dir": str(policy_dir), "nets": {}, "problems": [],
                  "manifest": manifest}
  runs: set[str] = set()
  for name in names:
    path = exported / name
    if not path.exists():
      report["problems"].append(f"{name}: MISSING from {exported}")
      continue
    h = md5(path)
    meta = read_meta(path)
    hits = index.get(h, [])
    # The run directory is the grandparent of the export (<run>/<file>.onnx), or the
    # parent when a run nests its exports one level down.
    srcs = sorted({p.parent.name for p in hits})
    runs.update(srcs)
    entry = {
      "md5": h,
      "source_runs": srcs,                       # empty = matches no known run export
      "run_path_metadata": meta.get("run_path"),  # always 'local'; kept to show it is useless
      "input_dim": input_dim(path),
      "has_goal_scale": "goal_scale" in meta,
    }
    if not srcs:
      report["problems"].append(
        f"{name}: md5 {h[:12]} matches NO export under {LOGS} -- this file was not "
        "produced by any run on this machine, or its run dir has been re-exported since. "
        "Verify numerically: onnx_parity.py --onnx-dir")
    if name in dims and entry["input_dim"] != dims[name]:
      report["problems"].append(
        f"{name}: input dim {entry['input_dim']} != {dims[name]} expected from deploy.yaml "
        "(the C++ throws on this at FSM entry)")
    if not entry["has_goal_scale"] and name in HRL_NETS:
      report["problems"].append(
        f"{name}: no goal_scale metadata -- the C++ silently falls back to deriving the "
        "goal scale from deploy.yaml command ranges, which are an operator clamp and "
        "deliberately do NOT match training")
    if manifest and manifest.get("md5", {}).get(name) not in (None, h):
      report["problems"].append(
        f"{name}: md5 {h[:12]} does not match {MANIFEST} ({manifest['md5'][name][:12]}) -- "
        "the directory changed since it was staged")
    report["nets"][name] = entry

  if len(runs) > 1:
    report["problems"].append(
      f"nets come from DIFFERENT runs {sorted(runs)} -- a mismatched HL/LL pair is a "
      "silently wrong hierarchy, not a load error")
  report["source_run"] = next(iter(runs)) if len(runs) == 1 else None

  if expect_checkpoint is not None:
    want = expect_checkpoint.resolve().parent.name
    report["expected_run"] = want
    if report["source_run"] is None:
      report["problems"].append(
        f"cannot confirm the deployed policy is {want}: the files match no single known "
        "run. Settle it numerically with onnx_parity.py --onnx-dir before any bridge run")
    elif report["source_run"] != want:
      report["problems"].append(
        f"WRONG POLICY DEPLOYED: exported/ holds {report['source_run']} but you asked to "
        f"validate {want}. This is the 2026-08-03 failure verbatim -- stage the right one "
        "(--stage) rather than running the bridge against a stranger")
    if manifest and manifest.get("source_run") not in (None, report["source_run"]):
      report["problems"].append(
        f"{MANIFEST} claims {manifest['source_run']} but the files are "
        f"{report['source_run']} -- the directory was modified after staging")

  report["ok"] = not report["problems"]
  return report


def stage(checkpoint: Path, policy_dir: Path) -> dict:
  """Copy the checkpoint's run-dir exports into exported/ and record what was done.

  Staging is owned by this tool on purpose: the deploy dir cannot hold a stranger if the
  thing that fills it is the thing that runs the bridge, and the record travels with the
  files. `PROVENANCE.json` is what the 2026-08-03 directory lacked -- there was no way to
  ask "what is this?" short of hashing candidates by hand.
  """
  run_dir = checkpoint.parent
  exported = policy_dir / "exported"
  names = net_names(policy_dir)
  srcs = [run_dir / n for n in names]
  missing = [s.name for s in srcs if not s.exists()]
  if missing:
    raise SystemExit(
      f"{run_dir} has no {', '.join(missing)}. Export first:\n"
      f"  python scripts/play.py <TaskID> --checkpoint-file {checkpoint} --export-onnx")

  exported.mkdir(parents=True, exist_ok=True)
  for s in srcs:
    shutil.copy2(s, exported / s.name)
  manifest = {
    "staged_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "checkpoint": str(checkpoint.relative_to(REPO) if checkpoint.is_relative_to(REPO)
                      else checkpoint),
    "source_run": run_dir.name,
    "md5": {s.name: md5(exported / s.name) for s in srcs},
    "staged_by": "scripts/deploy_provenance.py",
    # A run dir holds ONE export set with no record of which checkpoint produced it, so
    # this field is an ASSERTION, not an observation, until onnx_parity --onnx-dir agrees.
    "checkpoint_verified_numerically": False,
  }
  (exported / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--check", action="store_true", help="report what is deployed")
  ap.add_argument("--stage", action="store_true", help="stage a checkpoint's exports")
  ap.add_argument("--checkpoint-file", type=Path,
                  help="required with --stage; with --check alone it becomes the EXPECTED "
                       "policy and a mismatch is a failure")
  ap.add_argument("--policy-dir", type=Path, default=HRL_DIR)
  args = ap.parse_args()
  if not (args.check or args.stage):
    ap.error("pass --check or --stage")

  if args.stage:
    if not args.checkpoint_file:
      ap.error("--stage needs --checkpoint-file")
    m = stage(args.checkpoint_file.resolve(), args.policy_dir)
    print(f"[PROV] staged from {m['source_run']}")
    for k, v in m["md5"].items():
      print(f"[PROV]   {k:20s} {v}")
    print(f"[PROV] wrote {args.policy_dir / 'exported' / MANIFEST}")

  rep = check(args.policy_dir, expect_checkpoint=args.checkpoint_file)
  for name, e in rep["nets"].items():
    srcs = ", ".join(e["source_runs"]) or "<NO MATCH>"
    print(f"[PROV] {name:20s} md5={e['md5'][:12]} dim={e['input_dim']:<4} "
          f"goal_scale={'yes' if e['has_goal_scale'] else 'NO':4s} -> {srcs}")
  if rep["manifest"]:
    print(f"[PROV] manifest: staged {rep['manifest']['staged_at']} from "
          f"{rep['manifest']['source_run']}")
  for p in rep["problems"]:
    print(f"[PROV] PROBLEM: {p}")
  print(f"[PROVENANCE] {json.dumps(rep, default=str)}")
  return 0 if rep["ok"] else 1


if __name__ == "__main__":
  sys.exit(main())
