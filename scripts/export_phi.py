"""WP5 Phase 2: export the TRAINED `phi` to `adapt_encoder.onnx`, the real (non-dummy)
encoder WP5d's deploy path was built to accept. Single writer reused unchanged
(`HierarchicalRunner.export_adapt_encoder_onnx`) -- see doc/hrl/A1a_plan.md "WP5 Phase 2".

  python scripts/export_phi.py --model-dir data/2026-09-wp5-phase2/models \\
      --out deploy/robots/h1_2/config/policy/velocity_hrl/v0/exported/adapt_encoder.onnx
"""

import argparse
import json
from pathlib import Path

import torch
from export_dummy_phi import _CnnPhi
from phi_data import E_NAMES
from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner

POLICY_DIM, COMMAND_DIM, HISTORY_LEN = 89, 3, 50


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  model_dir = Path(args.model_dir)
  norm = json.loads((model_dir / "norm.json").read_text())
  module = _CnnPhi(n_z=5, d_in=POLICY_DIM + COMMAND_DIM, h=HISTORY_LEN)
  module.load_state_dict(torch.load(model_dir / "phi_real.pt", map_location="cpu"))

  HierarchicalRunner.export_adapt_encoder_onnx(
    module, args.out,
    history_len=HISTORY_LEN, input_dim=POLICY_DIM + COMMAND_DIM, z_names=E_NAMES,
    z_clip_lo=norm["clip_lo"], z_clip_hi=norm["clip_hi"], z_cold=norm["z_cold"],
    policy_dim=POLICY_DIM, command_dim=COMMAND_DIM,
    z_scale=norm["scale"], z_center=norm["center"],
  )
  print(f"[export_phi] wrote {args.out}")
  print(f"[export_phi] z_names={list(E_NAMES)} z_center={norm['center']} "
        f"z_scale={norm['scale']} z_clip_lo={norm['clip_lo']} z_clip_hi={norm['clip_hi']} "
        f"z_cold={norm['z_cold']}")


if __name__ == "__main__":
  main()
