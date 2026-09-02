"""Export a DUMMY WP5d adaptation encoder (``adapt_encoder.onnx``).

WP5d (the hardware deploy path for H-adapt) is deliberately testable before WP5 Phase 2
trains the real ``phi``: this writes a network of the correct SHAPE and with the complete
baked contract, so the third ONNX session, the history buffer, the cold start and the
fail-closed dimension checks can all be exercised, and the real encoder swapped in later
with no C++ change.

It goes through :meth:`HierarchicalRunner.export_adapt_encoder_onnx` -- the same single
writer WP5 Phase 2 will use -- which is what makes a dummy-based test say anything about
the real one. The latent's bounds are read from the LIVE payload-DR env cfg rather than
retyped, so they cannot drift from what training actually samples.

  python scripts/export_dummy_phi.py --out deploy/.../exported/adapt_encoder.onnx
  python scripts/export_dummy_phi.py --out /tmp/bad.onnx --break history   # fail-closed probe

``--break`` writes a deliberately WRONG encoder, to prove the deploy refuses it rather than
reading past the end of a buffer:
  history  H+1 frames declared vs the tensor      layout   the A0 flat column order
  dim      D-1 floats per frame                   order    a reversed time axis
  zdim     a latent one component too wide        cold     a prior outside the clamp
"""

import argparse

import torch

from src.tasks.velocity.config.h1_2_a1.env_cfgs import unitree_h1_2_flat_a1_env_cfg
from src.tasks.velocity.mdp.observations import E_NAMES
from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner

# The deploy contract these must match (asserted at load by State_RLHRL): phi's per-step
# vector is obs["policy"] ++ obs["command"], the first columns of the HL's own input.
POLICY_DIM, COMMAND_DIM = 89, 3
HISTORY_LEN = 50  # thesis plan Phase 2: H = 50 control steps = 1.0 s at 50 Hz


class _ConstantPhi(torch.nn.Module):
  """Emits a fixed latent regardless of history. A dummy must be CONSTANT, not random:
  the point of the flag-off/flag-on A/B is that every difference is attributable, and a
  stochastic dummy would make the bridge comparison unreadable."""

  def __init__(self, z: list[float]):
    super().__init__()
    self.register_buffer("z", torch.tensor([z], dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Touch the input so the export cannot prune it away and silently change the graph's
    # declared input shape -- which is exactly what the deploy validates against.
    return self.z + 0.0 * x.sum()


class _CnnPhi(torch.nn.Module):
  """RMA-shaped 1D-CNN, for TIMING rather than accuracy: a constant dummy measures only ORT
  call overhead (~2 us) and would badly understate what Phase 2 actually costs. Shape follows
  A-RMA's adaptation module -- per-frame MLP down to a channel embedding, three temporal
  convolutions over the H axis, then a linear head. Weights are random; only the graph size
  matters here.

  Input is [1, H, D] to match the deploy's flattening; the transpose to [N, C, T] is inside
  the graph, so C++ never has to know the convolution's axis convention."""

  def __init__(self, n_z: int, d_in: int, h: int, ch: int = 32):
    super().__init__()
    self.embed = torch.nn.Sequential(
      torch.nn.Linear(d_in, 128), torch.nn.ELU(), torch.nn.Linear(128, ch), torch.nn.ELU())
    self.conv = torch.nn.Sequential(
      torch.nn.Conv1d(ch, ch, 8, stride=4), torch.nn.ELU(),
      torch.nn.Conv1d(ch, ch, 5, stride=1), torch.nn.ELU(),
      torch.nn.Conv1d(ch, ch, 5, stride=1), torch.nn.ELU())
    with torch.no_grad():
      n_flat = self.conv(self.embed(torch.zeros(1, h, d_in)).transpose(1, 2)).flatten(1).shape[1]
    self.head = torch.nn.Linear(n_flat, n_z)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.conv(self.embed(x).transpose(1, 2)).flatten(1))


def _latent_bounds() -> tuple[list[float], list[float]]:
  """(lo, hi) per latent component, read from the live payload-DR env cfg."""
  cfg = unitree_h1_2_flat_a1_env_cfg(payload_dr=True)
  mass = cfg.events["base_mass"].params["ranges"]
  com = cfg.events["base_com"].params["ranges"]
  fric = cfg.events["foot_friction"].params["ranges"]
  lo = [mass[0], com[0][0], com[1][0], com[2][0], fric[0]]
  hi = [mass[1], com[0][1], com[1][1], com[2][1], fric[1]]
  return [float(v) for v in lo], [float(v) for v in hi]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--break", dest="brk", default=None,
                  choices=["history", "dim", "layout", "order", "zdim", "cold"])
  ap.add_argument("--arch", default="constant", choices=["constant", "cnn"],
                  help="constant: a fixed z (use for the flag-on/flag-off A/B, where a "
                       "stochastic output would make the comparison unreadable). "
                       "cnn: an RMA-shaped 1D-CNN of realistic SIZE, for timing.")
  args = ap.parse_args()

  lo, hi = _latent_bounds()
  # Cold-start prior (owner's call 2026-09-02): payload 0 kg and CoM 0 -- the physically
  # nominal robot -- with friction at the DR MIDPOINT. Friction has no nominal to fall back
  # on: it reads as an ABSOLUTE coefficient, so its zero would hand the HL a frictionless
  # floor, outside the training support. That asymmetry is why the prior is baked per
  # component rather than being "the zero vector".
  cold = [0.0, 0.0, 0.0, 0.0, 0.5 * (lo[4] + hi[4])]
  names, H, D = E_NAMES, HISTORY_LEN, POLICY_DIM + COMMAND_DIM
  pol, cmd = POLICY_DIM, COMMAND_DIM
  z = list(cold)

  if args.brk == "history":
    H += 1          # metadata claims one more frame than the graph accepts
  elif args.brk == "dim":
    D -= 1; pol -= 1
  elif args.brk == "layout":
    pol, cmd = 6, 86   # the A0 flat permutation: same 92 floats, command mid-vector
  elif args.brk == "zdim":
    names = names + ("extra",); z = z + [0.0]; lo = lo + [0.0]; hi = hi + [1.0]
    cold = cold + [0.0]

  module = (_ConstantPhi(z) if args.arch == "constant"
            else _CnnPhi(len(names), POLICY_DIM + COMMAND_DIM, HISTORY_LEN))
  # The graph is always built at the TRUE shape; only the metadata lies, which is precisely
  # the failure mode the load-time check has to catch.
  true_h = HISTORY_LEN
  true_d = POLICY_DIM + COMMAND_DIM if args.brk != "dim" else D
  HierarchicalRunner.export_adapt_encoder_onnx(
    module, args.out, history_len=true_h, input_dim=true_d, z_names=names,
    z_clip_lo=lo, z_clip_hi=hi, z_cold=cold, policy_dim=pol, command_dim=cmd,
  )
  if args.brk in ("history", "order", "cold"):
    # Rewrite the one key after the fact -- export_adapt_encoder_onnx cannot be asked to
    # write an inconsistent contract, which is itself the point.
    # export_adapt_encoder_onnx REFUSES to write an inconsistent contract -- that is its
    # job, and the `cold` case proves it (it raises before reaching here when asked
    # directly). To exercise the DEPLOY's independent load-time refusal we corrupt the one
    # key afterwards, which is also the realistic shape of the failure: a hand-edited or
    # mismatched export, not one this repo produced.
    import onnx
    m = onnx.load(args.out)
    key, val = {
      "history": ("phi_history_len", str(float(H))),
      "order": ("phi_time_order", "newest_first"),
      "cold": ("z_cold", "0.000,0.000,0.000,0.000,0.000"),  # frictionless floor
    }[args.brk]
    for e in m.metadata_props:
      if e.key == key:
        e.value = val
    onnx.save(m, args.out)

  print(f"[WP5d] wrote {args.out}: [1,{true_h},{true_d}] -> [1,{len(names)}] "
        f"z_names={','.join(names)} z_cold={cold}"
        + (f"  (DELIBERATELY BROKEN: {args.brk})" if args.brk else ""))


if __name__ == "__main__":
  main()
