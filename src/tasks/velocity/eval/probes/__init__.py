"""The probes, in play.py's dispatch order: with several flags set, the first one listed runs."""

from . import action_rate, bench, cadence, goals, lean, leg_odom, symmetry, vel_increment

# (PlayConfig field that selects it, module). A probe runs when its field is > 0 / truthy.
PROBES = (
  ("eval_steps", bench),
  ("diagnose_cadence", cadence),
  ("diagnose_symmetry", symmetry),
  ("diagnose_action_rate", action_rate),
  ("diagnose_goals", goals),
  ("check_leg_odometry", leg_odom),
  ("check_vel_increment", vel_increment),
  ("probe_lean", lean),
)


def select(cfg):
  """The probe module `cfg` asks for, or None (interactive play)."""
  return next((m for field, m in PROBES if getattr(cfg, field)), None)
