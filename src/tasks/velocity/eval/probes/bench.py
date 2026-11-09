"""--eval-steps: the headless deterministic benchmark. `eval_steps` steps x `eval_seeds` seeds,
one `[BENCH_SEED]` line per seed and the seed-aggregated `[BENCH]` line."""

import json

import torch

from src.tasks.velocity.mdp.observations import env_latent_e

from ..addons import build_addons
from ..context import foot_pos_b
from ..rollout import Observer, mean_std, rollout

# The per-seed keys of [BENCH_SEED] and, with a `_std` twin each, of [BENCH]. Pinned because
# downstream scorers (bench_flight_recorder's schema partition, plot_radar) read them by name;
# `seed_end` fails closed if the computed dict drifts from this list.
SEED_KEYS = (
  "err_vx", "err_vy", "err_yaw", "ss_err_vx", "ss_err_vy", "t90_s", "fall_rate", "mean_ep_len",
  "action_rate", "act_legs", "ank_roll_span_l", "ank_roll_span_r", "ank_pitch_span_l",
  "ank_pitch_span_r", "jacc", "jacc_legs", "jacc_arms", "jacc_legs_p95", "jacc_arms_p95",
  "ajit_legs", "ajit_legs_p95", "qjit_legs", "qjit_legs_p95", "act_legs_rad", "orient_dev",
  "height_dev", "ub_pose_dev", "ub_arm_vel", "mech_power_w", "gait_match", "cot",
  "mech_power_copper_w", "cot_copper", "omega_xy", "ss_vx_var", "stance_w_walk", "stance_w_stand",
  "stance_w_td", "stride_period_s", "payload_kg",
)

MASS_G = 75.0 * 9.81  # H1-2 ~75 kg; dimensionless CoT = energy / (m g distance)
# WP1 payload pilot: copper-loss-corrected CoT variant, metric-only (never enters any
# reward -- the trained/mechanical `power`/`cot` below are unchanged). k = 0.3 from
# Yang et al. 2022 (CoRL), "Fast and Efficient Locomotion via Learned Gait Transitions"
# (arXiv:2104.04644) eq. 3, Sum_i max(tau_i*omega_i + k*tau_i^2, 0), their "motor
# parameter" following the MIT-Cheetah/Di Carlo actuator convention. NOT fit to the
# H1-2's M107/GO2HV actuators specifically (no public winding-resistance/torque-constant
# spec for them); a documented literature default.
COPPER_LOSS_K = 0.3


class Bench(Observer):

  def __init__(self, ctx, limits=None):
    self.ctx, self.limits = ctx, limits
    env, runner = ctx.env, ctx.runner
    uenv = self.uenv = env.unwrapped
    robot = self.robot = uenv.scene["robot"].data
    self.n_envs = uenv.num_envs
    self.tm = tm = uenv.termination_manager
    self.has_fell = "fell_over" in tm.active_terms
    # Upper-body deploy-hygiene metrics (ADR-0002): arms+waist drift-from-default + speed.
    ub_ids, _ = uenv.scene["robot"].find_joints(
      [".*shoulder.*", ".*elbow.*", ".*wrist.*", ".*torso.*"])
    self.ub_ids = torch.as_tensor(ub_ids, device=robot.joint_pos.device)
    # Leg/arm-restricted smoothness+accel diagnostics (the deploy blocker is smoothness,
    # and action_rate is arm-dominated -- see ARDIAG's g_legs/g_arms below). Same per-
    # pattern try/except as that block's _find_group (find_joints raises if ANY pattern
    # in the list matches zero joints).
    def _grp(pats):
      ids = set()
      for p in pats:
        try:
          pids, _ = uenv.scene["robot"].find_joints([p])
        except ValueError:
          continue
        ids.update(pids)
      return torch.as_tensor(sorted(ids), device=robot.joint_pos.device) if ids else None
    self.leg_ids = _grp([".*hip.*", ".*knee.*", ".*ankle.*"])
    self.arm_ids = _grp([".*shoulder.*", ".*elbow.*", ".*wrist.*"])
    # Per-joint action scale kappa (rad per action unit), read from the live action term
    # rather than re-imported from the robot constants, so it can never desync from the
    # plant the checkpoint is actually being replayed on. Needed because every commanded-
    # side smoothness number is reported in RADIANS: kappa = 0.25*tau_max/Kp spans 6.7x
    # across the body (legs 0.25, shoulder_yaw 0.0375), so a raw-action-unit whole-body
    # norm physically over-weights the arms 3-6.7x. See CONTEXT.md "Physical units rule".
    _act_scale = uenv.action_manager.get_term("joint_pos").scale
    if not torch.is_tensor(_act_scale):
      _act_scale = torch.full((1, uenv.action_manager.total_action_dim),
                              float(_act_scale), device=robot.joint_pos.device)
    self.kappa = _act_scale[:1].to(robot.joint_pos.device)  # [1, A], env-invariant
    self.ctrl_dt = uenv.step_dt
    # CoT / cadence metrics (A1a M0): mechanical power, cost of transport, achieved stride period.
    try:
      self.contact_sensor = uenv.scene["feet_ground_contact"]
    except KeyError:
      self.contact_sensor = None
    # Stance-width metric (2026-09-07): the foot sites, same pair the leg-odometry bench uses.
    try:
      bench_foot_ids, _bfn = uenv.scene["robot"].find_sites(
        ["left_foot", "right_foot"], preserve_order=True)
      assert _bfn == ["left_foot", "right_foot"], f"foot site order: {_bfn}"
    except Exception:
      bench_foot_ids = None
    self.bench_foot_ids = bench_foot_ids
    self.step_dt = uenv.step_dt

    # Collect label / structure info for the JSON line.
    self.bench_meta = {"label": ctx.label}
    for k in ("hl_algorithm", "goal_components", "c"):
      if hasattr(ctx.agent_cfg, k):
        v = getattr(ctx.agent_cfg, k)
        self.bench_meta[k] = list(v) if isinstance(v, (list, tuple)) else v
    self.seed_results: list[dict] = []
    self.gait_offsets = torch.tensor([0.0, 0.5], device=env.device).view(1, -1)
    # d(T) duty for gait_match: MUST mirror feet_gait's schedule, else a d(T) checkpoint
    # reads as "drifting" against the fixed floor (the structure_keys lesson, eval side).
    self.gait_swing = 0.0 if ctx.dummy else getattr(runner, "cadence_swing_time", 0.0)
    self.duty_lo, self.duty_hi = (0.56, 0.70) if ctx.dummy else getattr(
      runner, "cadence_duty_range", (0.56, 0.70))

  def reset(self):
    uenv = self.uenv
    # WP2: log the ACTUAL sampled payload (mean over envs), not the requested
    # --eval-payload-kg -- so a payload-DR training run's rollout can be segmented by
    # payload post-hoc without re-running. `mode="startup"` DR does not resample on
    # reset, so this is constant for the whole play.py process; read once per seed.
    e_params = uenv.observation_manager.get_term_cfg("critic", "env_latent_e").params
    self.payload_kg = env_latent_e(uenv, e_params["torso_cfg"], e_params["foot_cfg"])[:, 0].mean().item()

    self.errs_vx, self.errs_vy, self.errs_yaw = [], [], []
    self.fall_flags, self.ep_lens, self.action_rates, self.orient_devs, self.height_devs = [], [], [], [], []
    self.ub_pose_devs, self.ub_arm_vels, self.powers, self.gait_matches = [], [], [], []
    # WP1 payload pilot: copper-loss power (metric-only), body roll/pitch-rate magnitude
    # (wobble proxy), and per-step cross-env vx variance (F5 analogue -- the N=64 parallel
    # envs under a pinned --eval-cmd-vx ARE the "N repeats of an identical command").
    self.powers_copper, self.omega_xys, self.achieved_vx_vars = [], [], []
    # Leg-restricted action rate + whole-body/leg/arm joint-accel (deploy diagnostics).
    self.act_legs_list, self.jacc_list, self.jacc_legs_list, self.jacc_arms_list = [], [], [], []
    # LCP smoothness suite (Chen 2025), legs-only, in the paper's units. `ajit` is the
    # THIRD derivative of the commanded joint target (rad/s^3) and `qjit` the third
    # derivative of the realized joint position -- the two metrics the paper's ablation
    # ranks on, because they separate smoothing methods ~13x where first/second
    # derivatives (our action_rate / jacc) separate them only 1.2-1.7x.
    self.ajit_list, self.qjit_list, self.act_legs_rad_list = [], [], []
    self.act_hist: list[torch.Tensor] = []   # last 3 raw action tensors, newest last
    self.prev_jacc_signed: torch.Tensor | None = None
    self.achieved_vxs = []  # per-step env-mean achieved vx (ramp metric for --eval-cmd-vx holds)
    self.prev_actions: torch.Tensor | None = None
    # Stance width (2026-09-07): lateral foot separation in the pelvis frame, the control
    # variable for lateral balance. REGIME-SPLIT on the same commanded-motion gate as the
    # CoT numerator -- the narrowing is reported in WALKING, and `pose` pulls hip_roll to 0
    # three times harder when standing (std 0.05 vs 0.15), so a pooled mean cannot say
    # which regime moved. `_td` samples only at touchdown, which is the biomechanical
    # definition of step width (foot PLACEMENT); the other two average over swing too.
    self.w_walk_sum = self.w_stand_sum = self.w_td_sum = 0.0
    self.w_walk_n = self.w_stand_n = self.w_td_n = 0
    self.energy_eng = self.dist_eng = self.td_count = 0.0  # CoT numerator/denominator + footfall count
    self.energy_eng_copper = 0.0  # copper-loss CoT numerator (metric-only)
    self.prev_contact: torch.Tensor | None = None
    self.n_feet = 0

  def after_step(self, step_i, actions, dones):
    uenv, robot, leg_ids, kappa, ctrl_dt, step_dt = (
      self.uenv, self.robot, self.leg_ids, self.kappa, self.ctrl_dt, self.step_dt)
    # 1. Tracking error.
    cmd = uenv.command_manager.get_command("twist")  # [B, >=3]
    achieved = torch.cat(
      [robot.root_link_lin_vel_b[:, :2], robot.root_link_ang_vel_b[:, 2:3]], dim=-1
    )
    ae = (cmd[:, :3] - achieved).abs()  # [B, 3]
    self.errs_vx.append(ae[:, 0].mean().item())
    self.errs_vy.append(ae[:, 1].mean().item())
    self.errs_yaw.append(ae[:, 2].mean().item())
    self.achieved_vxs.append(achieved[:, 0].mean().item())
    # WP1 stability proxy (F5 analogue): cross-env variance of achieved vx at this
    # step -- meaningful under a pinned --eval-cmd-vx, where the N envs are N
    # repeats of the identical command (same convention as ss_err_vx/t90 below).
    self.achieved_vx_vars.append(achieved[:, 0].var(dim=0).item())

    # 2. Survival: fall flag and episode length.
    if self.has_fell:
      self.fall_flags.append(self.tm.get_term("fell_over").float().mean().item())
    else:
      self.fall_flags.append(dones.float().mean().item())
    self.ep_lens.append(uenv.episode_length_buf.float().mean().item())

    # 3. Smoothness: action rate ||a_t - a_{t-1}||, whole-body + leg-restricted
    # (same slicing as ARDIAG's g_legs -- see the cross-check note on act_legs below).
    if self.prev_actions is not None:
      d_ar = actions - self.prev_actions
      self.action_rates.append(d_ar.norm(dim=-1).mean().item())
      self.act_legs_list.append(
        d_ar[:, leg_ids].norm(dim=-1).mean().item() if leg_ids is not None
        else float("nan"))
      if leg_ids is not None:
        self.act_legs_rad_list.append(
          (d_ar * kappa)[:, leg_ids].norm(dim=-1).mean().item())
    self.prev_actions = actions.clone()

    # 3c. Action jitter (LCP primary): d3(q_des)/dt3 in rad/s^3, legs-only.
    # q_des = q_def + kappa*a, and q_def is constant, so the third derivative of the
    # command is the third difference of kappa*a. Backward third difference
    # (a_t - 3a_{t-1} + 3a_{t-2} - a_{t-3}) / dt^3 -- exact here because the action
    # sequence carries no sensor noise, unlike the same operator on hardware logs.
    act_hist = self.act_hist
    act_hist.append(actions.clone())
    if len(act_hist) > 4:
      act_hist.pop(0)
    if len(act_hist) == 4:
      a3, a2, a1, a0_ = act_hist  # oldest -> newest
      d3 = (a0_ - 3.0 * a1 + 3.0 * a2 - a3) * kappa / (ctrl_dt ** 3)
      self.ajit_list.append(
        d3[:, leg_ids].norm(dim=-1).mean().item() if leg_ids is not None
        else float("nan"))

    # 3b. Joint acceleration (deploy diagnostic): whole-body + leg/arm-restricted
    # mean |qddot|. p95 (below, post-rollout) separates contact-impulse spikes
    # from steady-state jitter -- the mean alone conflates the two.
    jacc = robot.joint_acc.abs()
    # 3d. DoF position jitter (LCP realized-side counterpart): one difference of the
    # SIGNED joint acceleration, not of |qddot| -- differencing the absolute value
    # would miss every sign flip, which is most of what jerk is.
    if self.prev_jacc_signed is not None and leg_ids is not None:
      qjit = (robot.joint_acc - self.prev_jacc_signed) / ctrl_dt
      self.qjit_list.append(qjit[:, leg_ids].norm(dim=-1).mean().item())
    self.prev_jacc_signed = robot.joint_acc.clone()
    self.jacc_list.append(jacc.mean().item())
    self.jacc_legs_list.append(
      jacc[:, leg_ids].mean().item() if leg_ids is not None else float("nan"))
    self.jacc_arms_list.append(
      jacc[:, self.arm_ids].mean().item() if self.arm_ids is not None else float("nan"))

    # 4. Stability: orientation deviation + height deviation.
    self.orient_devs.append(
      robot.projected_gravity_b[:, :2].norm(dim=-1).mean().item()
    )
    nom_h = robot.default_root_state[:, 2]  # [B] nominal height
    self.height_devs.append(
      (robot.root_link_pos_w[:, 2] - nom_h).abs().mean().item()
    )
    # WP1 stability proxy: body roll/pitch-rate magnitude (wobble), distinct from
    # orient_dev (a position-like projected-gravity deviation) and err_yaw (the
    # commanded-axis yaw-RATE tracking error, about z only).
    self.omega_xys.append(robot.root_link_ang_vel_b[:, :2].norm(dim=-1).mean().item())

    # 5. Upper-body deploy hygiene: arm+waist drift from default + joint speed.
    ub_ids = self.ub_ids
    self.ub_pose_devs.append(
      (robot.joint_pos[:, ub_ids] - robot.default_joint_pos[:, ub_ids])
      .square().mean(dim=1).mean().item()
    )
    self.ub_arm_vels.append(robot.joint_vel[:, ub_ids].abs().mean().item())

    # 6. Mechanical power + cost of transport (gated by commanded linear speed > 0.1).
    power = (robot.qfrc_actuator * robot.joint_vel).abs().sum(dim=1)  # [B] watts
    self.powers.append(power.mean().item())
    # 6b. Copper-loss-corrected power (metric-only, WP1): adds k*tau^2 per joint
    # before summing -- see COPPER_LOSS_K comment above.
    power_copper = ((robot.qfrc_actuator * robot.joint_vel).abs()
                    + COPPER_LOSS_K * robot.qfrc_actuator.square()).sum(dim=1)
    self.powers_copper.append(power_copper.mean().item())
    lin_speed = robot.root_link_lin_vel_b[:, :2].norm(dim=-1)  # [B] achieved m/s
    eng = (cmd[:, :2].norm(dim=-1) > 0.1).float()  # commanded-motion gate
    self.energy_eng += (power * eng).sum().item() * step_dt
    self.energy_eng_copper += (power_copper * eng).sum().item() * step_dt
    self.dist_eng += (lin_speed * eng).sum().item() * step_dt
    # 6c. Stance width: |y_L - y_R| between the foot sites in the pelvis frame.
    # `walk_m` is the same commanded-motion gate as the CoT numerator above, so the
    # split matches how `cot` is scored.
    if self.bench_foot_ids is not None:
      wid = foot_pos_b(robot, self.bench_foot_ids)
      wid = (wid[:, 0, 1] - wid[:, 1, 1]).abs()  # [B] lateral separation, m
      walk_m = cmd[:, :2].norm(dim=-1) > 0.1
      self.w_walk_sum += wid[walk_m].sum().item()
      self.w_walk_n += int(walk_m.sum().item())
      self.w_stand_sum += wid[~walk_m].sum().item()
      self.w_stand_n += int((~walk_m).sum().item())

    # 7. Achieved stride period from footfall rising edges (same-foot touchdown interval).
    if self.contact_sensor is not None:
      is_contact = self.contact_sensor.data.current_contact_time > 0  # [B, n_feet]
      self.n_feet = is_contact.shape[1]
      prev_contact = self.prev_contact
      if prev_contact is not None:
        self.td_count += (is_contact & ~prev_contact).float().sum().item()
        # Step width proper: sample only on a rising edge, and only while walking.
        if self.bench_foot_ids is not None:
          td_m = (is_contact & ~prev_contact).any(dim=1) & walk_m
          if bool(td_m.any()):
            self.w_td_sum += wid[td_m].sum().item()
            self.w_td_n += int(td_m.sum().item())
      self.prev_contact = is_contact.clone()
      # 8. Gait match vs the commanded clock (H2 lock-in diagnostic): fraction of feet
      # whose contact agrees with the commanded stance schedule (same rule as feet_gait).
      hp = getattr(uenv, "hrl_phase", None)
      gait_thr = self.duty_lo
      if hp is None:  # A0 / non-cadence A1: the fixed 0.6 s clock
        hp = (uenv.episode_length_buf * step_dt) / 0.6
      elif self.gait_swing > 0.0:  # d(T) schedule, same map as feet_gait
        gait_thr = (1.0 - self.gait_swing / uenv.hrl_period).clamp(self.duty_lo, self.duty_hi).unsqueeze(1)
      leg_phase = (hp.unsqueeze(1) + self.gait_offsets) % 1.0
      self.gait_matches.append(((leg_phase < gait_thr) == is_contact).float().mean().item())

  def seed_end(self, seed_idx, seed):
    cfg, step_dt = self.ctx.cfg, self.step_dt
    def _m(lst): return float(torch.tensor(lst).mean())  # noqa: E731
    def _p95(lst):  # noqa: E731
      # NaN-poisoned lists (an empty leg/arm group) stay NaN rather than erroring out
      # of torch.quantile.
      return float("nan") if any(v != v for v in lst) else torch.quantile(torch.tensor(lst), 0.95).item()

    # Hold-eval split (stage D, 2026-07-14): steady-state errs over the last 2/3 of the
    # rollout separate the from-stand acceleration ramp from held tracking (the table-e
    # full-window numbers conflate them); t90 = time to first reach 90% of the commanded
    # vx (NaN when never reached, or for random-command aggregates where it's undefined).
    ss0 = cfg.eval_steps // 3
    if cfg.eval_cmd_vx is not None and abs(cfg.eval_cmd_vx) > 1e-6:
      sgn = 1.0 if cfg.eval_cmd_vx > 0 else -1.0
      thr = 0.9 * abs(cfg.eval_cmd_vx)
      t90 = next((i * step_dt for i, v in enumerate(self.achieved_vxs) if sgn * v >= thr),
                 float("nan"))
    else:
      t90 = float("nan")

    # The ankle spans come from the joint-limit add-on's commanded range (NaN without it).
    def _span(n): return self.limits.span(n) if self.limits is not None else float("nan")  # noqa: E731

    s = self
    res = {
      "err_vx":      _m(s.errs_vx),
      "err_vy":      _m(s.errs_vy),
      "err_yaw":     _m(s.errs_yaw),
      "ss_err_vx":   _m(s.errs_vx[ss0:]),
      "ss_err_vy":   _m(s.errs_vy[ss0:]),
      "t90_s":       t90,
      "fall_rate":   _m(s.fall_flags),
      "mean_ep_len": _m(s.ep_lens),
      "action_rate": _m(s.action_rates) if s.action_rates else float("nan"),
      "act_legs":    _m(s.act_legs_list) if s.act_legs_list else float("nan"),
      # Ankle command SPAN (rad): the deploy-relevant authority measure, since the two
      # rolls are what reject a lateral push. Scalars here rather than print-only so
      # they get the per-seed line and the seed mean+-std every other metric gets.
      "ank_roll_span_l":  _span("left_ankle_roll"),
      "ank_roll_span_r":  _span("right_ankle_roll"),
      "ank_pitch_span_l": _span("left_ankle_pitch"),
      "ank_pitch_span_r": _span("right_ankle_pitch"),
      "jacc":          _m(s.jacc_list),
      "jacc_legs":     _m(s.jacc_legs_list),
      "jacc_arms":     _m(s.jacc_arms_list),
      "jacc_legs_p95": _p95(s.jacc_legs_list),
      "jacc_arms_p95": _p95(s.jacc_arms_list),
      # LCP suite, legs-only, rad/s^3 (published Unitree H1 refs: action jitter 0.44 in
      # MuJoCo, 1.11-1.20 on real ground; NOT rate-matched to our 50 Hz -- a sanity band,
      # never a pass/fail gate).
      "ajit_legs":     _m(s.ajit_list) if s.ajit_list else float("nan"),
      "ajit_legs_p95": _p95(s.ajit_list) if s.ajit_list else float("nan"),
      "qjit_legs":     _m(s.qjit_list) if s.qjit_list else float("nan"),
      "qjit_legs_p95": _p95(s.qjit_list) if s.qjit_list else float("nan"),
      # Commanded agitation in PHYSICAL units (rad/step): act_legs is raw action units.
      "act_legs_rad":  _m(s.act_legs_rad_list) if s.act_legs_rad_list else float("nan"),
      "orient_dev":  _m(s.orient_devs),
      "height_dev":  _m(s.height_devs),
      "ub_pose_dev": _m(s.ub_pose_devs),
      "ub_arm_vel":  _m(s.ub_arm_vels),
      "mech_power_w": _m(s.powers) if s.powers else float("nan"),
      "gait_match":   _m(s.gait_matches) if s.gait_matches else float("nan"),
      "cot":          s.energy_eng / (s.dist_eng * MASS_G + 1e-6),
      # WP1 payload pilot additions (2026-08-27): copper-loss CoT variant (metric-only,
      # see COPPER_LOSS_K), body roll/pitch-rate wobble, and the F5-analogue cross-env
      # vx spread under a pinned command (meaningful only with --eval-cmd-vx set).
      "mech_power_copper_w": _m(s.powers_copper) if s.powers_copper else float("nan"),
      "cot_copper":   s.energy_eng_copper / (s.dist_eng * MASS_G + 1e-6),
      "omega_xy":     _m(s.omega_xys) if s.omega_xys else float("nan"),
      "ss_vx_var":    _m(s.achieved_vx_vars[ss0:]) if s.achieved_vx_vars else float("nan"),
      # Stance width in metres. Nominal geometry (hip offsets, hip_roll=0) is 0.326 m,
      # so a value near that IS the feet-together posture, and larger means abducted.
      "stance_w_walk":  s.w_walk_sum / s.w_walk_n if s.w_walk_n else float("nan"),
      "stance_w_stand": s.w_stand_sum / s.w_stand_n if s.w_stand_n else float("nan"),
      "stance_w_td":    s.w_td_sum / s.w_td_n if s.w_td_n else float("nan"),
      "stride_period_s": (cfg.eval_steps * step_dt)
                         / max(s.td_count / max(s.n_envs * max(s.n_feet, 1), 1), 1e-6),
      # WP2: actual sampled payload (mean over envs), see the reset()-time read above.
      "payload_kg": s.payload_kg,
    }
    if tuple(res) != SEED_KEYS:
      raise RuntimeError(f"[BENCH] per-seed keys drifted from SEED_KEYS: {sorted(set(res) ^ set(SEED_KEYS))}")
    self.seed_results.append(res)
    # WP1 payload pilot: a per-seed line, since [BENCH] below only ever prints the
    # seed-aggregated mean+-std. The sweep CSV needs one row per (T, payload, vx, seed)
    # for the T* seed-spread analysis, which the aggregate's std alone cannot give
    # (fitting T* per seed, then spreading THAT, is not the same as spreading the metric).
    print(f"[BENCH_SEED] {json.dumps({**self.bench_meta, 'seed': seed, **res})}")


def run(ctx) -> dict:
  cfg = ctx.cfg
  addons = build_addons(ctx)
  bench = Bench(ctx, addons.get("limits"))
  rollout(ctx, cfg.eval_steps, [*addons.values(), bench], pass_dones=ctx.policy_wants_dones)

  # Aggregate across seeds.
  keys, means, stds = mean_std(bench.seed_results, cfg.eval_seeds)

  def _fmt(k): return f"{means[k]:.4f} ± {stds[k]:.4f}" if cfg.eval_seeds > 1 else f"{means[k]:.4f}"  # noqa: E731

  print()
  print("=" * 58)
  for a in addons.values():
    a.report()
  print(f"  BENCHMARK SCORECARD  |  {cfg.eval_steps} steps x {bench.n_envs} envs x {cfg.eval_seeds} seed(s)")
  print("=" * 58)
  print(f"  Tracking  err_vx    : {_fmt('err_vx')}")
  print(f"  Tracking  err_vy    : {_fmt('err_vy')}")
  print(f"  Tracking  err_yaw   : {_fmt('err_yaw')}")
  print(f"  Survival  fall_rate : {_fmt('fall_rate')}")
  print(f"  Survival  ep_len    : {_fmt('mean_ep_len')}")
  print(f"  Smoothness act_rate : {_fmt('action_rate')}")
  print(f"  Smoothness act_legs : {_fmt('act_legs')}")
  print(f"  Accel     jacc      : {_fmt('jacc')}")
  print(f"  Accel     jacc_legs : {_fmt('jacc_legs')}")
  print(f"  Accel     jacc_arms : {_fmt('jacc_arms')}")
  print(f"  Accel     jacc_legs_p95: {_fmt('jacc_legs_p95')}")
  print(f"  Accel     jacc_arms_p95: {_fmt('jacc_arms_p95')}")
  print(f"  Stability orient_dev: {_fmt('orient_dev')}")
  print(f"  Stability height_dev: {_fmt('height_dev')}")
  print(f"  UpperBody pose_dev  : {_fmt('ub_pose_dev')}")
  print(f"  UpperBody arm_vel   : {_fmt('ub_arm_vel')}")
  print(f"  Energy    power_W   : {_fmt('mech_power_w')}")
  print(f"  Energy    CoT (norm): {_fmt('cot')}")
  print(f"  Energy    CoT copper: {_fmt('cot_copper')}")
  print(f"  Stability omega_xy  : {_fmt('omega_xy')}")
  print(f"  Stability ss_vx_var : {_fmt('ss_vx_var')}")
  print(f"  Stance    width_walk: {_fmt('stance_w_walk')}  (nominal 0.326 m)")
  print(f"  Stance    width_stand: {_fmt('stance_w_stand')}")
  print(f"  Stance    width_td  : {_fmt('stance_w_td')}  (at touchdown, walking)")
  print(f"  Gait      stride_s  : {_fmt('stride_period_s')}")
  print(f"  Gait      match     : {_fmt('gait_match')}")
  print(f"  Env       payload_kg: {_fmt('payload_kg')}")
  if cfg.eval_cmd_vx is not None:
    print(f"  Hold      ss_err_vx : {_fmt('ss_err_vx')}  (last 2/3)")
    print(f"  Hold      ss_err_vy : {_fmt('ss_err_vy')}")
    print(f"  Hold      t90_s     : {_fmt('t90_s')}")
  print("=" * 58)
  print()

  bench_out = {**bench.bench_meta, **{k: round(means[k], 6) for k in keys},
               **{f"{k}_std": round(stds[k], 6) for k in keys},
               "eval_steps": cfg.eval_steps, "num_envs": bench.n_envs, "eval_seeds": cfg.eval_seeds}
  for a in addons.values():
    a.finish(bench_out)
  print(f"[BENCH] {json.dumps(bench_out)}")
  return bench_out
