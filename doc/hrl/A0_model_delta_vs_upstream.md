# A0 model delta vs upstream `unitree_rl_mjlab`

Purpose: catalog every difference between this repo's H1-2 robot model and
`unitreerobotics/unitree_rl_mjlab` (the upstream template this repo was forked from,
`git remote upstream`), and explain why each one is there. Method: `git diff
upstream/main main` restricted to model-relevant files, read line-by-line against the
actual file contents rather than reconstructed from narrative. Merge-base: `1425b15`
(upstream tip); this fork's `main` is strictly ahead, so the diff is exactly
"everything changed since forking," no upstream-only commits to account for.

> **Re-verified 2026-09-22.** Merge-base is still `1425b15` and upstream has no commits
> since it, so the premise above holds unchanged. That pass added §4b (the disabled second
> velocity-curriculum stage, which the original missed and which bounds every velocity
> claim at 1.0 m/s) and three entries to the ruled-out list (`rel_standing_envs`,
> `env_latent_e`, `cost_of_transport`).
>
> **Re-verified in that pass:** §1 (gains: hip yaw/pitch/roll 98.7/6.3 → 200/2.5, torso
> 300/3.0, 157.7/10.1 → 300/4.0), §3 (`viscous_damping=0.001`), `frictionloss=0.0` and
> armature 0.025, all live in `h1_2_constants.py` against `upstream/main`; §4 via the
> `velocity_env_cfg.py` diff. §2 follows mechanically from §1. **NOT re-verified in that
> pass:** §5-§8, the deploy-config factors — they are unchanged since the original audit as
> far as this pass could tell, but were not re-diffed line by line. The Model v3
> training-side changes never landed here (reverted, `docs/adr/0009`), so nothing in this
> document is affected by them.

> The investigative version of this document — the causal-weight reasoning behind
> the ranking, and the corrections it made to its own earlier claims — is kept in
> the author's research knowledge base, outside this repo. What follows is the
> verified delta itself.

Scope: the robot **model** (actuator/plant parameters) that ships in the trained
policy. Reward shaping, the A1/HRL architecture, and the C++ deploy pipeline are
separate systems (deploy is addressed briefly at the end since upstream's own deploy
code is wholesale replaced, not incrementally patched).

## Ruled out — verified identical to upstream, do not re-investigate these

- **Robot geometry/inertia/collision XML**: `src/assets/robots/unitree_h1_2/xmls/`
  is byte-identical to upstream (`git diff --stat` empty). No mass, inertia, mesh, or
  collision difference anywhere.
- **`desired_kl=0.01`**: upstream's own `h1_2/rl_cfg.py` already ships this value
  (confirmed by reading the upstream file directly, not just the diff — the diff only
  added a comment). The whole ADR-0005 kl-0.005-vs-0.01 story is this project's own
  internal exploration (someone tried 0.005 during the v2 gain bisect, then reverted
  to upstream's original 0.01); it is **not** a fork-vs-upstream difference and cannot
  explain divergent behavior from a stock checkout.
- **`entropy_coef=0.01`**: also upstream's own default for A0 (A1 later overrides
  this to 0.005; A0 doesn't).
- **Sim timestep/decimation**: `timestep=0.005`, `decimation=4` (50 Hz control) —
  unchanged, confirmed via direct diff (no hits on `timestep=`/`decimation=` in the
  `velocity_env_cfg.py` diff).
- **Effort limits**: 200/300/40/18 per joint group — identical values in both
  (upstream just grouped torso under the 200 hip group instead of listing it
  separately; the number is the same).
- **`frictionloss`**: **0 in both.** Upstream leaves the actuator cfg field unset
  (`None`), which resolves to MuJoCo's own default (0, since the H1-2 XML's joints
  carry no `<joint ... frictionloss>` attribute at all — verified by grep). The fork
  sets it explicitly to `0.0` on every group — same effective value, just made
  explicit and commented. WL-E's fric-0.1 control run showed 0.1 also trains fine at
  kl 0.01, but that is a **live, unadopted** option (`docs/adr/0005` Amendment 2) —
  today's shipped model still trains at 0 friction, identical to upstream.
- **Armature**: **identical to upstream, per-motor, for all four original joint
  groups** — 0.025 (hip/torso), 0.04 (knee), 0.005 (ankle/shoulder p/r), 0.002
  (shoulder-yaw/elbow/wrist). Verify by reading upstream's raw file
  (`git show upstream/main:...h1_2_constants.py`) rather than the line-diff: the diff
  shows some armature lines as "added" purely because the surrounding block was
  restructured (torso/shoulders split into their own groups). When the fork split a
  bundled group into finer sub-groups, each new sub-group inherited its parent's
  original armature number. **Source: `unitree_rl_mjlab`'s own `h1_2_constants.py`
  asset file, not IsaacLab** — no H1-2 asset file exists in the separate `mjlab`
  framework package itself. Note that armature values disagree *between* other
  references (0.01 flat in the Unitree-derived XML reference, 1e-3 flat in the
  IsaacGym cfg reference); ours are upstream's per-motor figures, kept unchanged
  (`docs/adr/0005` §"non-gaps").
- **DR event structure**: `push_robot` (interval + velocity range), `foot_friction`
  (0.3-1.6 range), `base_com` offset, `encoder_bias` — all already present upstream in
  the shared `velocity_env_cfg.py`, values unchanged. This fork did not touch domain
  randomization ranges.
- **No control-latency/actuation-delay modeling in either.** `BuiltinPositionActuatorCfg`
  has `delay_min_lag`/`delay_max_lag`/`delay_hold_prob` fields (mjlab supports this);
  neither upstream nor this fork sets them away from 0. Shared gap, not a difference.
- **Standing-command fraction `rel_standing_envs`** (added to this list 2026-09-22):
  `velocity_env_cfg.py:187` reads **0.05**, and upstream's own `velocity_env_cfg.py:169`
  reads **0.05** as well. Identical, for **both** architectures. ⚠ Do not confuse the
  repo default with what a given run trained on: the A1a arms set it **per run on the
  launch command line** (the deployed keeper's `params/env.yaml:1757` records `0.12`),
  which is precisely why the A0 anchor never followed and three months of
  "A1a stands quieter than A0" ratios were confounded. A parameter that lives in a
  launch flag rather than a config file is invisible to a config diff. See
  `doc/hrl/benchmark_blindness_evidence.md` §6.5.
- **Critic-only privileged latent `env_latent_e`** and the **`cost_of_transport` reward
  term**: both are fork additions to `velocity_env_cfg.py`, but neither changes the
  plant or the deployable actor. `env_latent_e` is an observation on the critic group
  only (same asymmetric actor-critic pattern as Rough's `height_scan`), and
  `cost_of_transport` ships at **weight 0.0**, i.e. byte-identical to upstream's reward
  until deliberately enabled. Listed so a reader diffing the file does not mistake them
  for model changes.

## Actual modelling differences, ranked by plausible causal weight

### 1. PD stiffness/damping gains + actuator regrouping (the big one)

Upstream bundles joints into 4 loose groups with comparatively soft gains; this fork
(Model v2 / ADR-0005, "option B") splits into 7 groups and raises gains toward the
robot's real deploy/hold gains:

| joint group | upstream kp/kd | fork kp/kd |
|---|---|---|
| hip yaw/pitch/roll | 98.7 / 6.3 (bundled w/ torso) | 200 / 2.5 |
| torso | 98.7 / 6.3 (bundled w/ hip) | **300 / 3** (own group) |
| knee | 157.7 / 10.1 | 300 / 4 |
| ankle pitch/roll | 19.7 / 1.3 (bundled w/ shoulder p/r) | 40 / 2 |
| shoulder pitch/roll | 19.7 / 1.3 (bundled w/ ankle) | **120 / 2** (own group) |
| shoulder yaw | 7.9 / 0.5 (bundled w/ elbow/wrist) | **120 / 2** (own group) |
| elbow + wrists | 7.9 / 0.5 | 80 / 1 |

Confirmed the same numbers are in upstream's own `deploy.yaml` — upstream's deploy
pipeline uses these same soft gains, so this isn't a train/deploy mismatch on
upstream's side, just uniformly softer everywhere. ADR-0005's stated mechanism: `kp=7.9`
on the arms only produces sane behavior in a **frictionless** sim; real joint stiction
swallows that authority, so a policy trained at upstream's gains is assuming an
actuator strength the real robot's arms don't deliver. Splitting torso out of the hip
group and shoulders out of the ankle group additionally lets each body region get a
gain suited to its role (legs stay dynamic, torso/arms move toward deploy-realistic
hold behavior) instead of one number serving both a leg-adjacent and an arm-adjacent
joint.

### 2. Derived action scale (direct consequence of #1)

Both repos use the identical formula (`H1_2_ACTION_SCALE = 0.25 * effort_limit / kp`,
same line of code) — the scale itself was never touched. But because kp rose sharply
for torso/shoulders/arms, the resulting per-action-unit range shrank a lot for exactly
those joints:

| joint group | upstream scale (rad/unit) | fork scale |
|---|---|---|
| legs | 0.48-0.51 | 0.25 |
| torso | 0.51 | 0.167 |
| shoulder pitch/roll | 0.51 | 0.083 |
| shoulder yaw | 0.57 | 0.0375 |
| elbow + wrists | 0.57 | 0.056 |

The arm/shoulder-yaw authority is ~10x smaller per action unit than upstream's. This
means exploration noise during training can no longer fling the arms/torso around —
directly relevant to why the fork's arms look calm in deployment while a
stock-upstream policy trained at 0.51-0.57 rad/unit arm authority would have much
larger raw excursions to begin with.

### 3. Passive joint damping: none upstream, `0.001` here

Same pattern as `frictionloss` above: upstream leaves `viscous_damping` unset (→ 0);
the fork sets `0.001` on every joint explicitly — but unlike `frictionloss` (both 0)
and armature (both identical, see "Ruled out"), this one is a genuine nonzero
addition. Small in magnitude, but it is the exact value the vendor
deploy-bridge plant also uses (`damping 0.001`), so it's one more point of parity
between what this fork trained against and what it was deployed onto — upstream's
policy trained against a perfectly undamped joint has no equivalent parity point.

### 4. Command resampling range (RL/curriculum, not physical, flagged as secondary)

Upstream/this fork's own earlier default: hold each sampled velocity command
(3.0, 8.0) s before resampling. This fork changed it to (3.0, 20.0) s on 2026-07-14 —
20 s equals the episode length, so a fully-sustained command (the treadmill/deploy
case: "walk at 0.5 m/s and don't change") is now an in-distribution training
scenario, not an edge case the resampler always interrupted before the policy had to
cope with it. Not a plant/actuator change, but plausibly relevant to why sustained
commanded walking held up in real deployment rather than drifting into an
undertrained regime.

### 4b. Velocity curriculum: upstream's second stage is DISABLED (added 2026-09-22)

Missed by the original pass, and it bounds every velocity claim in the thesis.

Upstream runs a **two-stage** velocity curriculum: stage 0 at `lin_vel_x (-0.5, 1.0)`,
`lin_vel_y (-0.5, 0.5)`, then a second stage at `step = 5000 * 24` widening to
`lin_vel_x (-1.0, 2.0)`, `lin_vel_y (-1.0, 1.0)`. **This fork comments the second stage
out**, leaving a single stage at step 0 (`velocity_env_cfg.py`, `velocity_stages`).

So the trained command distribution is **`lin_vel_x ∈ [-0.5, 1.0]`,
`lin_vel_y ∈ [-0.5, 0.5]`, `ang_vel_z ∈ [-1.0, 1.0]` for the entire run**, and the policy
never sees a forward command above **1.0 m/s**, where upstream's would end at 2.0.

⚠ **The declared range in the config is misleading.** `commands.twist.ranges` still reads
`lin_vel_x (-1.0, 2.0)` / `lin_vel_y (-1.0, 1.0)`; the curriculum overwrites it at step 0
and never advances. Reading the `ranges` block alone gives the wrong answer. Confirmed on
the deployed keeper's own `params/env.yaml` (ranges block vs `curriculum.command_vel`).

**Independent cross-check:** the baked HIRO goal scale is `[0.75, 0.5, 1.0]`, and the scale
is a half-range (`half = (hi-lo)/2`). 0.75 → a 1.5-wide vx range, 0.5 → 1.0-wide vy,
1.0 → 2.0-wide yaw. Exactly the curriculum's stage-0 values, derived from a completely
different file.

**Consistent with deploy, which is the reassuring part:**
`velocity_hrl/v0/params/deploy_real.yaml:91-92` sets `lin_vel_x [-0.5, 1.0]` and
`lin_vel_y [-0.5, 0.5]`, i.e. the **effective trained** range rather than the declared one,
so the operator cannot command the robot outside its training distribution in x or y.
`ang_vel_z` is ±1.0 in both (see `docs/adr/0009`: the older ±0.5 "tame joystick" setting was
abandoned 2026-08-27).

**Why it matters for the thesis:** every tracking number, every hold evaluation and every
sim-to-real velocity claim is bounded at 1.0 m/s forward. State the bound once in the setup
chapter rather than letting a reader infer 2.0 m/s from the `ranges` block.

## Deploy-config factors (`config.yaml` + `deploy_real.yaml`, both diffed vs upstream)

These are inside `unitree_rl_mjlab`'s tree (unlike the bridge plant, which lives in a
separate `/opt/unitree_mujoco` clone — still pointed to below) but they configure the
runtime controller rather than the trained model. Diffed directly, same method as above.

### 5. `deploy_real.yaml` is a fork-added file split off `deploy.yaml` — for DDS domain
hygiene

Upstream's design already intends `deploy.yaml` for **both** real and sim, so this
split is not a gap-fill.

Upstream's `main.cpp` hardcodes DDS `domain=0` always and distinguishes sim vs real
**only** by which `--network` interface string is passed at launch (`lo` for local
sim loopback, the robot's NIC for real) — one YAML, one gain/scale table, used for
both targets. This fork added `H1_2_DOMAIN_ID` env-var/`config.yaml` `domain_id`
switching (domain 1 for local sim, domain 0 for real) specifically because running
local sim traffic on domain 0 risks colliding with an actual robot's DDS traffic on
the same network ("sim with remote makes no sense" for a workstation that might share
a network with a real H1-2) — a networking/isolation convenience, **not** a physical-
model gap upstream left unfilled. The file split into `deploy.yaml` (sim) /
`deploy_real.yaml` (real) followed from wiring that domain choice through
`H1_2_DEPLOY_CFG` in the setup scripts, and both files carry byte-identical v2
gains/scale/command-ranges (confirmed by diff) — the split itself is not a candidate
cause of deployment success.

The one real content difference introduced along the way:

- **`hold_joint_ids: [12..26]` exists only in `deploy_real.yaml`** (split deploy,
  ADR-0005 step 3b): on real hardware, torso + both arms track `default_joint_pos`
  directly instead of the policy's action; only the 12 leg joints are ever
  policy-driven on the robot. The policy still *observes* all 27 joints (unchanged
  obs vector) but its arm/torso actions are discarded at the FSM level
  (`State_RLBase.cpp`). This has no upstream equivalent at all (upstream's one-file
  design drives all 27 joints from the policy on both sim and real). Given §1-2 above
  (this fork's torso/arm gains are hold-tuned specifically because they're held, not
  because the policy learned fine motor control there), an upstream-style deploy that
  put the trained policy's raw arm/torso actions directly on hardware — at upstream's
  soft gains, with no hold override — is a plausible independent failure mode, but
  this is now a genuinely fork-specific mechanism, not a upstream-gap-fill.

### 6. `config.yaml` FixStand handover posture (2026-07-23, very recent)

`qs` for the FixStand pose changed from `hip_pitch -0.3 / ankle_pitch -0.2` (upstream)
to `hip_pitch -0.2 / ankle_pitch -0.3` (fork) — aligned to match `default_joint_pos` in
`deploy.yaml`/`deploy_real.yaml` (the policy's own action offset), so the walking
policy takes over from the exact posture its action offset assumes instead of
correcting ~0.1 rad on two joint pairs in its first control ticks after handover. Dated
the same day as this doc — i.e. this is one of the most recent changes preceding the
successful deployment, and directly targets the FixStand-to-policy handover transient
(not the backward-lean issue from `A1a_deploy_plan.md`, per the code comment). Mirrored
by hand in `scripts/bridge_replica.py`'s `FIX_Q` constant.

### 7. Runtime safety filter + joint-limit header: absent upstream, net-new here

`deploy/robots/h1_2/include/h1_2_limits.h` and `safety_logger.h` are entirely new
files (upstream has neither) — a stock deploy has no tilt/fall-triggered ramp-down and
no per-joint limit clamp/hold at all. Not a "model" change (it doesn't affect what the
policy trained against) but directly relevant to why a real session survives a
recoverable stumble instead of the policy's raw output driving straight through a
mechanical stop. Details: `.claude/docs/deployment.md` "Safety filter + flight
recorder".

### 8. Bridge plant choice — outside this repo's diff surface entirely

Lives in the separate `/opt/unitree_mujoco` clone (its own git, not part of
`unitree_rl_mjlab`): vendor reference plant (`armature 0.01 / frictionloss 0.1 /
damping 0.001`) vs an old harsher default. Pointer only:
`.claude/docs/deployment.md` "Bridge plant" section, `doc/hrl/A1a_deploy_plan.md`
post-mortem.

## Summary ranking (most to least likely to explain "mine works, stock doesn't")

1. **PD gain restructuring** + the resulting ~10x smaller derived arm/shoulder-yaw
   action scale — the single largest, best-evidenced change (ADR-0005's own bisect:
   upstream-equivalent gains/scale combinations repeatedly stalled or fell there).
2. **Split deploy (`hold_joint_ids`, real-only)** — upstream has no mechanism to hold
   the upper body on either target (one file, all 27 joints always policy-driven);
   combined with #1, an upstream-style deploy would put a policy's raw, softly-gained
   arm/torso actions directly on hardware, undamped by any hold behavior. (The
   `deploy.yaml`/`deploy_real.yaml` file split itself is DDS-domain hygiene, not a
   causal factor — see §5.)
3. **Runtime safety filter** (`h1_2_limits.h`/`safety_logger.h`) — entirely absent
   upstream; without it a stock deploy has no tilt/fall ramp-down and no joint-limit
   response at all.
4. Nonzero passive joint damping (0.001) — small, but a direct parity point with the
   deploy-bridge plant.
5. FixStand handover posture alignment (2026-07-23) — reduces a first-tick correction
   transient at the exact moment control hands from FixStand to the walking policy.
6. Command resampling range (3,20) vs (3,8) — secondary, affects whether sustained
   commands are in-distribution.
7. **Velocity curriculum truncated to stage 0** (added 2026-09-22, §4b) — not a candidate
   for "mine works, stock doesn't" in the *stability* sense, but it BOUNDS every velocity
   claim: trained `lin_vel_x` tops out at 1.0 m/s where upstream reaches 2.0. Listed here
   because the `ranges` block reads (-1.0, 2.0) and is overwritten at step 0, so the diff
   is easy to miss.

Ruled out (verified identical, not candidates): joint friction (0 in both), **armature
(0 diffs — identical per-motor values 0.025/0.04/0.005/0.002, inherited unchanged from
`unitree_rl_mjlab`'s own `h1_2_constants.py`, not from IsaacLab)**, sim dt/decimation,
`desired_kl`, `entropy_coef`, effort limits, robot XML/geometry, DR event ranges,
actuation delay (absent in both), command ranges in `deploy_real.yaml` (byte-identical
to `deploy.yaml` and to upstream).

Still outside this doc's evidence base (separate git clone, not diffable the same
way): the `/opt/unitree_mujoco` bridge plant choice, §8 above.
