# A0 model delta vs upstream `unitree_rl_mjlab` (written 2026-07-23, after first working hardware deployment)

Purpose: catalog every difference between this repo's H1-2 robot model and
`unitreerobotics/unitree_rl_mjlab` (the upstream template this repo was forked from,
`git remote upstream`), as candidate explanations for why this fork's A0 policy
deployed successfully on hardware. Method: `git diff upstream/main main` restricted to
model-relevant files, read line-by-line against the actual file contents (not
reconstructed from ADR/memory narrative — two things I expected to be fork changes
turned out to be upstream defaults, caught only by diffing directly). Merge-base:
`1425b15` (upstream tip); this fork's `main` is strictly ahead, so the diff is exactly
"everything changed since forking," no upstream-only commits to account for.

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
- **Armature (correction — see below, was wrongly listed as a difference in an
  earlier version of this doc)**: **identical to upstream, per-motor, for all four
  original joint groups** — 0.025 (hip/torso), 0.04 (knee), 0.005 (ankle/shoulder p/r),
  0.002 (shoulder-yaw/elbow/wrist). Confirmed by reading upstream's raw file directly
  (`git show upstream/main:...h1_2_constants.py`), not the line-diff, which
  misleadingly showed some armature lines as "added" because the surrounding block was
  restructured (torso/shoulders split into their own groups) — a diff-alignment
  artifact, not a value change. When the fork split a bundled group into finer
  sub-groups (e.g. torso out of the old hip+torso group), each new sub-group simply
  inherited its parent's original armature number. **Source: `unitree_rl_mjlab`'s own
  `h1_2_constants.py` asset file (upstream, i.e. the fork's own template), not
  IsaacLab** — no H1-2 asset file exists in the separate `mjlab` framework package
  itself (`find` turns up nothing), and no matching per-motor figures were found in
  the workspace's IsaacLab-derived H1-2 repos on a first pass. Upstream's ADR-era
  note (`docs/adr/0005` §"non-gaps") already flagged this: armature values disagree
  *between* other references (0.01 flat in the Unitree-derived XML reference, 1e-3
  flat in the IsaacGym cfg reference) and ours were explicitly "kept" as the
  pre-existing per-motor figures rather than matched to either — "kept" relative to
  upstream's own template, confirming the numbers were never fork-authored.
- **DR event structure**: `push_robot` (interval + velocity range), `foot_friction`
  (0.3-1.6 range), `base_com` offset, `encoder_bias` — all already present upstream in
  the shared `velocity_env_cfg.py`, values unchanged. This fork did not touch domain
  randomization ranges.
- **No control-latency/actuation-delay modeling in either.** `BuiltinPositionActuatorCfg`
  has `delay_min_lag`/`delay_max_lag`/`delay_hold_prob` fields (mjlab supports this);
  neither upstream nor this fork sets them away from 0. Shared gap, not a difference.

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

## Deploy-config factors (`config.yaml` + `deploy_real.yaml`, both diffed vs upstream)

These are inside `unitree_rl_mjlab`'s tree (unlike the bridge plant, which lives in a
separate `/opt/unitree_mujoco` clone — still pointed to below) but they configure the
runtime controller rather than the trained model. Diffed directly, same method as above.

### 5. `deploy_real.yaml` is a fork-added file split off `deploy.yaml` — for DDS domain
hygiene, not because upstream's single file was sim-only or physically incapable of
driving real hardware (correction, per the user: upstream's design already intends
`deploy.yaml` for **both** real and sim)

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

Ruled out (verified identical, not candidates): joint friction (0 in both), **armature
(0 diffs — identical per-motor values 0.025/0.04/0.005/0.002, inherited unchanged from
`unitree_rl_mjlab`'s own `h1_2_constants.py`, not from IsaacLab)**, sim dt/decimation,
`desired_kl`, `entropy_coef`, effort limits, robot XML/geometry, DR event ranges,
actuation delay (absent in both), command ranges in `deploy_real.yaml` (byte-identical
to `deploy.yaml` and to upstream).

Still outside this doc's evidence base (separate git clone, not diffable the same
way): the `/opt/unitree_mujoco` bridge plant choice, §8 above.
