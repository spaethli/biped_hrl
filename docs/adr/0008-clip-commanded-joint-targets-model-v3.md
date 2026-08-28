# Clip commanded joint targets to the hardware limits, and correct leg mass (Model v3)

**Status: SUPERSEDED BY ADR-0009 (2026-08-28), and REVERTED.** The parity gap diagnosed
below is real and the diagnosis stands; the fix cost 2.1-2.4x of the commanded ankle roll
range and bought nothing, because the firmware already truncated the commands the clip was
added to truncate. Read ADR-0009 for the measurements and the decision. Nothing here is
applied. Originally: accepted (2026-08-25), **superseded ADR-0006**, which defined Model v3 as leg mass
+ foot sole geometry; v3 is redefined here as **action clip + leg mass**. ADR-0006 keeps its
investigation record (V1 falsification, Spec A rejection, Spec B confirmation) and its foot-sole
geometry proposal, which stays staged and uncut.

Training never priced the commanded joint target leaving the hardware limits, while the robot
truncates it. **We clip the processed action to the hard joint limits in training AND in every
deploy config, and add an L1 penalty on the clipped excess so the saturated region stays
discriminable.** The measured leg-mass error is corrected in the same cut.

## Context

The C++ deploy path clamps each commanded joint position to `h1_2_joint_limits`
(`State_RLBase.cpp`), so a target past a mechanical stop is silently truncated and the joint
does something other than what the policy intended. Training had no equivalent. The existing
`joint_pos_limits` reward (weight -10.0, `velocity_env_cfg.py:308`) does not cover this: it
penalises `asset.data.joint_pos`, the **measured** position against the 0.9 soft band, and
MuJoCo clamps `qpos` to the joint range anyway, so the term is close to inert on the ankles
while the *command* is unpriced.

Measured on the deployed A0 baseline (`a0_v2_optB_rs20_baseline/model_10000.pt`, md5-verified
against `PROVENANCE.json`), commanded targets against the same audited bounds:

| joint | sim rate | sim max over | hardware rate | hardware max over |
|---|---|---|---|---|
| left_ankle_pitch | 0.0216 | 0.926 rad | 0.1246 | 1.395 rad |
| right_ankle_pitch | 0.0183 | 0.997 | 0.1482 | 0.866 |
| left_ankle_roll | 0.0082 | 0.226 | 0.0779 | 0.359 |
| right_ankle_roll | 0.0042 | 0.131 | 0.0437 | 0.337 |

Sim numbers from `play.py --check-joint-limits` (600 steps × 64 envs × 2 seeds); hardware from
`safety_analyzer.py`'s `raw_policy_violations` on the 2026-08-25 sessions. Violations are ~100%
ankles on both sides, in the same ranking, and `left_ankle_pitch` has a legal span of 1.42 rad -
so the policy commands roughly a full extra range past the stop. Measured violations are ~0, so
the stops hold; this is entirely commanded. The command distributions differ (sim = random
resampled twist, hardware = mostly standing), so the ~6-10x rate gap is indicative, but the
joint identity and the magnitudes are not in doubt.

The defect is therefore **created in training**, not merely amplified on hardware: gradient
descent has no reason to avoid a region that costs nothing.

Separately, the nominal leg mass is measurably wrong: the real robot is 73.70 kg (head off) with
an 18.60 kg leg, against 66.98 kg and 15.399 kg in the model, the entire 6.7 kg gap in the legs.
That value is byte-identical across five sims and the official Unitree URDFs, so it is a
manufacturer CAD-vs-real discrepancy inherited by every downstream H1-2 sim.

## Considered options

**A penalty without a clip** would keep a smooth gradient everywhere and avoid a saturated
region, but sim still would not reproduce the truncation the robot applies. The parity gap would
be discouraged, not closed, and the policy could still learn a trajectory it cannot execute.

**A clip without a penalty** gives exact parity for one config line, but every action past the
bound produces an identical outcome, so nothing distinguishes 0.1 rad over from 1.0 rad over and
the policy can park in saturation. We take both: the clip for parity, the penalty for
discrimination inside the region the clip creates.

**Clipping at the 0.9 soft limits** (what `joint_pos_limits` already uses) would be consistent
with the existing reward and leave margin against encoder noise grazing the stop. Rejected: it
makes training *stricter* than deploy, so it stops being a parity fix and becomes a behaviour
change layered on one. The hard limits are what the firmware and the deploy clamp enforce, and
`test_deploy_joint_limits_match_the_training_model` already pins them equal to the MJCF
`jnt_range`, so there is one table.

**Relying on the `State_RLBase` safety clamp for the deploy side** would need no config change,
but it is not the same operation: it clamps `(1-α)·action + α·q_meas` **after** the hold blend
and before the `joint_offset` reversal, so it coincides with a training clip only at α = 0. The
deploy action term honours a yaml `clip` at the identical pipeline point as mjlab
(`isaaclab/envs/mdp/actions/joint_actions.h:54`), so setting it there makes the two bit-comparable
and demotes the safety clamp to a position backstop that should never fire.

**Landing the clip alone and leaving leg mass staged** was the initial preference and was
reversed deliberately (see Consequences: attribution).

## Decision

Model v3 comprises four changes, cut together and applied uniformly to all architectures:

1. **Clip the processed action to the hard joint limits**, all 27 joints, in
   `config/h1_2/env_cfgs.py` where the h1_2 action term is already patched. Bounds are the MJCF
   `jnt_range`, i.e. the same table as `h1_2_limits.h`.
2. **Penalise the clipped excess**, L1 sum of per-joint `|preclip - clipped|` in radians. L1, not
   L2, matching the form of the existing `joint_pos_limits` term: L2 would let one large
   excursion dominate a hundred small ones, and the small persistent ones produce the 12-15%
   hardware rate. **The excess must be recomputed as `raw_actions * scale + offset`**, because
   mjlab's `process_actions` overwrites `_processed_actions` with the clamped value and the
   pre-clip target is not retained.
3. **Set the same `clip` table in all eight deploy yamls** (A0 and A1 × `deploy`, `deploy_est`,
   `deploy_est_pin0625`, `deploy_real`).
4. **Correct leg mass** per ADR-0006's spec verbatim: the 6 leg links per side scaled by
   **1.20787 = 18.6 / 15.399** on `mass` and all three `diaginertia`, `ipos`/`quat` untouched so
   no CoM shift is introduced. New total 73.386 kg, one leg 18.600 kg.

**The mass correction is justified as fidelity, not as a lean fix.** ADR-0006's validation
experiment V1 cut leg mass alone, retrained A0, re-measured the lean on the robot, and found it
moved nothing. That result stands and is not relitigated here; the mass is corrected because it
is measurably wrong, and the lean claim is explicitly not made.

The penalty is routed **twice**, because A0 and A1 do not share a training signal: A0's low level
trains on the env reward, while A1's trains only on the intrinsic reward
(`ll_task_reward_coef = 0`), so an env `RewardTermCfg` never reaches it. A1 gets a new
`ll_action_clip_coef` alongside the existing `ll_joint_limits_coef`.

**Weight sizing: -3.5** (A0 `weight`), mirrored into A1 as `ll_action_clip_coef = 3.5`. The
sign differs by construction, not by choice: A0 applies a signed `weight`, while the A1 runner
computes `r_lo - coef * excess`, so the mirror of `-3.5` is `+3.5`. "Verbatim" means the same
magnitude.

The rule originally adopted, matching what `joint_pos_limits` (-10.0) contributes at the baseline
violation level, **turned out to be unexecutable, and its failure is itself evidence for this
ADR.** Measured on a trained A0 rollout, `joint_pos_limits` contributes **< 5e-7 per step**,
against `action_rate_l2` at 0.0306 and `track_linear_velocity` at 0.875, so matching it yields a
weight of ~0. That term scores *measured* position against the 0.9 soft band, and a competent
policy never goes there because MuJoCo's `qpos` clamp keeps it out. A zero-action control probe,
which does drive joints into their stops, reads 0.0284, five orders of magnitude larger, which is
why the historical WL-D audit's -0.00025 was not representative of a trained policy.

The rule was therefore re-anchored to the smallest currently-*live* shaping term. At the measured
total mean excess of **0.008676 rad/step** (essentially all ankles: `left_ankle_pitch` 0.005360,
`right_ankle_pitch` 0.003261, rolls ~5e-5), matching `action_rate_l2`'s 0.0306/step gives
`0.0306 / 0.008676 = 3.5`. An independent route, 2-5% of episode reward, gives 2.0-5.0, so 3.5
sits inside it and the two derivations agree.

## Consequences

* **Checkpoints are not comparable across the v3 boundary**, and v3 runs log to a `*_v3`
  namespace. This is a plant *and* action-interface change; replays need the constants they
  trained with.
* **`joint_pos_limits` (-10.0) stays unchanged, and is now known to be inert.** It penalises a
  different quantity, measured position crossing the soft band, which momentum can still cause
  under a perfectly clipped command. But it contributes < 5e-7 per step on a trained policy (see
  Weight sizing). It is kept because removing it would add a third simultaneous change to this
  cut, not because it is doing work. A later cleanup should either delete it or re-point it at a
  quantity that actually occurs.
* **`--check-joint-limits` must switch to the pre-clip recomputation.** Reading
  `_processed_actions` after the clip lands would report 0.0 forever, the instrument would die
  at exactly the moment it is needed, which is the same defect shape as the `joint_pos_limits`
  bug this ADR fixes.
* **Attribution is knowingly given up.** The clip and the mass land together with no
  disambiguating arm. V1 measured leg mass against the *lean on hardware*, not sim tracking or
  `act_legs`, so it does **not** license attributing a v3 regression to the clip. Accepted as the
  owner's call; the disambiguating clip-only arm runs only if v3 misses the bar.
* **A1's coefficient is copied from A0 into a reward of different scale.**
  `h1_2_a1/env_cfgs.py:105` records a term at weight 1.0 already swamping tracking, so the risk
  is real. Mitigated by logging the term's share of episode reward, so a swamp appears as a
  number rather than an inference.
* **Sim scores are not expected to improve.** The clip removes an option the policy was
  exploiting. The bar is therefore non-inferiority on the A0 anchor, `err_vx ≤ 0.0853`,
  `act_legs ≤ 0.6130` (0.0831 / 0.5975 plus the 2.6% 64-env noise floor), `fall_rate` 0,
  over-limit rate ≈ 0, n = 2 seeds, followed by a hardware session scoring `trig_joint` against
  the 32% measured on 2026-08-25.
* **A0 lands first, A1 before any comparison.** No A0-v3-vs-A1-v2 number is quoted; that would be
  an RQ2 confound of exactly the kind the comparison-cleanliness rule exists to prevent.
* **Foot sole geometry stays staged.** Ours is still the only H1-2 sim using hand-authored flat
  foot capsules, and it remains the CoP-transfer lever, so it is carried forward rather than
  dropped, just not cut here.
* Three-way parity (training clip == deploy yamls == `h1_2_limits.h`) becomes a tested invariant,
  extending the existing limits fixture in `tests/test_deploy_parity.py`.
