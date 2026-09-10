# Revert Model v3: the action-clip penalty costs lateral ankle authority

**Status:** accepted (2026-08-28). **Supersedes ADR-0008**, which defined Model v3 as action
clip + L1 excess penalty + leg mass. Its three **training-side** changes are reverted and the
model returns to **v2**; the **deploy yaml clip is KEPT** (see Decision). ADR-0006 is no longer superseded: its foot-sole geometry proposal is live again as a
staged, uncut option.

ADR-0008 closed a real training/deploy parity gap. Measured on the robot and reproduced in sim,
closing it cost **2.1-2.4x of the commanded ankle roll range**, the joint that rejects lateral
disturbance, and bought nothing, because the firmware already truncated the commands the clip
was added to truncate.

## Context

ADR-0008's premise was correct: `joint_pos_limits` prices the MEASURED joint position, MuJoCo's
`qpos` clamp already prevents that, so the COMMANDED target was unpriced and A0 learned to
command its ankles up to 1.4 rad past the mechanical stops on 23-25% of hardware ticks.

Three A0 sessions on 2026-08-27 compared v2 against v3 on the robot. All three ran with the clip
active (the `clip: null` override was appended below the bounds table, and yaml-cpp resolves a
duplicate key to the FIRST occurrence, so it was a silent no-op; verified against the real file
and with a two-non-null-table control). That accident made the comparison cleaner than intended:
one deploy config, three sessions, two of them replicates.

The operator verdict was that v2 stood better, resisted pushes better, and walked better, and
that v3 was too unstable to walk at all.

## Evidence

Commanded ankle range, `p1..p99` of the post-clip target. Sim is 600 steps x 64 envs x 2 seeds
(`play.py --check-joint-limits`); hardware is the flight recorder's `raw_q`, which is the same
quantity (`State_RLBase.cpp:221` logs `action`; the safety clamp at `:164` writes a local).

| arm | plant | trained w/ clip+penalty | roll span L / R | headroom roll L / R | pinned (pitch L) |
|---|---|---|---|---|---|
| v2 anchor | v2 | no | **0.345 / 0.331** | 0.021 / 0.020 | 2.37% |
| cliponly | v2 | yes | **0.163 / 0.139** | 0.171 / 0.178 | 0.03% |
| v3 | v3 | yes | **0.154 / 0.141** | 0.165 / 0.187 | 0.05% |

| hardware session | policy | roll span L / R | pinned |
|---|---|---|---|
| 13:42:55 (308 s) | v2 | 0.413 / 0.505 | 14.07% |
| 13:56:37 (198 s) | v2 | 0.436 / 0.373 | 21.08% |
| 13:51:52 (100 s) | v3 | 0.192 / 0.223 | 0.46% |

Sim ratio v3/v2 **0.45 / 0.43**; hardware ratio **0.45 / 0.51**. Per-seed spread is ~2%, and both
seeds clear the pre-registered `< 0.75x` rule on both ankles.

**The two contrasts decompose the cause.** `cliponly` and v2 share a plant and differ only in
whether training carried the clip and penalty: roll span differs **2.1-2.4x**. `cliponly` and v3
share the clip and penalty and differ only in leg mass: roll span differs **6% and 2%**, inside
the per-seed spread. The authority loss is entirely training-side. **The leg mass is inert.**

**`headroom` identifies which half did it.** v2 sits **0.19 sigma** from the ankle-roll bound
(sigma = `distribution.std_param` 0.428 x the 0.25 leg action scale = 0.107 rad) and pins it
0.5-2.4% of steps. Both clipped arms sit **~1.8 sigma** away and never touch it. A hard clip
cannot produce that: past the bound every action has the same outcome, so there is no gradient
pushing the policy inward. Only the `action_clip` penalty can, and it does, through sampling: a
mean command near the bound puts probability mass past it and pays for it. The predicted leak was
one sigma; the measured retreat is 1.8, so the penalty does not merely leak, it optimises the
mean out of the region.

**The clip's deploy-side benefit was approximately zero.** `State_RLBase.cpp:160-165` clamps every
commanded target to `h1_2_joint_limits` unconditionally and always has, so the robot already
truncated those over-limit commands before ADR-0008. The yaml clip moved the same truncation
earlier in the pipeline; the two coincide at `alpha = 0`, which is normal operation.

**The weight was never a stable quantity.** ADR-0008 sized `-3.5` so `action_clip` would match
`action_rate_l2`'s contribution at a measured mean excess of 0.008676 rad/step. Re-measured on the
same policy under the clip, mean excess is 0.012687 and the term contributes 0.0444 against
`action_rate_l2`'s 0.0306, i.e. **1.45x** its intended strength. The anchor moves 46% between
rollouts.

## Decision

Revert the three **training-side** ADR-0008 changes:

1. the training clip on the processed action (`config/h1_2/env_cfgs.py`, `H1_2_ACTION_CLIP`),
2. the `action_clip` L1 excess penalty and its A1 mirror `ll_action_clip_coef`,
3. the leg-mass correction (`h1_2.xml` back to 66.984 kg total).

**Keep the clip table in all eight deploy yamls.** It is not what cost the authority and it does
no harm: v2 ran pinned by it **14-21% of steps** on 2026-08-27 and was the best-performing
configuration of the day. It also clips *before* the `State_RLBase` hold blend rather than after,
so a safety-hold ramp starts from an already-legal target; at `alpha = 0` the two are identical,
so it is neutral-to-slightly-better and never worse. With training carrying no clip it is a
deploy-side **backstop**, not a parity device, and `test_deploy_yaml_clip_matches_the_limit_header`
keeps it pinned to `h1_2_limits.h` across all eight configs.

The `*_v3` log namespaces revert to `h1_2_velocity` / `h1_2_velocity_a1_v2`. **v2 checkpoints are
valid again and are the current model**; the three v3-era A0 runs are kept as the evidence for
this ADR, not as candidates.

**`ang_vel_z` stays at `[-1.0, 1.0]` in all eight deploy yamls, matching training.** This rode
in on the ADR-0008 commit unmentioned, and this ADR first reverted it as unexplained drift. That
was wrong: it is a deliberate operator change (2026-08-27). At the older `[-0.5, 0.5]` the robot
barely responded to small stick deflections, because the deploy range scales the command the
policy sees; matching training restored turning authority. It supersedes the "deploy narrow for a
tame joystick" convention, which was a preference, not a safety requirement. Safe for current
exports because `goal_scale` is baked into the HL ONNX metadata, so these ranges no longer feed
the A1 scale derivation. The rationale is now recorded in the yamls themselves, which is what was
missing when it read as drift.

## Consequences

* **Comparability is preserved.** v3 would have invalidated every v2 checkpoint across A0 and A1
  and forced retraining for A2-A4 later, to buy a fidelity correction that ADR-0006's V1 already
  falsified as a lean fix and that these runs show to be inert on every metric measured.
* **The commanded-target defect is reopened, knowingly.** A0 again commands past its ankle stops.
  The measured consequence of that on hardware is nil, because the firmware truncates, and the
  policy that does it is the one that stands and walks best. It is a fidelity gap, not a fault.
* **`joint_pos_limits` (-10.0) remains inert** at < 5e-7 per step on a trained policy. That
  finding survives the revert and is worth acting on separately; this ADR does not.
* **`clip without penalty` is untested and stays open.** It would give the parity ADR-0008 wanted
  without the retreat, at the cost of one training arm. Filed as an option, not scheduled.
* **The bench cannot see this class of defect.** Every pre-registered metric improved under v3:
  `err_vx` 0.0903 vs 0.0941, `act_legs` 0.567 vs 0.600, `jacc_legs_p95` 62.1 vs 78.6, `fall_rate`
  0 for both. In the standing condition all three arms use 0.026-0.069 rad of ankle roll and none
  approaches a bound. **A reserve is invisible to any metric collected in the regime where the
  reserve is not needed.**

  ⚠ **Correction 2026-09-07 — the stated REASON was wrong, and it pointed at a no-op fix.**
  This bullet originally read "because sim standing applies no lateral disturbance", and
  prescribed "adding a disturbance term to the benchmark" as the prerequisite. **There already
  is one.** `push_robot` (`events`, `push_by_setting_velocity`) fires every **5-6 s** in
  training with linear `y` in **±0.5 m/s** and roll rate in **±0.52 rad/s**, and `play.py`
  removes only `base_mass` from the event set, so the **bench applies it too**. Adding a
  disturbance term would therefore have changed nothing.

  The conclusion (the bench cannot see this defect) still stands; only the mechanism is now
  open. What is actually missing is a metric that scores the **RESPONSE** to the push that
  already happens — recovery time, peak ankle-roll demand, lateral excursion after an event —
  rather than pooled statistics over a rollout in which a push is a small minority of steps.
  Candidate explanations for why the loss stayed invisible, none of them yet tested: the event
  sets base VELOCITY (an impulse the policy answers with one recovery step) rather than applying
  a sustained lateral load; the ±0.5 m/s magnitude may sit below what the hardware delivered; and
  pooled p1/p99 spans weight the disturbed steps only by their small duty cycle. **Do not cite
  "sim applies no lateral disturbance" — it is false.**
* **The step from "less range" to "worse push rejection" is an inference.** The range is measured
  in both sim and hardware and they agree to a few percent; the causal link to push rejection
  rests on the operator's hardware observation, at n=1 session for v3 and a single training seed
  per arm.
* **Kept from the episode:** `play.py --check-joint-limits` (now reporting per-joint p1/p50/p99,
  span, headroom and pinned fraction, and validated against hardware), `scripts/score_joint_hold.py`,
  and `hold_joint_ids` in the flight-recorder metadata. None depends on the clip.
* **yaml duplicate keys are now guarded.** `test_deploy_yaml_clip_matches_the_limit_header` loads
  the eight configs through a `SafeLoader` subclass that refuses duplicate mapping keys, so the
  2026-08-27 failure mode cannot recur silently. This matters precisely because a plain
  `yaml.safe_load` check would keep the LAST key while the robot keeps the FIRST, i.e. it would
  pass while the two disagree. Both defects are in `check_test_sensitivity.py` (53/53).
