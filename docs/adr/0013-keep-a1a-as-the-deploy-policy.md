# Keep A1a (no payload DR) as the deploy policy, against the simulation ranking

**Status:** accepted (2026-09-14, owner and supervisor). Fixes the deploy policy and the
thesis's headline hierarchical arm. The evidence for the *reason* was incomplete; the
2026-09-18 amendment at the end tests it and **narrows it to the seed-123 DR arms**.

The payload-DR arms are not adopted. **A1a**
(`2026-09-02_12-57-09_a1a_fullmirror_cot5_standing12_s42/model_10000.pt`, `cot5` + `rse 0.12`,
no payload DR) remains the deploy policy, on the operator's hardware judgement that the DR
arms never came fully to rest at zero command, and that `A0_DR` under the backpack did not
walk better than A1a. The DR arms are reported as a measured negative, not deleted.

This is recorded as a decision precisely because **no simulation metric supports it, and the
weak ones rank the arms the other way.**

## Context

Base cell, 0 kg, pooling `data/2026-09-09-wp3-baseline-arms/{all,nodr,cotcap}_bench.json`:

| arm | `ub_arm_vel` | `ub_pose_dev` | `act_legs` | `err_vx` |
|---|---|---|---|---|
| A0_DR (flat) | 0.120 / 0.124 | 0.00034 | 0.578 / 0.580 | 0.086 / 0.086 |
| **A1a** | 0.307 / 0.316 | 0.0083 / 0.0114 | 0.618 / 0.627 | 0.127 / 0.126 |
| A1a_DR | 0.322 / 0.331 | 0.0110 / 0.0103 | 0.588 / 0.614 | 0.117 / 0.134 |
| A1a_DR_cotcap | 0.320 / 0.305 | 0.0106 / 0.0109 | 0.582 / 0.582 | 0.104 / 0.112 |

A1a sits inside the DR arms' range on both upper-body metrics, and is the **worst** of the
four hierarchical arms on `act_legs` and `err_vx`. The large upper-body gap (2.7x
`ub_arm_vel`, 30x `ub_pose_dev`) is hierarchy-versus-flat, and A1a is on the hierarchical
side of it. The 2026-09-14 hardware metrics do not separate them either: both DR Runs land
inside A1a's own 2.4x within-session band, in its better half.

This is the second instance of the pattern in ADR-0009, where Model v3 improved every
pre-registered simulation metric and then failed on the robot for a reason the benchmark
structurally could not see. Once is an accident; twice is a property of the benchmark, and it
is a thesis result in its own right.

**Do not defend this choice with simulation numbers.** They argue against it.

## Open: the mechanism is a hypothesis, not yet evidence

A candidate separator emerged after the decision was taken and is **not yet confirmed**. With
`pin_period: 0.0` the HL owned its gait clock on hardware for the first time, and its
commanded stride period at zero command splits the arms across nearly the whole action range
(`cadence_period_range` is `[0.35, 1.0]`):

| policy | commanded period, standing |
|---|---|
| A1a (5 Runs) | 0.996 – 0.999, i.e. pinned at the ceiling |
| A1a_DR_cotcap_s42 (1 Run) | 0.801 |
| A1a_DR_s123 (2 Runs) | 0.400 / 0.433 |

A 0.40 s clock at standstill is more than twice A1a's rate, and the low level entrains to it
(0.96-0.98 in simulation). Continuous small motion at a standstill is what the operator
described. `act_legs_rad` averages *amplitude* and is blind to a *frequency* difference,
which would explain why it saw nothing.

This was n=2 and n=1 on the DR side, from one session, and found after the conclusion it
supports; it was promoted to a reported metric so it could be tested. **Tested 2026-09-18: it
holds for the seed-123 arms only (amendment below).** Settling time (base speed from motion
capture, arm velocity from the encoders, since only the main body is captured) was the
pre-registered adjudicator.

## Measured 2026-09-14: cadence separates categorically, settling time only suggests

`scripts/analyze_cadence_settling.py`, all 11 Runs. Commanded cadence while standing, with
the fraction of policy steps sitting at the `[0.35, 1.0]` ceiling:

| policy | n Runs | mean period (s) | at ceiling |
|---|---|---|---|
| A1a | 6 | 0.996 – 0.999 | **97.7 – 99.7%** |
| A1a_DR_cotcap_s42 | 1 | 0.801 | 0.2% |
| A1a_DR_s123 | 2 | 0.400 / 0.433 | 0.1 – 0.4% |

Zero overlap, and A1a is not merely slower but **saturated**: it asks for the longest stride
the action space allows, essentially every step it stands. That is a categorical difference
in what the HL wants at rest, and it is the only quantity found so far that separates these
policies at all.

Arm-velocity settling time, median seconds to stay under threshold for 2 s, over standing
segments at least 10 s long (shorter segments measure the operator's stick, not the policy,
and including them censored every Run):

| policy | th=0.05 | th=0.10 | th=0.15 | th=0.20 |
|---|---|---|---|---|
| A0_DR (flat) | 2.2 – 5.7 | **0.6 – 1.0** | 0.4 – 0.9 | 0.2 – 0.9 |
| A1a | 2.0 – 9.9 | 1.6 – 5.6 | 1.4 – 5.5 | 1.1 – 5.5 |
| A1a_DR (both arms) | 6.3 – 13.3 | 5.4 – 7.8 | 4.1 – 7.2 | 2.9 – 4.8 |

Two readings, and only the first is solid. **The flat policy settles far faster than any
hierarchical arm** at every threshold, which is the hardware form of the 2.7x `ub_arm_vel`
simulation gap and is not in dispute. **The DR-versus-A1a ordering is the operator's, but it
is not threshold-robust**: all three DR Runs sit at or above A1a's worst at th=0.05 and 0.10,
the ranges touch at th=0.15, and at th=0.20 the DR values fall back *inside* A1a's range. The
dose-response against cadence is also not clean, since `cotcap` at 0.801 s settles slower than
A1a at 0.999 s.

## Motion-capture ground truth, 2026-09-15: it is the residual, not the delay

Three Runs have usable Vicon ground truth (one A1a, two A1a_DR_s123; run 6's capture is
corrupt and run 3's is weak). Base speed, pelvis-referenced, lever `r = (0, 0, 0.32) m`:

| Run | policy | residual speed at rest | settle to <0.05 m/s | to <0.10 m/s |
|---|---|---|---|---|
| run 5 | A1a | **0.0018 m/s** | 5.5 s | 5.0 s |
| run 3 | A1a_DR_s123 | 0.0042 m/s | 5.0 s | 4.7 s |
| run 4 | A1a_DR_s123 | 0.0089 m/s | 7.6 s | 6.7 s |

**Settling TIME does not separate them** — A1a sits between the two DR Runs. **The residual
speed at rest does**: A1a comes to rest 2.3x to 5x quieter. The operator's description was
"still wobbling a bit, not motionless", which is a statement about the floor, not the delay,
and the measurement matches the description rather than the hypothesis that preceded it.

Both conclusions survive a lever sweep over `r_z` ∈ {0.25, 0.32, 0.40} m: the floors move by
at most 0.0004 m/s and the ordering never changes, so the result does not rest on knowing the
marker-cluster offset precisely. The 0.03 m/s threshold is excluded as lever-limited (the
`ω × r` term alone reaches 0.022-0.030 m/s at the 95th percentile of standing roll/pitch).

⚠ n = 1 for A1a at the time (superseded by the amendment below).

So the decision was **supported but not established**: three DR Runs from two policies in one
session, a post-hoc metric, and an ordering that weakened as the threshold loosened. The seed
band was named as what would settle it; it did, below.

## Amendment 2026-09-18: the seed band, and what it does to the reason

Run: 2026-09-16, two seeds x four variants, 21 Runs all with motion capture, plus the 09-14 Runs
as a between-session replicate (0 kg only; the broom Run and the 7.5 kg Runs excluded).
**Outcome: the result does not strengthen as a DR effect. It holds for seed 123 only.**

| policy (Runs) | commanded period at rest, s | arm settling to 0.1 rad/s, s | residual-speed floor, m/s |
|---|---|---|---|
| A1a s42 (8; both sessions) | 0.996 - 1.000 | 2.1 - 6.0 | 0.0018 - 0.0085 (5 with a floor) |
| A1a s123 (3) | 0.87 - 0.98 | 2.7 - 5.8 | 0.0011 - 0.0025 |
| A1a_DR s42 (2) | 0.984 - 0.988 | 3.1 - 3.2 | 0.0019 - 0.0035 |
| A1a_DR s123 (4; both sessions) | **0.39 - 0.43** | **5.4 - 15.0** | 0.0039 - 0.0089 |
| A1a_DR_cotcap s42 (3) | 0.68 - 0.80 | 3.0 - 7.2 | 0.0020 - 0.0048 |
| A1a_DR_cotcap s123 (2) | 0.85 | **16 - 22** | **0.0127 - 0.0504** |
| A0, flat (5) | no HL | 0.6 - 1.1 | 0.0005 - 0.0024 |

- The categorical cadence separation of 2026-09-14 (zero overlap) was a seed-123 result; the
  seed-42 DR arm sits at A1a's ceiling. The 2.3-5x floor gap was n=1 for A1a; A1a s42 alone now
  spans 4.7x, so only cotcap s123 separates categorically and H-mem s123 is about 1.4-2x with
  overlapping ranges. Settling time again does not separate (H-mem s42 equals A1a).
- The no-DR recipe is not clean at seed 123 either: its period is pinned only 9-12% of steps
  and `A1a_s123` fell twice in the bridge, on the strafe deceleration.
- **Consequence.** Keeping A1a is defensible for a payload of 7.5 kg or less, but the recorded
  reason holds for the seed-123 DR arms only. The keeper is a seed-42 draw chosen from
  single-seed hardware data. No hierarchical DR arm was loaded on the robot; in sim the DR
  benefit is the standing hold at 10-15 kg (15 kg drift: cotcap 0.008-0.011 m/s and 7 W, no-DR
  0.082-0.128 m/s and 52-61 W), untested on hardware. The adaptive arm never flew and shows no
  consistent gain over plain cotcap in sim.
- **Not established:** any DR effect on walking (not command-matched, 29-63 s per cell, and the
  same checkpoint differs 20% in walking cot between sessions).
- **Would settle it:** cotcap/H-mem s42 against A1a s42 under 10-15 kg standing holds, a third
  seed per recipe, a command-matched walking protocol.

Tables, method, the correction of a mis-quoted "residual speed" (regime RMS versus this floor)
and the meeting record: journal `a1a-deploy-journal.md`, 2026-09-19.
