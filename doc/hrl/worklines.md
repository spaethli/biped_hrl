# Worklines plan (2026-07-16, grilled) — parallel tracks + delegation

This chat (the planning chat) owns this file, sequencing, and hand-offs; implementation
runs in delegated chats/agents (see `.claude/skills/delegate`). Decisions from the
2026-07-16 grill session.

## Decisions locked

1. **Sequencing: WL-C verification gates the WL-D batch.** The HIRO-relabel hold-bias
   suspect gets verified FIRST (cheap: code analysis + one relabel-off control run).
   If a fix lands quickly, every WL-D arm trains on the fixed config and is a real
   candidate. If verification drags past ~2 days, WL-D launches on the current config
   anyway, downgraded to lever-validation probes (co-training coupling means an HL fix
   forces retrains of anything trained before it).
2. **WL-D levers**: `ll_cadence_coef` sweep; `hl_cot_coef` refinement 0.15/0.25;
   stand-still gating term (new, targets the unsettled in-place stepping); A0 gait
   terms mirrored into the LL intrinsic ONE AT A TIME (pick by influence, not all at
   once); lower-body default posture / ankle-roll anchor (replay defect: ankles roll
   inward); heel-to-toe roll-over / ankle push-off reward (RESEARCH FIRST, no training
   until a formulation is specced and approved).
3. **Budget: full 10k every arm** (10001 iters, 4096 envs, seed 42, `model_10000`),
   R-round protocol: aggregate bench + `--eval-cmd-vx` holds + `[HOLDDIAG]` + goal
   probe + replay look. Naming per the deviation-tagging convention.
4. **Delegation: `/delegate` skill** (hand-off prompt generator + model picker) plus
   background subagents from the planning chat for small verifiable tasks. The skill
   also nudges the delegating chat to stop doing token-inefficient implementation
   itself.

## Worklines

| WL | What | Owner chat | State |
|---|---|---|---|
| **A** | Planning, sequencing, doc/gate ownership, hand-offs | THIS chat | live |
| **B** | Deploy pipeline: bridge gates G2.x, sim2sim/sim2real battery, hardware prep (plan: `A1a_deploy_plan.md`) | was the planning chat (de facto); hand off to a fresh WL-B chat (prompt below) | G2.0 done; G2.1 rerun pending with the clamp-filter build; next steps below |
| **C** | HL held-command fix: verify the HIRO-relabel hold-bias suspect in `td3.py`, then fix + validate | NEW chat (prompt below) | **CLOSED 2026-07-16.** Relabel FALSIFIED; the blocker was an eval artifact. F2 shipped (baked goal_scale) + WL-B's C++ pin + deployed pair re-exported; tables (e)/(f) re-measured. Gate RELEASED. Residuals handed on: 2 decisions for Liam in the WL-D prompt (arm-weight base config; cadence premise), (b) err_yaw re-run, live FSM check → WL-B G2.0. |
| **D** | Reward/gait lever batch on the cluster (levers above) | NEW chat (prompt below) | **unblocked** (WL-C gate released 2026-07-16); launch on the current config |
| **E** | Training-nominal realism + IsaacGym parity audit (added 2026-07-16): A0 barely lifts feet on the vendor-plant bridge; fric-0.1-nominal control run at kl 0.01 (the ADR-0005 fric stall predates kl 0.01 - untested); mjlab vs IsaacGym training-parameter comparison | WL-E chat | **CLOSED 2026-07-17 (Liam's ruling): nominal stays fric 0, fric-0.1 stays a sanctioned option (constants comment updated); audit arms folded into WL-D (arms 7-9)** |

Gate for WL-C -> WL-D: a verdict on the relabel suspect (confirmed + fix identified,
or falsified + next suspect named). Either way WL-D launches then, on the best-known
config.

## WL-C verdict (2026-07-16): FALSIFIED — and the blocker itself does not exist

Full evidence + numbers → `A1_findings.md` row "WL-C: the held-command 'HL hold
degeneracy' is an EVAL-TOOLING ARTIFACT". Summary:

1. **HIRO relabeling is NOT the cause** (and is nearly a no-op): `relabel_frac` 0.03–0.09
   across iters 1000–10000; instrumented `_relabel` on real keeper windows shows it does
   not shrink `|g|` (0.652→0.621) and preserves the HL's pull (`corr(g_vx, cmd−achieved)`
   0.827→0.798). `|g_stored−g_emp|`=0.51 proves the low frac is a *discriminative* relabel,
   not a stored≈achieved tie.
2. **The real cause is the hold eval itself.** `play.py:297-299` pins the hold command by
   collapsing the twist ranges to a point; `goal_space.py:79-83` derives the HIRO goal
   scale from those same live ranges → scale hits the `1e-3` floor → `V* = s_t + 1e-3·g`
   → the A1 goal channel is **inert in every `--eval-cmd-vx` run**. A0 has no goal space,
   hence "isolated to the hierarchy". Restoring the training scale on the **unchanged**
   keeper: ss@0.5 0.304→**0.044**, ss@1.0 0.882→**0.097** (A0-optB-rs20: 0.055/0.077).
   **A1 holds as well as A0. There is no HL hold degeneracy.**
3. **Consequences**: every A1 hold/`[HOLDDIAG]` number in `A1a_plan.md` (e)/(f) is void;
   rs20's rationale is void; the "sustained-command under-tracking = deploy risk" read is
   void. A0 hold numbers stay valid.

**F2 SHIPPED 2026-07-16 (Liam: "do fix 2, the others are deprecated"). F1/F3 dropped —
F2 subsumes both.**

The goal scale is now a **property of the policy, baked into the checkpoint** (like the
normalizer), instead of being re-derived from whatever command ranges the replay env
carries. 75 lines, 3 files, no training-semantics change:

- `goal_space.py` — `GoalSpace.freeze_scale(t)` / `scale_frozen`; `scale(env)` returns the
  frozen value when set, else derives live (unchanged).
- `hrl_runner.py` — `save()` bakes `goal_scale` into the checkpoint; `load()` pins it;
  `learn()` un-pins (training always derives live, so the scale keeps tracking the twist
  curriculum and resumed runs are unaffected); `goal_scale` added to the ONNX metadata.
- `play.py` — absence shim for pre-2026-07-16 checkpoints (keeper/D1/D2 have no baked
  value): pins the scale the run TRAINED with, reusing `GoalSpace`'s own derivation against
  the task's non-play ranges + the twist curriculum's stages. Same pattern as the
  `hl_obs_vel` / `hl_velocity_goals_only` shims.

Why F1 is dead: the freeze happens at `load()`, *before* play.py's `--eval-cmd-vx` mutation,
so the range collapse can no longer reach the scale — `--eval-cmd-vx` may keep pinning the
command however it likes.
Why F3 is dead **in sim**: the shim/bake supplies yaw 1.0, so the aggregate `err_yaw` bias
is gone for old and new checkpoints alike.

**Tests (all pass):** keeper (pre-bake) through the real `play.py`, `[SHIM] ... pinned to the
training value [0.75, 0.5, 1.0, 1.0, 1.0, 1.0, 0.2]`:

| keeper `model_10000` | before F2 | after F2 |
|---|---|---|
| ss@0.5 (t90) | 0.304 (never) | **0.0436** (0.56 s) |
| ss@1.0 (t90) | 0.882 (never) | **0.0979** (0.76 s) |
| agg err_vx / err_vy | 0.0808 / 0.0774 | 0.0798 / 0.0746 *(invariant — their scales were already right)* |
| agg err_yaw | 0.2344 | **0.1953** *(the halved-yaw-scale bias, removed)* |

The hold numbers reproduce the monkeypatch results (0.0438 / 0.0972) through the shipped
path, and the aggregate moves **only** in yaw — exactly the predicted mechanism, which is a
clean independent check on the diagnosis. **A0 regression byte-clean**: no shim fires (no
goal space), ss@1.0 0.0779 = the documented 0.077. Fresh smoke: checkpoint carries
`goal_scale=[0.75,0.5,1.0,1.0,1.0,1.0,0.2]` (= the measured training value ⇒ training still
derives live), eval takes the baked path (no shim), ONNX metadata carries
`goal_scale=0.750,0.500,1.000,1.000,1.000,1.000,0.200`. Resume re-bakes the live value.
Smokes deleted, `WANDB_MODE=disabled` (no runs created).

### Deploy decision (Liam, 2026-07-16): deploy.yaml stays the last call — which is exactly why the scale must leave it

Liam: *"I want to keep it like this with deploy yaml still being the last call for deployed
policies. Maybe I want to limit a policy which is trained on -0.5 to 1 to -0.25 to 0.5 for
safety reasons."* **Agreed, and F2 does not take that away** — the ONNX `goal_scale` metadata
owns *only* the goal scale, never the command limits. But the two are the SAME yaml field
today, and that is a trap:

`deploy.yaml commands.base_velocity.ranges` currently drives **two unrelated things**:

| consumer | what it does | should follow the operator's safety limit? |
|---|---|---|
| `observations.h:118-120` (`velocity_commands`, joystick = **real robot**) | clamps the commanded twist | **YES** — this is the safety limit Liam wants |
| `goal_space.h:97,119` (`hrl::GoalSpace::scale/center`) | decodes the HL's `g` into `V*` | **NO** — it is trained policy semantics |

So setting `lin_vel_x: [-0.25, 0.5]` for safety would clamp the joystick **and** silently
halve the A1 goal scale (0.75 → `half(-0.25,0.5)` = 0.375), gutting the HL's goal authority
and making the deployed hierarchy under-command velocity — the deploy twin of the sim bug F2
just fixed. **A0 is immune** (no goal space), so the trap is A1-only, exactly as in sim.
Today's `ang_vel_z: [-0.5, 0.5]` already trips it: deployed yaw scale 0.5 vs **1.0 trained**.

Sim-bridge footnote: `keyboard_velocity_commands` (`State_RLBase.cpp:27`) reads the ranges
into a `cfg` variable and **never uses it** (the key map is hardcoded, `w`=1.0). So in the
bridge a range edit today changes *only* the goal scale — pure downside, zero safety benefit.
Don't validate a command limit there expecting it to clamp.

**⇒ WL-B change SHIPPED 2026-07-16 ("do the WL-B change so i can use the safety limit").**
`State_RLHRL::ensure_models_loaded` now reads `goal_scale` from the HL ONNX metadata and pins
it via `hrl::GoalSpace::freeze_scale` **before any range is read**; the ranges are untouched
and keep owning the clamp. **`commands.base_velocity.ranges` is now a pure operator knob in
`delta` mode — the -0.25..0.5 safety limit is free to use.** Log line at FSM entry:
`[HRL] goal scale pinned from ONNX metadata [...]`.

- Files (all robot-local; `deploy/include/isaaclab/` untouched): `include/hrl/goal_space.h`
  (`freeze_scale`/`scale_frozen` + dim check), `src/State_RLHRL.cpp` (`onnx_metadata_floats`,
  same throwaway-session pattern as the existing `onnx_input_dim`, since the shared
  `OrtRunner` keeps its session private).
- **Deployed keeper pair re-exported** with metadata (`0.750,0.500,1.000,1.000,1.000,1.000,
  0.200`); weight-identical — `onnx_parity.py` re-passes at the documented LL 4.2e-05 / HL
  8.5e-06, dims 96/94 unchanged.
- **Tests**: build clean; standalone test of the robot-local path **8/8** — metadata reader
  recovers the trained scale, absent key → empty (fallback), `scale()` returns the trained
  value *ignoring the yaml ranges*, wrong-dim throws, `to_target(delta)` decodes with it.
  (The FSM itself needs DDS + the bridge, so the live check is G2.0/G2.1 in a bridge session.)

**Two carve-outs, both deliberate:**
1. **Legacy ONNX** (pre-2026-07-16, no metadata) → loud warn + old range-derived path, so the
   deploy plan's "old 7-dim checkpoint runs unchanged" back-compat holds. **Narrowing is
   unsafe on those** — re-export first.
2. **`hl_target_mode: absolute`** still decodes against `center()`, which is deliberately NOT
   pinned: the ONNX `goal_center` is in the TRAINING frame (height = pelvis z **1.02**) while
   deploy measures at the imu site (`nominal_root_height` **1.3076**) — adopting it would
   inject a 0.29 m offset. So `center()` stays range-derived and `State_RLHRL` warns that
   ranges must equal the trained ones under `absolute`. `delta` (the A1 default, and what is
   deployed) is unaffected. `goal_center` is still exported for diagnostics/parity only.

**Relabel-off control** `2026-07-16_11-06-48_a1_td3_pose0p5_ar0p02_cadhl_cot0p2_norelabel_s42` (W&B `vg8x3gjf`)
(keeper-identical except `relabeling=none`, incl. the keeper's `resampling_time_range` (3,8)
— today's default (3,20) would have confounded it). Reframed mid-flight from blocker test to
**structure ablation** (the blocker it was built to test does not exist). `model_10000`,
64×600×2; holds with the training scale restored:

| run | agg err_vx | err_vy | err_yaw | fall | act_rate | CoT | stride | ss@0.5 (t90) | ss@1.0 (t90) |
|---|---|---|---|---|---|---|---|---|---|
| keeper (`relabeling=hiro`) | 0.081 | 0.077 | 0.234 | 0 | 1.20 | 0.815 | 0.615 | 0.044 (0.56 s) | 0.097 (0.76 s) |
| control (`relabeling=none`) | 0.065 | 0.070 | 0.150 | 0 | 1.09 | 0.933 | 0.357 | 0.030 (0.35 s) | 0.069 (0.70 s) |
| A0-optB-rs20 | 0.085 | — | — | 0 | — | 0.537 | 0.592 | 0.055 (0.6 s) | 0.077 (0.9 s) |

Honest read: **1 seed each — do NOT call relabel-off "better".** The load-bearing point is
that **both hold fine** (and both ≈ A0), which is what the falsification predicts: the holds
were never broken by relabeling. Tracking/CoT/stride differences sit inside the seed-luck band
this project has been burned by before (`a1_posture_fix`: "tracking 'gain' was seed luck"), and
the control's stride 0.357 sits at the cadence band floor (0.35) vs the keeper's 0.615 — a real
divergence worth a 2nd seed **only if** someone wants to argue relabeling should be dropped from
the locked A1 structure. Given `relabel_frac` ≈ 0.03–0.09, HIRO relabeling is close to a no-op
here; that it is a near-inert component of the "final A1 structure" is itself a reportable
finding, but dropping it is a structure change and Liam's call, not WL-C's.

## WL-E verdict (2026-07-17): friction does NOT stall at kl 0.01; foot drag is training-side, not plant-side

Full evidence: ADR-0005 Amendment 2. Summary of the four deliverables:

1. **Defect quantified** (swing-clearance metric added to `scripts/bridge_replica.py`,
   shared instrument): A0-optB swing apex is plant-INDEPENDENT (walk-0.5 apex
   0.061-0.068 m on vendor / old-harsh / training-nominal override alike, identical
   step counts), and ~35% under the trained 0.10 m clearance target. The vendor plant
   does not cause the drag, it unmasks a marginal clearance the old harsh plant's
   damping visually hid; latency jitter is the vendor-specific part (stand qvel_rms
   0.11→0.31 over 0-4 ms vs flat ~0.06 on harsh). Raising clearance is a reward-shaping
   lever (WL-D territory), not a plant fix.
2. **Friction control run** `a0_v2_optB_fric0p1_kl01_s42` (10k, 4096 envs, s42):
   converges from scratch, lift-off ~1108 (mild ~1.5-2x slowdown, no stall), bench
   err_vx/vy/yaw **0.083/0.108/0.087**, 0 falls, act_rate 0.635, CoT 0.550; holds ss
   0.050@0.5 / 0.089@1.0, t90 <1 s. Comparator note: it trained under the
   post-2026-07-14 resampling (3, 20) s, so its training-side twin is
   `a0_v2_optB_rs20_baseline` (fric 0, same range: err_vx 0.085, CoT 0.537, ss
   0.055/0.077), with the 2026-07-10 optB (3, 8) run (0.089/0.109/0.100, CoT 0.531)
   as the secondary reference; the benchmark protocol is identical for all (play
   pins (3, 8)). At least baseline quality against both. ADR-0005 correction 1's
   stall attribution is FALSIFIED (it was the kl-0.005 + torso/arm-hold confound).
   Bridge replica: passes the full chain at 0/2/4 ms on vendor + stress; gains are
   real but marginal (clearance +4-6 mm, stand qvel at 4 ms 0.254 vs 0.308).
3. **Parameter audit** (mjlab vs unitree_rl_gym vs CorelLab h12_rma, spot-checked):
   all three train friction-free at 50 Hz control with explicit-PD references vs our
   implicit servo; references carry LESS passive dynamics than us (armature 1e-3 / 0
   vs our per-motor 0.002-0.04), identical obs-noise scales, no delay modeling
   anywhere. unitree_rl_gym is 12-DOF legs-only (arms fixed): any "IsaacGym transfers
   better" comparison carries that scope confound. CorelLab's deploy XML = damping 1 /
   armature 0.1 / frictionloss 0.2 = exactly our old harsh bridge plant (provenance of
   those values; a jitter-masking gate, not a fidelity gate). Top-3 transfer-gap
   suspects, each falsifiable: (i) actuator structure, implicit servo vs the explicit
   discrete torque PD every reference trains AND deploys with (test: explicit-PD
   actuation arm in mjlab, compare replica latency sweep); (ii) perturbation DR
   strength, their push 1.5 m/s @5 s + mass -1..+3 kg + friction floor 0.1 vs our
   0.5 / none / 0.3 (test: one A0 arm with their ranges); (iii) obs-noise temporal
   structure, CorelLab low-passes velocity noise at 2 Hz (drift-like) vs our white
   noise (test: correlated-noise arm, judge bridge stand twitch under state_noise).
4. **Recommendation** (rebase ruling = Liam's): adopting frictionloss 0.1 as nominal
   is now known to cost nothing in-sim and buy a small bridge margin, and it closes a
   real train→deploy/hardware gap (vendor sim and real joints have friction; the
   references' fric-free training is a pipeline artifact, not a principle). But it
   re-opens a v1→v2-style rebase (all v2 checkpoints invalid). If the rebase is
   unwanted now, the fallback stays ADR-0005's original: vendor-sim eval + A2 Wide DR
   randomizes over friction. The run's env.yaml carries its own fric-0.1 record.
   **RULED 2026-07-17 (Liam): nominal stays fric 0; fric 0.1 stays a sanctioned
   option for a later rebase window (constants comment updated to cite Amendment 2
   instead of the falsified stall); audit follow-ups added to WL-D as arms 7-9.**

## WL-B next steps (2026-07-16) + hand-off prompt

Near-term job = PIPELINE validation on the current keeper (which is known
not-deploy-ready: ~~the WL-C degeneracy~~ **[corrected 2026-07-16: the WL-C "degeneracy"
was an eval artifact — the keeper tracks held commands ≥ A0 in sim. The remaining
not-deploy-ready items are the known ones: ~10× twitchier than A0 in the bridge, arms
(M2's job), and F3 below]**). Candidate QUALIFICATION waits for the
post-WL-C / M2 checkpoint. Order:

1. Bridge re-run with the clamp-filter build (user session: band -> `i` -> `h`;
   number keys 3/5 held >=30 s, `0` stop, gentle `q`/`e`; `w` last). Expect: survives
   commands, stepping stays unsettled. Analyze `<ts>_hrl.csv` + safety CSV.
   Closes G2.1 (pipeline tier) if 0 falls + no filter engagements.
2. W1 regression still open: one pre-velgoal 7-dim checkpoint through the bridge
   (old yaml block, absent new keys) - confirms back-compat.
3. G2.4 split mode (hold_joint_ids via deploy_real params in sim) - untested.
4. G2.6 cadence modes: pin_period 0.625 vs HL-owned, held vx 0.5.
5. G2.7 stress scene: held 0.5 + steps (blocking at 0.5, advisory at 1.0).
6. E1/E2 prep (hardware, policy NOT in control): checklist for logging onboard
   odometry in low-level mode + the IMU specific-force check - schedulable any time
   robot access exists; REQUIRED before G3. E1 criterion: BASE(pelvis)-frame velocity.
7. Then hold until WL-C/M2 delivers a candidate; re-run the whole battery on it
   (G1.1 mjlab numbers first: record ACHIEVED vx at held commands as the parity
   reference, not just err).

Model for the WL-B chat: **sonnet class** (protocol is written, analyzers exist;
escalate anomalies). Paste into a fresh chat:

```
You own WL-B (deploy pipeline) per doc/hrl/worklines.md. Read first:
doc/hrl/A1a_deploy_plan.md (gates + post-mortem; canonical), .claude/docs/deployment.md
(bridge location /opt/unitree_mujoco, vendor plant, battery tooling, AND the
"goal scale vs the command-range safety limit" section). Work the "WL-B next steps"
list in worklines.md in order.

WL-C/WL-B changes landed 2026-07-16 that you inherit (do NOT re-implement):
- The A1 goal scale is now pinned from the HL ONNX `goal_scale` metadata
  (`State_RLHRL` -> `hrl::GoalSpace::freeze_scale`), so deploy.yaml's
  `commands.base_velocity.ranges` is a pure operator clamp in `delta` mode and Liam may
  narrow it for safety (e.g. lin_vel_x -0.25..0.5). This also silently fixed the old
  yaw-scale defect (was 0.5 vs 1.0 trained).
- The deployed keeper ONNX pair was RE-EXPORTED to carry that metadata
  (weight-identical; onnx_parity re-passes LL 4.2e-05 / HL 8.5e-06).
- **This code has never run in the live FSM** (it was verified by build + a standalone
  test of the robot-local path + parity; DDS/bridge were unavailable). **Fold into G2.0:
  confirm the log line `[HRL] goal scale pinned from ONNX metadata [0.75, 0.5, 1, 1, 1,
  1, 0.2]` appears at FSM entry.** If instead you see the "carries no 'goal_scale'
  metadata" warning, the exported pair is stale — re-export before any range edit.
- Carve-outs: legacy (pre-2026-07-16) ONNX have no metadata -> warn + old range-derived
  path, narrowing unsafe there (relevant to WL-B step 2, the 7-dim back-compat run);
  and `hl_target_mode: absolute` still derives center() from the ranges, so don't narrow
  under that mode (State_RLHRL warns). Deployed config is `delta`.
- Candidate note: post-F2 re-measurement makes D1/D2 (arm-calm) dominate the keeper on
  holds + arms + tracking, losing only CoT/stride. The staged keeper predates the arm
  lever; the candidate choice is Liam's, not yours — flag it, don't switch. The user runs GUI bridge sessions; you
prepare exact per-session key scripts, then analyze the produced logs
(scripts/deploy_gate_analyzer.py on logs/deploy_safety/<ts>_hrl.csv + the safety CSV)
and write gate verdicts into A1a_deploy_plan.md status block (edit in place, house
style). Pre-check candidates headlessly with scripts/bridge_replica.py and
scripts/onnx_parity.py before any GUI session. Do not touch: td3.py/relabeling (WL-C),
reward terms (WL-D), training constants (WL-E owns the nominal question).
```

## Hand-off prompt: WL-C (HL hold-degeneracy verification)

Model: **opus/fable class** (mechanism diagnosis; a silently wrong verdict poisons the
whole batch). Paste into a fresh chat:

```
Verify the HIRO-relabeling hold-bias suspect for the TD3-HL velocity-hold degeneracy
(the A1a held-command blocker). Work spec-first per CLAUDE.md: analysis before any fix.

Read first: doc/hrl/A1_findings.md (row "A1a stage D: arm calm + held-command root
cause", incl. what was already falsified: hold duration/rs20, heading, exact-zero
commands, dither); doc/hrl/A1a_plan.md table (f) + stage D row; the suspect statement:
HIRO relabeling rewrites stored HL goals toward achieved-consistent deltas, so the
critic learns that "hold current velocity" actions score well. Code: src/tasks/velocity/
rl/hrl/td3.py (relabel path ~lines 300-350, _command_period/action storage), high_level.py.

Tasks, in order:
1. Code-level mechanism analysis: trace what g the relabel stores when the robot did
   NOT reach the commanded velocity (the common case under sustained commands). State
   precisely whether/why relabeled transitions systematically favor small deltas.
2. Cheap empirical check on the existing keeper checkpoint (logs/rsl_rl/
   h1_2_velocity_a1_v2/2026-07-09_10-04-50_*/model_10000.pt): e.g. inspect the actor's
   g-output distribution under held-command obs sweeps, or replay-buffer statistics if
   reconstructable. No cluster needed.
3. The decisive control: launch ONE training run, keeper config with
   --agent.relabeling none (everything else identical, seed 42, 10001 iters, 4096
   envs, run name per the deviation convention e.g. a1_..._norelabel_s42; launch per
   CLAUDE.md conda convention, NOT conda run). Compare [HOLDDIAG] + --eval-cmd-vx 0.5/
   1.0 vs the keeper. relabel-off holding = suspect confirmed.
4. Report a verdict (CONFIRMED with fix proposal / FALSIFIED with next suspect) into
   doc/hrl/A1_findings.md stage-D row + a short note in doc/hrl/worklines.md WL-C.
   Do NOT implement a fix without approval.

Do not touch: deploy/ (WL-B owns it), reward terms (WL-D owns them).
```

## Hand-off prompt: WL-D (reward/gait lever batch)

Model: **sonnet class** for implementation+launch (levers are small, protocol is
fixed, pass criteria objective); escalate findings interpretation to the planning
chat. LAUNCH ONLY after the WL-C gate (this file, Decisions #1). Paste into a fresh
chat:

> **TWO DECISIONS FOR LIAM BEFORE WL-D LAUNCHES** (raised by the WL-C re-measurement,
> 2026-07-16 — neither is WL-D's to make, and both change what the batch measures):
>
> 1. **Does the WL-D base config carry the arm weights?** `ll_posture_weights` is still
>    `None` by default — D1/D2's shoulders-16 / elbow+wrist-4 was a *temp rl_cfg edit*,
>    never promoted ("promotion to default pending deploy-line settlement"). Post-F2,
>    D1/D2 are the best runs in the set (best holders 0.030/0.039, best arms, best
>    aggregate tracking) **but cost +10-28% CoT** — and CoT is exactly what WL-D
>    optimizes. So the base config choice moves the batch's headline metric by more than
>    most of its levers will. Pick one: (a) promote 16/4 to default and run the batch on
>    the calmed base, or (b) run on the keeper-like base and treat arms as a later stack.
> 2. **Is the cadence HL still the thing to tune?** Arms 1-2 tune it, but fix0p8 (pinned
>    0.8 s clock) now beats every HL-cadence policy on CoT at equal tracking, un-confounded
>    (see table (e)). If the cadence channel is not earning its keep, arms 1-2 are tuning
>    the wrong component and the batch should re-aim at the energy gap directly.

```
Implement and launch the A1a reward/gait lever batch on the cluster, per
doc/hrl/worklines.md (WL-D) and doc/hrl/A1a_plan.md stage D. Spec-first per CLAUDE.md:
present the arm list + any new reward term implementation for approval BEFORE
launching.

WL-C VERDICT (read worklines.md WL-C first — it changes this brief):
- The verdict selects NO config change. The held-command "blocker" was an EVAL artifact
  (--eval-cmd-vx zeroed the HIRO goal scale); the fix (F2) is eval/deploy-side only, so
  TRAINING SEMANTICS ARE UNCHANGED, nothing needs a retrain, and every existing
  checkpoint stays a valid comparator. Train on the current default config.
- Your hold numbers are only meaningful on a tree that HAS the F2 fix (baked goal_scale
  + the play.py [SHIM] line). Sanity check: a pre-2026-07-16 checkpoint must print
  "[SHIM] no baked goal_scale ... pinned to the training value [0.75, 0.5, 1.0, ...]".
  If it does not, stop — you are measuring the artifact, not the policy.
- A1 tracking/holds are NO LONGER a defect: every A1 run now holds >= A0. The ONLY
  surviving A1-vs-A0 gap is ENERGY (CoT 0.66-1.04 vs 0.537, arm_vel 0.37-0.58 vs 0.14).
  That is your target; do not spend arms on tracking.

Protocol per arm: 10001 iters, 4096 envs, seed 42, cluster launch per
.claude/docs/cluster.md; bench at model_10000 with aggregate + --eval-cmd-vx 0.5/1.0
+ [HOLDDIAG] + goal probe (see CLAUDE.md commands); replay look for gait character.
Baseline comparator: A1a_plan.md tables (f) [RE-MEASURED post-F2, use this] and (e)
[re-measured]. CAUTION: table (b)'s err_yaw column is biased high (pre-F2 yaw scale)
and (d)'s probe numbers carry the frozen-phase asterisk — err_vx/err_vy/CoT/stride in
(b) are fine.

Arms:
1. ll_cadence_coef 1.0 and 2.0 (keeper 0.5) - defect: unsettled stepping/entrainment. other cause could be missing stand still reward
   [WL-C caveat: "unsettled stepping" is a BRIDGE observation whose attributed cause
   (the HL hold degeneracy) is falsified. In sim the re-measured probes are healthy
   (|g|vx 0.10-0.22, sat~0, gait_match 0.94-0.98), so this defect may not exist in sim
   at all. Reproduce it in a sim metric BEFORE burning two 10k arms on it.]
2. hl_cot_coef 0.15 and 0.25 (keeper 0.2) - stride-vs-tracking refinement.
   [WL-C caveat: arms 1-2 tune the cadence HL, whose premise is now in question — the
   pinned-0.8s-clock control (fix0p8) beats every HL-cadence policy on CoT at equal
   tracking (0.663 vs 0.818-1.04), and its old disqualifier was the same artifact. Get
   the planning chat's ruling before spending the batch here.]
3. Stand-still gating term (NEW, LL-intrinsic, gated |cmd|<0.1): reward feet stillness
   / contact hold at stand. Spec the term (reuse existing mdp helpers; flag per the
   A0-comparison cleanliness rule) and get approval before training.
4. A0 gait terms mirrored into the LL intrinsic ONE PER ARM: enumerate A0's gait/
   quality terms from src/tasks/velocity/config/h1_2/env_cfgs.py, propose the 2-3 most
   relevant to the named defects (in-place stepping, foot quality), one term per arm.
5. Ankle-roll / lower-body posture anchor (defect: ankles roll inward in replays):
   extend the ll_posture_weights mechanism (shoulders 16 / elbow+wrist 4 precedent,
   see A1a_plan table f) with a small ankle-roll weight; propose the value.
6. Heel-to-toe roll-over / ankle push-off: RESEARCH ONLY in this batch - survey how
   locomotion papers reward heel-strike -> toe-off progression (foot-pitch phase
   locking, ankle power at terminal stance, contact-point progression), propose 1-2
   concrete reward formulations with the math, and STOP for approval. No training arm.

WL-E transfer-audit arms (added 2026-07-17 per Liam; source: worklines.md WL-E verdict
item 3 + ADR-0005 Amendment 2). These are A0-scoped (train on `Unitree-H1_2-Flat`,
optB config), judged by the deterministic bench PLUS the bridge replica
(`scripts/bridge_replica.py`, both scenes, 0/2/4 ms, clearance + qvel_rms); pass =
bench within noise of the rs20 baseline AND the named replica metric improves. They
touch env/actuator config, not reward terms, so they don't collide with arms 1-6:

7. Explicit-PD actuation arm: replace the implicit position servo with an explicit
   discrete torque PD (kp*(q_des-q) - kd*dq at sim dt, torque-clipped to effort),
   matching what the references train AND the bridge/firmware run. Needs an mjlab
   actuator implementation: SPEC FIRST, approval before training. Falsifies suspect
   (i): expect the vendor-plant latency sensitivity (stand qvel_rms 0.11->0.31 over
   0-4 ms) to flatten toward the stress-plant profile.
8. Perturbation-DR arm: unitree_rl_gym ranges - push max_vel 0.5 -> 1.5, add
   base-mass DR (-1, +3) kg, foot-friction floor 0.3 -> 0.1. One arm, all three
   together (matching the reference recipe as a package). Falsifies suspect (ii):
   expect improved stress-scene + vendor@4ms robustness at unchanged tracking.
9. Correlated obs-noise arm: low-pass the velocity-channel obs noise (~2 Hz cutoff,
   CorelLab `noise_freqs` pattern) instead of white Unoise, same amplitudes.
   Falsifies suspect (iii): expect calmer bridge stand under `hrl.state_noise` /
   real-estimator-like drift at unchanged bench.

Report results as a table into doc/hrl/A1a_plan.md (new stage-D subsection) in the
house style (honest reads, deviations named), and sync doc/hrl/worklines.md WL-D.
Winners = arms that fix their named defect without regressing holds/tracking/falls.

Do not touch: deploy/ (WL-B), td3.py/relabeling (WL-C).
```

## Hand-off prompt: WL-E (training-nominal realism + IsaacGym parity audit)

Model: **opus/fable class** (comparative audit + a verdict that may re-open the
ADR-0005 nominal; the control run itself is mechanical). Independent of WL-C/WL-D
(touches neither td3.py nor reward terms); launch immediately. If its verdict later
favors a friction nominal, that is a REBASE decision for the user (like v1->v2), not an
automatic invalidation of other worklines. Paste into a fresh chat:

```
Investigate why the A0 flat policy barely lifts its feet and walks worse on the NEW
vendor-plant bridge than on the old harsh bridge, and whether the training nominal
should adopt joint friction. Work spec-first per CLAUDE.md; no constants change
without approval.

Read first: docs/adr/0005-model-v2-nominal-fidelity.md (esp. the amendment bisect
table - the frictionloss-0.1 run "stuck @1846" ran at desired_kl 0.005, BEFORE the
kl-0.01 discovery, and was confounded with the torso+arm-hold interaction: the
friction-at-kl-0.01 question is UNTESTED); .claude/docs/deployment.md (bridge plant
section: vendor 0.01/0.1/0.001, why training-nominal failed there);
doc/hrl/A1a_deploy_plan.md (bridge post-mortem); doc/hrl/worklines.md (WL-E).

Known already (do not re-derive): mjlab training HAS obs noise
(enable_corruption=True, src/tasks/velocity/velocity_env_cfg.py ~line 60-127),
push_robot, and foot-friction DR (stripped only in play mode); the ADR-0005 workspace
survey found the IsaacGym references train FRICTION-FREE too (their URDFs carry no
joint friction; 0.1 lives only in their deploy-sim XMLs) and armature disagrees
between references (0.01 XML vs 1e-3 IsaacGym cfg).

Tasks, in order:
1. Quantify the defect: add a swing-foot clearance metric (per-step swing apex height
   of each foot) to scripts/bridge_replica.py output. Run A0 optB on scene.xml
   (vendor) vs scene_stress.xml (old harsh) vs a training-nominal plant override, at
   2 ms delay. Deliverable: clearance + qvel_rms table answering whether foot drag is
   caused by the vendor plant or was always marginal and merely masked by the old
   plant's damping.
2. The kl control run: A0 optB config + frictionloss 0.1 training nominal (all
   joints, via h1_2_constants.py on a branch/temporary edit) at desired_kl 0.01,
   10001 iters, 4096 envs, seed 42, run name a0_v2_optB_fric0p1_kl01_s42 (launch per
   CLAUDE.md conda convention; expect slow lift-off, judge no earlier than ~2x
   expected lift-off per the ADR). Success: converges to <= optB tracking (err_vx
   ~0.09, 0 falls). Then benchmark + replica on the vendor bridge: does foot
   clearance/gait improve vs the fric-0 optB?
3. Parameter audit table: ours (mjlab) vs unitree_rl_gym (IsaacGym baseline in the
   workspace) vs the CorelLab h12 stack: sim dt + decimation, actuator/PD
   implementation (implicit servo vs explicit torque PD, where kp/kd live), armature
   source+values, joint friction (train AND deploy sim), obs-noise ranges, push/mass/
   friction DR ranges, action scale+clip, control delays. Deliverable: the table +
   the 3 differences most likely to explain "mjlab transfers worse than IsaacGym"
   claims, each with a falsifiable follow-up.
4. Verdict: should the training nominal adopt friction (re-opens ADR-0005 correction
   1)? Evidence-based recommendation ONLY - the rebase decision is the user's. Report
   into ADR-0005 (amendment note), doc/hrl/worklines.md WL-E, and a findings-ledger
   row.

Do not touch: td3.py/relabeling (WL-C), reward terms (WL-D), deploy C++ and /opt
scene XMLs (WL-B). The bridge_replica.py metric addition is allowed (shared
instrument).
```

## Standing rule for the planning chat (WL-A)

When this chat catches itself deep in implementation (multi-file edits, debugging
loops, running trainings), stop and emit a hand-off instead: that work is usually too
easy for the expensive model and burns the planning context. Small, objectively
verifiable one-offs may go to background subagents (sonnet) directly.
