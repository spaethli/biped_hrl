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
| **B** | Deploy pipeline: bridge gates G2.x, sim2sim/sim2real battery, hardware prep (plan: `A1a_deploy_plan.md`) | deploy chat + the user's bridge sessions | G2.0 done; G2.1 rerun pending with the clamp-filter build |
| **C** | HL held-command fix: verify the HIRO-relabel hold-bias suspect in `td3.py`, then fix + validate | NEW chat (prompt below) | next action, gates WL-D |
| **D** | Reward/gait lever batch on the cluster (levers above) | NEW chat (prompt below) | blocked on WL-C gate (or 2-day timeout) |
| **E** | Training-nominal realism + IsaacGym parity audit (added 2026-07-16): A0 barely lifts feet on the vendor-plant bridge; fric-0.1-nominal control run at kl 0.01 (the ADR-0005 fric stall predates kl 0.01 - untested); mjlab vs IsaacGym training-parameter comparison | NEW chat (prompt below) | independent of C/D; launch now |

Gate for WL-C -> WL-D: a verdict on the relabel suspect (confirmed + fix identified,
or falsified + next suspect named). Either way WL-D launches then, on the best-known
config.

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

```
Implement and launch the A1a reward/gait lever batch on the cluster, per
doc/hrl/worklines.md (WL-D) and doc/hrl/A1a_plan.md stage D. Spec-first per CLAUDE.md:
present the arm list + any new reward term implementation for approval BEFORE
launching. Use the config the WL-C verdict selects (check worklines.md WL-C note).

Protocol per arm: 10001 iters, 4096 envs, seed 42, cluster launch per
.claude/docs/cluster.md; bench at model_10000 with aggregate + --eval-cmd-vx 0.5/1.0
+ [HOLDDIAG] + goal probe (see CLAUDE.md commands); replay look for gait character.
Baseline comparator: the current keeper row in A1a_plan.md tables (b)/(d)/(f).

Arms:
1. ll_cadence_coef 1.0 and 2.0 (keeper 0.5) - defect: unsettled stepping/entrainment.
2. hl_cot_coef 0.15 and 0.25 (keeper 0.2) - stride-vs-tracking refinement.
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
