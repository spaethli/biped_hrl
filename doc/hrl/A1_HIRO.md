# A1 — HIRO (hybrid PPO + TD3) — design & status

> **Status (2026-06-16): tracking wall SOLVED.** `absolute`-target HL + `tracking`-weighted
> HL reward reaches deterministic benchmark **err_vx 0.098 / err_vy 0.091 / err_yaw 0.167,
> 0 falls — A0-level** on vx/vy (A0 0.09/0.11/0.10), 4.3× better than the prior best A1.
> Remaining = polish (yaw, smoothness) + writeup ablations. See "Current results" + the
> chronological journey in the A1 findings ledger (research KB).

Shared machinery (co-train loop, goal space, warm-start, reward decomp, TD3 internals,
benchmark/probe tools, checkpoint/ONNX, gotchas) is in **`.claude/docs/hrl-infra.md`** —
this doc only covers what is A1-specific.

## What A1 is
Architecture A1 of the thesis: the **structural hierarchy**. A hybrid HIRO design —
**on-policy PPO low level** (= the A0 learner, so RQ2 isolates the hierarchy) + **off-policy
TD3 high level** with HIRO goal-relabeling. The HL fires every `c` steps and hands the LL a
velocity(+pose) goal; the LL never sees the command. Relabeling is a runtime toggle (clean
ablation). Stays in `unitree_rl_mjlab` + `mjlab` + `rsl_rl`; no Isaac Lab.

## Why two levels (honest assessment, 2026-06-18)
A1 ≈ A0 on flat in-sim tracking is **expected, not a failure** — a hierarchy can't beat a flat
policy at one stationary task (the goal bottleneck only constrains it). As measured A1 has **no
in-sim upside and real costs** (yaw 0.17 vs 0.10, complexity, load-bearing A0 warm-start, + a
runtime base-velocity estimate A0 doesn't need — see deploy note), so "hierarchy improves
locomotion" is **not** supported. **Correction (2026-06-23):** the prior "jerkier A1" cost
(act_rate 1.68 vs 0.66) was a **reward artifact** — A0 is penalized for jerk, A1's LL never is
(LL = intrinsic reward only). Under a matched penalty-free reward A1-from-polished is ~10×
*smoother* than A0 (Track F true-lean → the hierarchy-benefit roadmap (research KB)), so smoothness is not a
genuine A1 cost; the deploy-relevant concern is only the shaped-reward gait. The payoff
is elsewhere: (1) **substrate for A2/A3** — A2's A-RMA adaptation attaches to the HL/LL split,
where the sim2real story (M3, the primary metric) comes from; (2) **clean RQ2 control** — A1's
LL = A0 PPO, so A1≈A0 proves the hierarchy is performance-neutral → any A2 gain is attributable
to adaptation, not confounded; (3) **interface** — goal as a safety-clamp / planner- or
MPC-driven (A3/A4) / inspection surface, reusing the LL without retraining. **Decision:** don't
polish A1 for in-sim tracking; fix smoothness only (transfer liability that would confound A2),
then test where a hierarchy *could* win (OOD/robustness below) and move to A2.

## The two levers that solved the wall
Each fixes one probe-isolated failure (see the A1 findings ledger (research KB) for the diagnosis chain):

1. **`hl_target_mode=absolute`** (`HrlRunnerCfg`, default `delta`; CLI
   `--agent.hl-target-mode absolute`). The HL emits a state-independent
   `V* = center + scale·g` (static command→goal map) instead of the relative
   `V* = s_t + scale·g` (which needs a state-dependent law the HL never learned →
   `|g|→1` saturation collapse). Fixes **LL reachability/saturation**: `|g|` 0.84→0.16,
   LL reach err 0.61→0.11. The map lives in `GoalSpace.center/to_target/to_g` (one place);
   relabel uses `ref = center` (absolute) | `s_t` (delta). LL unchanged (still sees delta
   `V*−s_i`, same reward).
2. **`hl_reward_mode=tracking`** (`HrlRunnerCfg`, default `task`; CLI
   `--agent.hl-reward-mode tracking`). The HL accumulates **velocity command-tracking only**
   (`mdp.track_linear_velocity + track_angular_velocity`) with LL-execution penalties
   removed (the LL's concern). Under the default `task` reward the HL objective is
   penalty-dominated → value-max policy is `g≈0` ("ask for neutral velocity", refuses
   forward). Fixes the **HL command→goal collapse**: HL goal err vx 0.71→0.11,
   forward-avoidance gone. The **env reward is unchanged** (no RQ2 confound); only the
   HL-internal objective changes. Safe with `absolute` (V* bounded to command range).

**Attribution (A/B, 2026-06-17):** `tracking` is the essential lever (alone: err_vx 0.14);
`absolute` is a refinement on top (→0.098, smoother). With `task` reward neither map tracks.

## HL learners (`hl_algorithm`)
The switch is the **HL learner**, not just a flag — relabeling needs a replay buffer, so it
is structural. `--agent.hl-algorithm {oracle,ppo,td3}`.

- **`oracle`** (M2) — emits the absolute command as `V*` (no learning). Validation harness:
  proved the warm-started LL is a competent goal-conditioned tracker (0.05 m/s, no
  directional bias). Use as the LL upper-bound control.
- **`ppo`** (M3) — a 2nd `rsl_rl.PPO` at the HL timescale (`HighLevelPpo`). **Naive on-policy
  HL fails to track** (credit-assignment thinness: ~3 HL transitions/env/iter; tracking term
  swamped by penalties). HL actor obs `policy ++ command`; critic obs `critic` (already
  contains command — do NOT append). Linear `scale` map, NOT tanh (rsl_rl's
  `GaussianDistribution` applies no Jacobian correction → a tanh squash would be a silent
  bug). Std cap `(1e-3,1.0)` available (fixes blowup). Baseline only; kept for the writeup.
- **`td3`** (M4/M5, **the designed HL**) — self-contained off-policy TD3 (`HighLevelTd3` +
  `HLReplayBuffer`). Internals/tuning in `hrl-infra.md`. `relabeling {none,hiro}` sub-toggle
  (HIRO off-policy correction, inline). `relabel=none` is the clean control that attributes
  any further gain to relabeling alone.

## Goal space (A1)
Default **7-dim `(velocity, orientation, height)`** — confirmed decision (orient+height are
stabilizing regularizers: better tracking esp. yaw, lower std; velocity-only 3-dim is the
documented clean negative result). `goal_components` in `config/h1_2_a1/rl_cfg.py` is the
single source of truth; `goal_dim` derived. Goal weights: velocity weight 3.
See `hrl-infra.md` for the `GoalSpace` map and `scale`/`center` mechanics.

## Config & launch
- Defaults (`config/h1_2_a1/rl_cfg.py`, `HrlRunnerCfg`): `hl_algorithm=oracle`(*),
  `c=8`, `gamma_hi` derived `0.99**c`, goal 7-dim, `entropy_coef=0.005`,
  `hl_target_mode=delta`(*), `hl_reward_mode=task`(*).
  (*) the SOLVED config overrides these — see below.
- **Solved-config launch** (the A0-level result):
  ```bash
  conda activate unitree_mjlab_h1_2_rl
  python scripts/train.py Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096 \
    --agent.max-iterations 5001 --agent.hl-algorithm td3 --agent.relabeling hiro \
    --agent.hl-target-mode absolute --agent.hl-reward-mode tracking \
    --agent.warm-start-path logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt \
    --agent.run-name <name>
  ```
- TD3 knobs override as `--agent.hl-td3.<field>`. Launch via the env + `python` directly,
  not `conda run`. `max-iterations` is N+1 (final ckpt named after the last index).

## Current results (deterministic benchmark, 64 envs × 600 steps × 2 seeds; c=8, warm-start)
| run | err_vx | err_vy | err_yaw | act_rate | \|g\|vx (sat) | HL err vx | LL reach vx | fwd/bwd end vx |
|---|---|---|---|---|---|---|---|---|
| delta baseline (relabel) | 0.42 | 0.35 | 0.64 | 1.41 | 0.84 (58%) | 0.51 | 0.61 | — |
| absolute, task reward | 0.65 | 0.45 | 0.28 | 2.40 | 0.16 (2%) | 0.71 | 0.11 | 0.92 / 0.19 |
| delta + tracking | 0.140 | 0.111 | 0.199 | 2.46 | 0.36 (2%) | 0.20 | 0.22 | 0.110 / 0.177 |
| **absolute + tracking** | **0.098** | **0.091** | **0.167** | 1.68 | 0.16 (0.6%) | **0.11** | 0.10 | **0.096 / 0.083** |
| A0 reference | 0.09 | 0.11 | 0.10 | 0.66 | — | — | — | — |

Run `a1_td3_absolute_hltrack_5k` (td3+relabel, c=8, warm-start, absolute+tracking, range
(-0.5,1.0), `model_5000`). The absolute→tracking A/B (only `hl_reward_mode` differs) is the
decisive step: HL goal err vx 0.71→0.11, forward-avoidance gone, `|g|` stays de-saturated,
LL reach ~0.10. **realized/requested ratio is now low only because V*≈command and the robot
is already near it** — that metric is informative only when failing.

## Remaining gaps & open work
- **Polish (not the wall):** yaw 0.167 (largest residual = HL yaw goal err 0.171; still 4×
  better than prior A1); posture (orient_dev 0.13, height_dev 0.11 vs A0 ~0.03) —
  LL-execution quality, not tracking.
- **Smoothness (WL-S, closed 2026-08-28) — two regimes, two different fixes.** Score per
  regime (`--eval-cmd-vx 0.5` / `0.0`); the aggregate is ~95% walking and hides standing.
  **Standing:** `ll_joint_acc_coef=1e-7` + `rel_standing_envs=0.20` reaches the 128-touchdown
  floor on both seeds and *improves* tracking (aggregate `err_vx` 0.083 vs A0 0.095) — the
  levers are superadditive, and the smoothness-vs-tracking frontier is retired. ⚠ Quote
  standing as a range and name the channel: **0.65–0.91× A0 on the mean, 4.2× on realized
  `jacc_legs_p95`, 8.6–11.4× on commanded `ajit_legs_p95`** (at `cmd 0` the mean sits *above*
  the p95 — the set-down transient carries it). On the *aggregate* `jacc_legs_p95` every A1
  arm beats A0 (0.48–0.92×) while sitting 1.0–1.8× above on the mean.
  ⚠ RQ2: A0 trains at `rel_standing_envs=0.05`. **BOTH REGIMES SOLVED 2026-09-05 by
  `hl_cot_coef=5` + `rel_standing_envs=0.20`** (`2026-09-02_17-34-56_..._cot5_standing20_s42`,
  1 seed): walk `act_legs` 0.94x A0, `ajit_p95` 0.50x, `jacc_p95` 0.56x, `ss_err_vx` 0.92x,
  `err_vy` 0.58x, **CoT 0.82x**, stride 1.00x; stand `act_legs` **0.67x**, `ajit`/`jacc` means
  0.93x/0.95x, touchdowns **128 = the floor**, `double_support` 0.998 vs 0.997; 0 falls in
  both. Unique among 11 arms in the `cot {5,7}` x `rse {0.10..0.20}` grid. ⚠ **Two caveats:**
  the standing p95 (transient) channel is still 3.3-11x, unclosed in every arm ever measured;
  and the TRACKING win is speed-specific — `ss_err_vx` is 4.5x A0 at cmd 0.25 and 2.2x at 1.0
  (the smoothness/energy wins ARE speed-robust). That is a signed OVERSHOOT the HL originates
  (`+0.152 m/s` at cmd 0.25, `hl_err` 0.198 > `ll_err` 0.117): `CoT = energy / walked distance`
  pays for covering more ground than commanded. Fix implemented as `hl_cot_cap_commanded`
  (default off, `docs/adr/0004` Amendment 2026-09-05); untested arm pending.
  **Walking alone was solved 2026-09-01 by `hl_cot_coef=5`, not by the cadence range.** A1 had been stepping at ~0.35 s vs A0's 0.578 s
  because the HL commands the range floor at `hl_cot_coef=0.2`; the CoT weight was calibrated
  on a denominator that changed 2026-07-10 (`docs/adr/0004` Amendment). At coef 5 the HL picks
  an interior 0.775 s and the arm beats A0 on **9 of 10** walking metrics — `act_legs` 0.96×,
  `ajit` 0.91×/`p95` 0.50×, `jacc_p95` 0.55×, `err_vx` 0.61×, `err_vy` 0.86×, **CoT 0.77×**,
  power 0.91×, 0 falls (only `jacc` mean is above, 1.36×). 1 seed. This also clears
  **ADR-0004 S4's A1a half**. Imposing the period instead (`(0.625,0.625)` pin or a 0.5–1.0
  draw) matches A0 on `act_legs` but costs `err_vy` 1.38–1.50×, which the HL-chosen period does
  not. Standing at coef 5 is unscoreable (ran at `rse 0.10`, inside that cell's 130–2620
  replicate band) — the open arm is coef 5 + `rse 0.20` + `jacc 1e-7`. Numbers + 39-arm tables
  → the A1a experiment journal (research KB), 2026-08-31 / 2026-09-01.
- **Absolute-necessity ablation — DONE** (`a1_td3_delta_hltrack_5k`): tracking reward alone
  (delta) reaches 0.14/0.11/0.20 → absolute not strictly necessary, just a refinement (see
  Attribution above).
- **`hl=ppo` + absolute + tracking — FAILED** (no cap: `|g|`→13; std cap: `|g|`→4 + falls). The
  two levers don't rescue naive PPO: a std cap bounds σ but not the unbounded Gaussian mean, and
  rsl_rl has no squashed density → PPO can't cleanly bound `g`. TD3's deterministic `tanh` is
  load-bearing (keeps `|g|≤1` so `absolute` bounds `V*`).
- **Open — does the hierarchy buy anything in-sim? A0-vs-A1 robustness/OOD test:** A1's LL was
  relabeled over a broader goal distribution than A0's command curriculum → may generalize
  better to OOD commands / push perturbations. If A1 holds where A0 degrades = a real hierarchy
  benefit; else a clean negative. (The decisive test is still A2's M3 sim2real drop.)
- **`action_rate` penalty — SUPERSEDED, not run.** Its premise (`act_rate` ~1.7 vs A0 0.66)
  was the aggregate number WL-S retired, and both regimes now have fixes that work
  (`hl_cot_coef` walking, `rel_standing_envs` standing). The structural point stands and still
  constrains any future smoothness term: it must enter as a **direct reward term, not the
  goal/intrinsic channel** — action smoothness isn't a goal-space state, so HIRO's intrinsic
  goal-distance reward structurally can't encode it.
- **Reserve variant (not implemented):** LL observes the **absolute** target `V*` instead of
  the delta `V*−s_i`. Two payoffs: matches A0's absolute-command format, AND removes A1's
  runtime base-velocity dependency — the delta needs current vx/vy to build `s`, so deploy must
  estimate base velocity (sportmode/state estimator); absolute `V*` needs none (see
  `.claude/docs/deployment.md`). Low priority for accuracy (oracle proves delta is learnable at
  0.05 m/s) but the cleanest deploy fix. Touch points: goal-obs write in `hrl_runner`
  learn()/`get_inference_policy`, the probe, a matching `ll_goal_obs` flag.
- Then: full omnidirectional cluster run, optional `c`/goal-range/HL-LR sweep.
- **Sim deploy DONE (2026-06-18):** two-ONNX C++ hierarchy (`high_level.onnx`+`low_level.onnx`,
  `State_RLHRL`, oracle + learned), runs in MuJoCo; TD3 abs/delta respond → mechanism in
  `.claude/docs/deployment.md`. **Open sim2real next step:** the LL is twitchy under injected
  estimator noise on the goal-state (`hrl.state_noise`), worst standing still → train LL+HL with
  **state-noise DR** on the goal-state obs (and/or `imu_lin_vel` obs). See the A1 findings ledger (research KB).

## A1 file map (`src/tasks/velocity/`)
```
config/h1_2_a1/
  __init__.py   register Unitree-H1_2-Flat-A1; runner cfg = goal_components source of truth
  env_cfgs.py   reuse flat env; _restructure_obs_groups (actor→policy/command/goal, drop
                actor); fell_over=time_out (A1 only)
  rl_cfg.py     HrlRunnerCfg: LL=PPO + c/goal_components/goal_weights/hl_algorithm/relabeling/
                gamma_hi/warm_start_path/freeze_ll_path/hl_target_mode/hl_reward_mode/hl_ppo/hl_td3
rl/hrl/
  hrl_runner.py  HierarchicalRunner(VelocityOnPolicyRunner): co-train loop, warm-start
                 (_partial_load gap-aware), freeze-LL, save/load+hl, onnx, get_inference_policy
  goal_space.py  GoalComponent/GoalSpace + registry; center/to_target/to_g/scale; goal_dim derived
  high_level.py  HighLevel ABC + OracleHighLevel + HighLevelPpo
  td3.py         HighLevelTd3 (+ inline HIRO relabeling)
  storage.py     HLReplayBuffer + HLBatch
```
No `GoalConditionedWrapper`, no `relabeling.py` — the goal is a plain obs term and relabeling
is inline in `td3.py`.
