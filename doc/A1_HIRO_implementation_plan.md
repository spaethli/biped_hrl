# A1 — HIRO (Hybrid PPO + TD3) Implementation Plan

> **Status:** Design doc for review. No code written yet.
> **Scope:** Architecture A1 of the thesis (structural hierarchy). Hybrid design:
> on-policy PPO low level + off-policy TD3 high level with HIRO goal-relabeling
> correction. Relabeling is a runtime toggle (enables a clean ablation).
> **Framework:** stays in `unitree_rl_mjlab` + `mjlab` + `rsl_rl`. No Isaac Lab.

---

## 1. Design goals & constraints

| Goal | How it's met |
|---|---|
| Isolate the *hierarchy* effect for RQ2 (A0 vs A1) | Low level uses the **same PPO** as A0; same reward/obs/DR env. Only the structure changes. |
| Genuine HIRO goal relabeling | High level is **off-policy TD3 with a replay buffer**; relabeling correction operates on replayed HL transitions where it is mathematically valid. |
| Don't fork mjlab/rsl_rl | All new code lives under `src/tasks/velocity/`. We reuse `rsl_rl.PPO` for the LL and write a small self-contained TD3 for the HL (≈ acts on a 3-dim goal only → tiny nets). |
| Deployment-faithful | Both HL and LL consume **only real-robot-available observations** (the `actor` obs group; no privileged `base_lin_vel`). Intrinsic reward uses privileged sim velocity, but reward is never needed at deployment. |
| Hierarchy + relabeling as ablation knobs | `hl_algorithm {ppo,td3}` and (for td3) `relabeling {none,hiro}` → see the run matrix below. |

### The switch is the HL learner, not just a relabel flag
Relabeling = off-policy correction = **needs a replay buffer**. PPO is on-policy:
it collects a rollout, runs a few epochs, then discards the data — there is
nothing stored to relabel and replay. So the choice is structural:

- **`hl_algorithm="ppo"`** → high level is on-policy PPO → **2-level PPO**
  (no relabeling possible). Reuses `rsl_rl.PPO` for *both* levels → minimal new code.
  This is the "naive hierarchy" baseline.
- **`hl_algorithm="td3"`** → high level is **off-policy TD3 + replay buffer**, with
  a `relabeling {none,hiro}` sub-toggle. This is the new code (TD3 + buffer +
  relabeling). HIRO's correction lives here, at the sparse HL (one transition per
  `c` steps → a small buffer is cheap).

Why relabeling can't be on-policy (the math): PPO's surrogate uses the ratio
`π_new(a|s,g)/π_old(a|s,g)` with `π_old` the behaviour policy at collection time.
Relabeling `g → g̃` changes the policy **input**, so the stored action `a` was
never sampled from `π_old(·|s,g̃)`; the relabeled tuple is off-policy w.r.t. the
new goal and needs importance weights or a replay buffer. **The low level stays
on-policy PPO in all configs and is never relabeled** (relabeling is HL-only).

### Run matrix (all from one codebase)
| Run | HL learner | Relabel | Answers |
|---|---|---|---|
| **A0** (`Unitree-H1_2-Flat`) | — | — | baseline |
| **A1 `hl=ppo`** | PPO | n/a | does *structural hierarchy* help? (clean: both PPO) |
| **A1 `hl=td3 relabel=none`** | TD3 | ✗ | isolates *off-policy HL* (and is the clean control for relabeling) |
| **A1 `hl=td3 relabel=hiro`** | TD3 | ✓ | full HIRO; vs the row above isolates the *relabeling correction* |

Caveat to keep in mind for the writeup: `hl=ppo` → `hl=td3 relabel=hiro` changes
*two* things (PPO→TD3 **and** relabeling). The `relabel=none` TD3 row is the clean
control that attributes any further gain to relabeling alone.

---

## 2. Roles, timescales, and the goal space

```
                    ┌─────────────────────────────────────────────┐
external twist  c → │  HIGH LEVEL  μ_hi(s, c) → g    (TD3, off-pol)│  fires every c steps
(vx*,vy*,ψ̇*)        │     g = (Δvx, Δvy, Δψ̇) ∈ ℝ³  (velocity delta)│
                    └───────────────────────┬─────────────────────┘
                                            │ goal g (held constant over the window)
                    ┌───────────────────────▼─────────────────────┐
state s         →   │  LOW LEVEL   μ_lo(s\c, g) → a  (PPO, on-pol) │  fires every step
(proprio, no cmd)   │     a = 27 joint position targets            │
                    └───────────────────────┬─────────────────────┘
                                            │ a → PD → MuJoCo
                                            ▼
                       intrinsic reward r_lo = −‖(v_t + g) − v_{i+1}‖   (sim velocity)
```

- **Control rate:** decimation 4 × dt 0.005 s → 50 Hz control step.
- **`c` (HL period):** **8** (= 0.16 s) [confirmed]. Must divide `num_steps_per_env`.
- **Goal `g`:** velocity **delta** per the thesis spec, `(Δvx, Δvy, Δψ̇)`. The goal
  bound is **derived from the active command-curriculum ranges** [confirmed] — the
  HL output is `tanh × goal_scale` where `goal_scale` is read from the twist
  command ranges at construction (and re-read if the curriculum stage changes), so
  editing the curriculum automatically rescales the goal. The window target is
  `V* = v_t + g` computed at HL fire-time from the privileged sim velocity `v_t`.

### Critical design decision — the command is removed from the LL observation
The LL obs group is the A0 `actor` group **minus the `command` term, plus the
`goal` term**. If the LL could see the external command, it would bypass the
hierarchy and track the command directly, collapsing A1 into A0. Removing the
command forces the goal `g` to be the *only* channel through which task intent
reaches the LL — which is exactly what makes the A0↔A1 comparison meaningful.

| Network | Obs groups | Dim | Notes |
|---|---|---|---|
| HL actor `μ_hi` | `actor` (incl. command) | 92 | deployable obs only |
| HL critic (twin Q) | `actor` ⊕ `goal` | 92 + 3 | TD3 Q(s,g) |
| LL actor `μ_lo` | `actor∖command` ⊕ `goal` | 89 + 3 = 92 | deployable obs only |
| LL critic | `critic∖command` ⊕ `goal` | (critic−3) + 3 | privileged ok (not deployed) |

(92 = ang_vel 3 + proj_grav 3 + command 3 + phase 2 + joint_pos 27 + joint_vel 27 + last_action 27.)

---

## 3. Reward decomposition

| Level | Reward | Source |
|---|---|---|
| **HL** | the **A0 task reward**, summed over the `c`-step window: `R = Σ_{i=t}^{t+c-1} r_task,i` | existing reward manager, unchanged |
| **LL** | intrinsic `r_lo,i = −‖V* − v_{i+1}‖₂` with `V* = v_t + g` (velocity subspace = (vx, vy, ψ̇)) | computed in the LL env wrapper from sim base velocity |

- HL discount `γ_hi = γ^c` (≈ 0.99⁸) so **HL and LL horizons align** [confirmed].
  Applies to both the `hl=ppo` (GAE) and `hl=td3` (TD target) paths.
- LL keeps `γ=0.99, λ=0.95` (A0 values).
- The LL receives **no** task reward (pure HIRO). Optional `ll_task_reward_coef`
  blend is left as a config knob (default 0) but off for the clean ablation.

---

## 4. Co-training loop (one learning iteration)

```
for it in range(max_iterations):
    # ---- Rollout: num_steps_per_env LL steps (must be a multiple of c) ----
    for k in range(num_steps_per_env):
        if k % c == 0:                         # HL fires
            v_t   = env.base_velocity()        # privileged sim velocity (vx,vy,ψ̇)
            g     = hl.act(actor_obs)          # TD3 actor (+ exploration noise)
            V_tar = v_t + g                    # window target, frozen for c steps
            hl_open = HLTransition(s_t=actor_obs, g=g, R=0, seq=[])  # start accumulating
        ll_obs = build_ll_obs(obs, g)          # drop command, append goal
        a      = ppo.act(ll_obs)               # rsl_rl PPO act() (stores LL transition)
        obs, _, dones, extras = env.step(a)
        v_next = env.base_velocity()
        r_lo   = -(V_tar - v_next).norm(dim=-1)
        ppo.process_env_step(ll_obs_next, r_lo, dones, extras)   # LL transition complete
        hl_open.R   += task_reward             # accumulate task reward into HL return
        hl_open.seq.append((ll_obs, a))        # store (s_i,a_i) for relabeling
        if (k+1) % c == 0 or dones.any():      # HL window closes
            hl_open.s_next = actor_obs_next; hl_open.done = dones
            hl_buffer.add(hl_open)             # push to TD3 replay buffer

    # ---- Updates ----
    ppo.compute_returns(last_ll_obs); ll_losses = ppo.update()       # on-policy LL update
    if hl_algorithm == "ppo":                                        # on-policy HL update
        hl.compute_returns(last_actor_obs)                           # GAE over the few HL steps
        hl_losses = hl.update()                                      # standard rsl_rl PPO
    else:                                                            # off-policy HL update
        hl_losses = hl.update(hl_buffer, ll_policy=ppo.actor,
                              relabel=relabel_strategy, n_grad_steps=…)
    log(it, ll_losses, hl_losses, …)
    if it % save_interval == 0: save(...)
```

The HL window-close step routes the transition to the right sink:
`hl=ppo` → an HL `RolloutStorage` (on-policy, consumed & cleared each iter);
`hl=td3` → the replay buffer (off-policy, persists across iters).

Notes:
- `num_steps_per_env=24`, `c=8` → 3 closed HL windows / env / iter × 4096 envs ≈
  **12 k HL transitions/iter**. For `td3` this fills the buffer; for `ppo` it's the
  on-policy batch (short GAE horizon of 3 HL steps — dense windowed reward makes
  this workable; bump `num_steps_per_env` if HL credit assignment looks weak).
- HL TD3 does `n_grad_steps` (default 8) sampled minibatch updates per iteration.
- Episode resets mid-window close the HL window early and mask the bootstrap.

---

## 5a. High-level PPO path (`hl_algorithm="ppo"`)

`src/tasks/velocity/rl/hrl/hl_ppo.py` (thin wiring around `rsl_rl.PPO`)

- A second standard `rsl_rl.PPO` whose **actor** is an `MLPModel` `[92 → 256 → 256 → 3]`
  with a `GaussianDistribution` (scalar std) squashed/scaled to `goal_scale`, and a
  **critic** `MLPModel` `[92 → 256 → 256 → 1]`. Same machinery as the A0 agent, just
  a 3-dim action head.
- Fed HL transitions at the HL timescale via its own `RolloutStorage`
  (`num_transitions = num_steps_per_env / c` per env). `act → process_env_step →
  compute_returns → update` mirror the LL exactly.
- **No relabeling** (on-policy). This is the naive-hierarchy baseline.

## 5b. High-level TD3 path (`hl_algorithm="td3"`, self-contained)

`src/tasks/velocity/rl/hrl/td3.py`

- **Actor** `μ_hi`: MLP `[92 → 256 → 256 → 3]`, `tanh` × goal_scale. EmpiricalNormalization on input (reuse `rsl_rl.modules.EmpiricalNormalization` for export parity).
- **Twin critics** `Q1,Q2`: MLP `[(92+3) → 256 → 256 → 1]`.
- **TD3 tricks:** target nets (soft τ=0.005), target-policy smoothing (clipped
  Gaussian noise on next-goal), clipped double-Q target, **delayed actor update**
  (policy_freq=2), exploration noise (Gaussian, decaying).
- **Replay buffer:** ring buffer of `HLTransition` storing
  `(s_t, g_t, R, s_{t+c}, done)` **plus** the per-window sequence `[(s_i, a_i)]`
  (needed only for relabeling; can be dropped from buffer when `relabeling=none`
  to save memory).

### TD3 target (with relabeling applied at sample time)
```
g̃        = relabel_strategy.relabel(batch, ll_policy)     # HL-only correction
a'       = clip(μ_hi_target(s') + clip(ε,−c,c), g_low, g_high)
y        = R + γ_hi · (1−done) · min(Q1_target(s',a'), Q2_target(s',a'))
L_Q      = MSE(Q1(s,g̃), y) + MSE(Q2(s,g̃), y)
L_actor  = −Q1(s, μ_hi(s)).mean()           # every policy_freq steps
```

---

## 6. Pluggable relabeling

`src/tasks/velocity/rl/hrl/relabeling.py`

```python
class RelabelStrategy(ABC):
    @abstractmethod
    def relabel(self, batch: HLBatch, ll_policy: MLPModel) -> torch.Tensor:
        """Return goals to use for this HL minibatch (shape [B,3])."""

class NoRelabel(RelabelStrategy):
    def relabel(self, batch, ll_policy): return batch.goals

class HiroOffPolicyCorrection(RelabelStrategy):
    """HIRO §3.2: g̃ = argmax_g Σ_i log π_lo(a_i | s_i, g)."""
    n_candidates = 10          # original + achieved-delta + 8 Gaussian
    sigma = 0.5                # std of candidate sampling (× goal range)
    def relabel(self, batch, ll_policy):
        # candidates: [B, K, 3] = {g_orig, (v_{t+c}-v_t), 8×N(achieved, σ)}
        # for each candidate, set LL goal, eval stored a_i under LL Gaussian,
        # sum log-probs over the window seq, pick argmax → [B,3]
        ...
```

- Candidate set per HIRO: original `g`, the **achieved** velocity delta
  `v_{t+c}−v_t`, and 8 Gaussian samples centred on the achieved delta, clipped to
  goal bounds.
- Log-prob uses the **current** LL Gaussian (`ll_policy.distribution`), evaluated
  on the stored `(s_i, a_i)` with the candidate substituted into the goal slot.
- Log a **relabel-acceptance rate** (fraction where `g̃ ≠ g_orig`) — a useful
  diagnostic and an ablation talking point.

---

## 7. File-by-file plan

```
src/tasks/velocity/
├── config/h1_2_a1/
│   ├── __init__.py          # register Unitree-H1_2-Flat-A1 (+ -Rough-A1)
│   ├── env_cfgs.py          # reuse unitree_h1_2_flat_env_cfg(); add `goal` obs
│   │                        #   group; build LL actor/critic groups (drop cmd)
│   └── rl_cfg.py            # HrlRunnerCfg: ll (PPO) + hl_algorithm + hl_ppo/hl_td3 + relabel
└── rl/
    ├── runner.py            # (existing VelocityOnPolicyRunner — unchanged)
    └── hrl/
        ├── __init__.py
        ├── hrl_runner.py    # HierarchicalRunner: drives LL-PPO + HL (ppo|td3) co-train,
        │                    #   save/load, export_policy_to_onnx (two models)
        ├── hl_ppo.py        # HighLevelPpo (wires a 2nd rsl_rl.PPO at the HL timescale)
        ├── td3.py           # HighLevelTd3 (actor, twin critics, update)
        ├── storage.py       # HLReplayBuffer (td3) + HLTransition/HLBatch
        ├── goal_env.py      # GoalConditionedWrapper: base_velocity(), build_ll_obs,
        │                    #   intrinsic reward, goal buffer the obs term reads
        └── relabeling.py    # RelabelStrategy + NoRelabel + HiroOffPolicyCorrection
```

### Goal injection mechanism (keeps the env config reusable)
The A1 `env_cfgs.py` adds a 3-dim `goal` observation **group** whose term reads
`env.unwrapped.hrl_goal` (a `[num_envs,3]` buffer). The `HierarchicalRunner`
writes that buffer when the HL fires. Construction-time obs sample includes a
zero goal so `MLPModel._get_obs_dim` resolves correctly. This way the LL/HL
models are **standard `MLPModel`s** (clean ONNX export) and the env stays a plain
mjlab env (the goal is "just another observation").

### Config sketch (`rl_cfg.py`)
```python
@dataclass
class HrlRunnerCfg(RslRlBaseRunnerCfg):
    class_name = "HierarchicalRunner"
    c: int = 8                              # HL period (divides num_steps_per_env)
    ll: RslRlOnPolicyRunnerCfg              # reuse A0 PPO cfg (actor/critic/algorithm)
    hl_algorithm: Literal["ppo","td3"] = "td3"
    hl_ppo: HlPpoCfg                        # used when hl_algorithm=="ppo"
    hl_td3: HlTd3Cfg                        # used when hl_algorithm=="td3" (τ, policy_freq, noise)
    relabeling: Literal["none","hiro"] = "hiro"   # td3 only; ignored for ppo
    gamma_hi: float = 0.99 ** 8             # = γ^c, horizon-matched [confirmed]
    goal_scale: tuple | None = None         # None → derive from twist command ranges [confirmed]
    ll_task_reward_coef: float = 0.0        # 0 = pure HIRO (LL intrinsic only)
```
`goal_scale=None` means the runner reads the active `commands["twist"].ranges`
(`lin_vel_x/y`, `ang_vel_z`) at build time to bound the goal, so editing the
command curriculum rescales the goal automatically.

---

## 8. Checkpointing, ONNX, deployment

- **Checkpoint** (`model_<it>.pt`): `{ll_actor, ll_critic, ll_normalizer, hl_actor,
  hl_critics, hl_targets, hl_normalizer, hl_optimizer, iter}`. Replay buffer
  optionally saved for resume.
- **ONNX export:** two files — `policy_hl.onnx` (92→3) and `policy_ll.onnx`
  (92→27). Reuse the proven `MLPModel.as_onnx()` path for both (normalizer copied
  via deepcopy — same rule as A0).
- **Deployment (later milestone):** the C++ `State_RLBase` runs `policy_ll.onnx`
  every step and `policy_hl.onnx` every `c` steps, feeding the HL output into the
  LL input slot. Build the LL input as `proprio∖command ⊕ goal`. This is the only
  C++ change; flagged for after sim works. (Two-model chaining; no base-linear-
  velocity estimate required because the LL consumes the raw goal, not `V*`.)

---

## 9. Logging (W&B project `biped_hrl`, experiment `h1_2_velocity_a1`)

LL (existing PPO metrics) + `intrinsic_reward/mean`, `task_reward/mean`,
`hl/q_loss`, `hl/actor_loss`, `hl/goal_mean_{vx,vy,wz}`, `hl/relabel_accept_rate`,
`hl/buffer_size`, `episode_length`, plus the A0 reward-term breakdown.

---

## 10. Milestones (recommended order)

1. **Scaffolding + plumbing.** A1 task package, goal obs group, `GoalConditionedWrapper`,
   `HierarchicalRunner` skeleton. Env builds; shapes verified.
2. **Oracle-HL sanity test.** Bypass the HL and feed the *true external command*
   as the goal directly. The LL should learn to track goal=command and reach ≈ A0
   performance. **This validates the entire LL plumbing (obs surgery, intrinsic
   reward, goal injection) before any HL learning is introduced.**
3. **`hl=ppo` (2-level PPO).** Wire `HighLevelPpo`; co-train. This is the naive-
   hierarchy baseline and the simplest learned HL — get it stable first.
4. **`hl=td3 relabel=none`.** Add `HighLevelTd3` + replay buffer; co-train. Compare
   to `hl=ppo`.
5. **`hl=td3 relabel=hiro`.** Add the correction; log accept-rate; ablate vs row 4.
6. **Tuning + full omnidirectional run on cluster** (A100). Sweep `c ∈ {5,8,10,16}`,
   goal ranges, HL LR.
7. **Deployment** (two-model ONNX + C++ chaining) — after sim is stable.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Co-training instability (HL & LL are moving targets) | TD3 delayed/soft updates, small HL LR, relabeling; staged milestones (oracle → ppo → td3) catch instability early |
| Hierarchy collapse (LL ignores `g`) | command removed from LL obs; monitor intrinsic reward & goal-sensitivity; oracle-HL milestone catches plumbing bugs early |
| Too few HL transitions/iter | off-policy buffer accumulates across iters; `num_steps_per_env % c == 0` |
| Deployment needs base lin-vel | avoided: LL consumes raw `g`, not `V*`; sim-only privileged velocity used only for the (train-time) reward |
| `c` sensitivity | explicit ablation sweep |

---

## 12. Confirmed decisions

1. **`c` = 8** (0.16 s).
2. **Goal range tied to the command curriculum** (`goal_scale=None` → derived from
   `commands["twist"].ranges`), so editing the curriculum rescales the goal.
3. **Train A1 from scratch** (no A0 warm-start) — cleaner A0↔A1 comparison.
4. **`γ_hi = γ^c`** — HL and LL horizons aligned.
5. **HL switch is the learner:** `hl_algorithm ∈ {ppo, td3}`; `relabeling ∈ {none,
   hiro}` applies only to `td3`. Relabeling cannot exist on the on-policy `ppo` HL.

---

## 13. Potential A1 variant — HAC (Hindsight Actor-Critic) [not committed]

A possible *second* HRL algorithm for the A1 slot, comparing HIRO vs HAC under the
same env/reward/DR. **Not part of the current plan** — noted for scope discussion.

**What HAC is** (Levy et al., ICLR 2019; ref: `Hierarchical-Actor-Critic-HAC-PyTorch/HAC.py`,
`h-baselines/.../goal_conditioned`): off-policy, ≥2-level, with three mechanisms —
(a) **subgoal testing** (penalize HL by −H when the LL can't reach a tested subgoal),
(b) **hindsight action transition** (relabel the HL action with the *achieved* state →
HL trains as if the LL were optimal), (c) **hindsight goal transition** (HER on the
lowest level). Sparse goal-reached reward; DDPG/TD3 learners.

**Why it does NOT swap into the current hybrid:** HAC's strength needs hindsight at
**both** levels — its low level is fundamentally HER-based (off-policy, replay buffer,
sparse reward). Our hybrid deliberately keeps **LL = on-policy PPO with a dense
distance reward** (so the A0↔A1 comparison isolates the hierarchy). Doing HAC properly
means **off-policy at both levels** → migrate the LL off PPO (skrl/Isaac, since rsl_rl
is on-policy only) → which *breaks the "LL = A0 PPO" control* that A1 relies on.

**If pursued, treat as A1-HAC, a parallel variant, not a toggle:**
- Off-policy both levels (TD3/DDPG). Reference: port from `HAC-PyTorch` (PyTorch,
  single-env → batch/GPU) or `h-baselines` (TF, reference only).
- Engine options: a custom batched TD3 at both levels, **or** adopt `skrl` (GPU
  off-policy, but it owns the rollout loop → larger refactor).
- New comparison it buys: **A1-HIRO vs A1-HAC** = "off-policy correction" vs
  "hindsight + subgoal testing" for non-stationarity, on the same robot/task.
- Cost: a second off-policy LL path (not shared with A0/A2), more tuning (subgoal
  test ratio, tolerance/threshold, level horizon H), and it no longer reuses the
  proven PPO LL.

**Recommendation:** keep HIRO as the committed A1; revisit A1-HAC only if time remains
after A1-HIRO and A2, as an extra RQ-relevant data point.
```
