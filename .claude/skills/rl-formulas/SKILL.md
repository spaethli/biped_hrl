---
name: rl-formulas
description: Generate report/thesis-ready LaTeX formulas plus implementation locations for this project's RL components (rewards, goals, observations, PPO losses, the A1/HIRO hierarchy). Use whenever the user asks for "formulas", "the math", "LaTeX for the report/thesis", or equations describing the oracle high level, low-level PPO, goal space, intrinsic reward, GAE/bootstrap, or related pieces.
---

# RL formula export

You produce thesis-ready math for the A1 HIRO implementation in this repo. The output
must match the **live code**, not training data or memory — the goal space and rewards
have changed several times in this project.

## Always do this

1. **Read the live code FIRST — never derive formulas from memory.** Read at minimum:
   - `src/tasks/velocity/rl/hrl/goal_space.py` — `extract`, `oracle_target`, `reward`, `scale`, the per-component extractors and weights.
   - `src/tasks/velocity/rl/hrl/hrl_runner.py` — the `learn()` co-train loop (goal injection `Δ = V* - s`, intrinsic reward call, `process_env_step`, the HL window calls).
   - `src/tasks/velocity/rl/hrl/high_level.py` — `OracleHighLevel.act` and `HighLevelPpo` (M3).
   - `src/tasks/velocity/config/h1_2_a1/rl_cfg.py` — goal weights, `c`, `gamma`, `gamma_hi`, `entropy_coef`, `desired_kl`, `num_steps_per_env`, `HlPpoCfg`.
   - `src/tasks/velocity/config/h1_2_a1/env_cfgs.py` — obs surgery (`_restructure_obs_groups`: drop command, add goal) and `fell_over.time_out = True`.
   - `src/tasks/velocity/velocity_env_cfg.py` — physics dt / decimation / control rate, twist command ranges + curriculum, reward terms.
   - For PPO/GAE specifics, cite the installed `rsl_rl` `algorithms/ppo.py` (`compute_returns`, the time-out bootstrap at `process_env_step`).
   Read numeric constants (weights, `c`, γ, nominal height, command ranges) from the code — do not guess them.

2. **Output two parts:**
   - (a) A markdown table: each concept → clickable `file:line` location (use `[name](path#Lstart-Lend)` links).
   - (b) LaTeX in copy-paste ` ```latex ` blocks, grouped in this order: timescales/notation, goal-space state `s`, oracle target `V*`, low-level observation `Δ`, intrinsic reward `r^lo`, PPO objective + GAE, termination bootstrap, and a one-line canonical-HIRO relation.

3. **Define every symbol** the first time it appears. Use `amsmath` constructs (`align`, `cases`, `\lVert\cdot\rVert`, `^\top`). Prefer one equation per block so they paste cleanly.

4. **Scope = the requested milestone** (`$ARGUMENTS` may set it):
   - M1/M2 → oracle HL + goal-conditioned LL only.
   - M3 → add the learned HL PPO: `V* = s_t + scale ⊙ g`, windowed HL return `R = Σ_i r^task_i`, `gamma_hi = gamma^c`, HL actor obs `(policy, command)`.
   - No scope given → produce M1–M2 and note M3 is available.
   - A component filter (e.g. "reward only", "goal space") → return just that block with its location.

5. **PPO loss — ONE convention, and A0 LL ≡ A1 LL.** The A1 low level is the *same*
   `rsl_rl.PPO` object as the A0 agent, so emit the **identical** PPO formula for both —
   never show A0 with one form and A1 with another (that drift is a bug, not a real
   difference). rsl_rl (`algorithms/ppo.py:259-275`) implements the clipped-surrogate
   **loss-to-minimize**, not the objective-to-maximize:
   `L_surr = E[max(-A·ρ, -A·clip(ρ,1-ε,1+ε))] = -E[min(ρA, clip(ρ)A)]`,
   then `L = L_surr + c_v·L_VF - c_e·H` (entropy subtracted; clipped value loss).
   Default to this code-faithful **max(-A·ρ, …) loss form** and add one sentence noting
   it equals standard clipped PPO `-L^CLIP`. If the user prefers the paper's
   `min(ρA, …)` objective form, use it — but then apply it consistently everywhere and
   keep the leading minus in the total loss.

6. **Flag divergences from canonical HIRO** — they are report-worthy:
   - absolute target `V*` + directional observation `Δ = V* - s` vs canonical `V* = s_t + g`;
   - intrinsic reward = **sum of per-component L2 norms** (`Σ_c w_c ‖·‖`), not a single norm over the whole goal vector;
   - learned-HL goal map is **linear** `scale ⊙ g` (a raw Gaussian sample), NOT tanh-squashed (rsl_rl's GaussianDistribution applies no Jacobian correction, so tanh would be a silent bug);
   - `fell_over` is a truncation (bootstrapped), so the always-negative reward has no value-0 terminal to exploit.

## Example invocations
- `/rl-formulas` → M1–M2 formulas + location table.
- `/rl-formulas M3` → learned-HL PPO additions.
- "give me the LaTeX for the intrinsic reward" → just the reward block, with `goal_space.py` reward lines cited.

## Notes
- Deep context for the architecture lives in `doc/hrl/A1_HIRO.md` (design/status) + `.claude/docs/hrl-infra.md` (shared machinery) — consult them for design rationale, but take the *equations* from the code.
- Keep numbers current: e.g. default goal space is velocity(3)+orientation(3)+height(1)=7, weights velocity 3 / orientation 1 / height 1, `c=8`, γ=0.99, γ_hi=0.99^8, control rate 50 Hz (dt 0.02 s). Re-verify each run since these have changed before.
