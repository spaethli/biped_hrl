# A1 — findings ledger (what was tried, ruled out, learned)

Condensed chronological record of the A1 milestones, failed fixes, and diagnoses that led
to the SOLVED config. Kept so A2/A3 don't re-walk dead ends. Design/status →
`doc/hrl/A1_HIRO.md`; shared machinery → `.claude/docs/hrl-infra.md`. Judge every claim by the
deterministic benchmark, not training-time stochastic metrics.

## Ruled-out / lessons ledger (read this first)
| Lever tried | Outcome | Why / lesson |
|---|---|---|
| Train LL from scratch (orig plan) | FAILED | On-policy PPO can't discover walking in feasible time. **A0 warm-start is load-bearing**, not optional. |
| Naive on-policy `hl=ppo` (M3) | FAILS to track | ~3 HL transitions/env/iter (credit-assignment thin) + tracking term swamped by penalties; co-training oscillates track-and-fall ↔ survive-and-don't-track. Motivates off-policy TD3. |
| TD3 F1 random warmup + F2 split LR | FAILED (worse) | 100k random-coverage transitions evicted from the 500k ring within ~50 iters → coverage is transient. Revealed the deeper bug: a **co-training spiral**, not buffer coverage. |
| TD3 F3 freeze-LL (spiral impossible) | clean NEGATIVE | HL *still* failed to track (err_xy ~2.5) → the spiral was never the core problem. Isolated that the HL itself wasn't learning the map. |
| TD3 F4 add `base_lin_vel` to HL obs | NO-OP, reverted | A/B det-eval: velocity-obs HL ≈ blind HL. Velocity-blindness FALSIFIED; also a privileged input (needs on-robot estimate) for zero gain. **Lesson: always det-eval before committing to a fix** — built on a training-metric asymmetry that an A/B would have killed in a minute. |
| HIRO relabel + `c`-sweep {4,8,12} (M5) | relabel helps vx/vy; cadence is NOT the limiter | Finer `c` (more HL transitions) didn't improve tracking → "too few HL steps" unsupported. Only clean monotonic effect: coarser `c` → smoother + better height. Wall is STRUCTURAL. (Caveat: 1 seed/c.) |
| `absolute`-target HL | fixed LL saturation, **moved wall to HL** | `|g|` 0.84→0.16, LL reach 0.61→0.11 (near oracle). But HL then outputs `g≈0` regardless of command (refuses forward) → end err didn't improve. Isolated the 2nd failure. |
| `tracking` HL reward (+ absolute) | **SOLVED** | Penalty-dominated task reward was making value-max HL = `g≈0`. Tracking-only HL objective → HL asks for the command. A0-level tracking. |
| `tracking` reward alone, delta (ablation) | absolute NOT required | delta+tracking 0.14/0.11/0.20 → `tracking` is the primary lever; `absolute` only refines it (0.14→0.098). |
| `hl=ppo` + absolute + tracking (± std cap) | FAILED both | No cap: `|g|`→13, yaw 1.40. Std cap (1e-3,1.0): still `|g|`→4 + falls (ep_len 9.6) — the cap bounds σ, not the unbounded Gaussian **mean**. PPO can't cleanly bound `g` (rsl_rl has no squashed density); TD3's deterministic `tanh` is load-bearing. |
| Remove warm-start (control) | catastrophic | `|g|` pins 1.0 (100% sat), action_rate 13.5 vs A0 0.66. Confirms warm-start load-bearing from both directions. |
| Deploy goal-state noise (sim, `hrl.state_noise`; learned `absolute` TD3) | stands but **twitchy, worst standing still** | Clean-trained LL isn't robust to estimator noise on its velocity/height goal feedback (cmd=0 → goal=−noise → phantom corrections; HL input isn't noised). **Sim2real fix: train LL+HL with state-noise DR** on the goal-state obs. Adding measured `imu_lin_vel` as a direct obs is an option but privileged + F4 showed base_lin_vel in HL obs was a no-op for *clean* tracking — revisit only as a noise-robustness lever. Deploy mechanism + knob → `.claude/docs/deployment.md`. |

## Milestones

### M1 — scaffolding ✅
A1 task package, declarative goal obs group (`goal_space.py`, not a wrapper),
`HierarchicalRunner`. Env builds; shapes verified.

### M2 — oracle HL ✅ (forced the load-bearing fixes)
Oracle emits the command as `V*`; the warm-started LL tracks it and **out-tracks A0**
(err_xy 0.56→0.33, err_yaw 0.81→0.49). Stable to 10k iters (two long runs, no collapse).
Forced three fixes that became permanent: **A0 warm-start** (gap-aware partial copy),
**`fell_over=time_out`** (A1 only — kills the suicide attractor from the negative
goal-distance reward), **`entropy_coef=0.005`** (0.01 lets std blow up to ~2 and collapse).
Velocity-only(3) vs default(7) ablation (full 10k each): survival identical; **7-dim tracks
better** (yaw 0.47 vs 0.62, calmer std 0.87 vs 1.30) → orient+height are stabilizing
regularizers. **Decision: keep 7-dim.** Inherited A0 weakness: poor yaw → circling
(controlled-for in A0-vs-A1).

### M3 — `hl=ppo` ⚠️ implemented, naive HL fails to track
4 diagnosis runs (4096 envs, A0 warm-start). The learned HL never learns command→goal;
lands in one of two basins:
| run | knobs | outcome |
|---|---|---|
| `a1_ppo_hl_5k` | ent .005, velocity-only(3) | survive, don't track (err_xy ~1.9, yaw ~2.1); entropy collapsed, goals shrank → "emit ~no change". |
| `a1_ppo_7dim_ent02_5k` | ent .02, 7-dim | std blew up (goal_abs 8.7) → impossible targets → died (ep_len ~20). |
| `a1_ppo_7dim_ent01_stdcap_5k` | ent .01 + std cap (1e-3,1.0) | stable & survives, still no track (err_xy ~1.3) → exploration/survival NOT the bottleneck; the HL *mean* won't learn. |
| `a1_ppo_track4x_nocap_2k` | ent .01, +4× track weight | oscillated: best track-while-alive any ppo got (err ~0.2 @ ep_len ~100) → goals grew → constant falling → recovered survival but stopped tracking. Never held both. |
**Conclusion:** credit-assignment + co-training-instability wall → motivates off-policy TD3.
Post-session the 4× tracking reward was **reverted** (RQ2 confound), std cap off, 7-dim
default restored. Untried ppo levers (low priority, TD3 is the fix): longer
`num_steps_per_env` (more HL transitions), 4×-track + std-cap combined.
*Post-fix re-eval (see "play/replay fix"): the M3 `track4x` artifact det-evals to
(0.25,0.20,0.46) — the ppo HL mean did learn a partial map; training-time numbers were
inflated by sampling noise + oscillation.*

### M4 — `hl=td3` — runs 1/2 + F1–F4
TD3 internals: `hrl-infra.md`. First run `a1_td3_5k` failed: **actor saturated immediately**
(|g|≈0.88 by it250; deterministic policy gradient races to the tanh bounds vs a fresh
critic) and **self-locked via coverage collapse** (noise σ=0.2 around ±0.88 → all buffered
goals in [0.68,1.0], critic has no interior data, no inward gradient). Missing TD3
`start_timesteps` (uniform-random warmup).
- **F1** (random goals while `len(buffer)<warmup`) **+ F2** (split actor LR 3e-4 / critic
  1e-3): run `a1_td3_warmup_5k` was WORSE (reward 0.3→−552, LL std 0.33→1.73). Warmup
  coverage evicted in ~50 iters. **Revised root cause: a co-training spiral** — extreme
  goals degrade the LL; the HL reward (summed task reward) is then dominated by
  LL-violence penalties that are ~goal-independent, so Q(s,g) carries ~no tracking gradient.
- **F3 freeze-LL** (`freeze_ll_path`, velocity-only frozen oracle LL → spiral structurally
  impossible): run `a1_td3_frozenLL_5k` clean NEGATIVE — HL still failed (err_xy ~2.5). So
  the spiral wasn't the core problem either. (Velocity-only chosen so a HIRO-delta HL's
  goals stay in-distribution for the memoryless frozen LL.)
- **F4 velocity observability** (add `base_lin_vel` to HL obs): A/B det-eval blind
  (0.85,0.39,0.28) ≈ velocity-obs (0.85,0.38,0.25) → NO-OP, reverted. Falsified the
  velocity-blindness hypothesis built on a misleading training-metric yaw/xy asymmetry.

### M5 — HIRO relabeling + `c`-sweep ✅ implemented; wall shown STRUCTURAL
Relabeling inline in `HighLevelTd3` (k=10 candidates: stored g, empirical
`(s_{t+c}−s_t)/scale`, +8 Gaussian; scored by current LL log-likelihood of the stored
action trace; logs `hl/relabel_frac`). Benchmark, `model_3000`:
| c | HL steps/ep | err_vx | err_vy | err_yaw | action_rate | height_dev | falls |
|---|---|---|---|---|---|---|---|
| 4 | 250 | 0.47 | 0.30 | 0.41 | 1.52 | 0.072 | 0 |
| 8 | 125 | 0.42 | 0.35 | 0.64 | 1.41 | 0.064 | 0 |
| 12| 83  | 0.39 | 0.26 | 0.58 | 1.23 | 0.048 | 0 |
| A0 | — | 0.09 | 0.11 | 0.10 | 0.66 | 0.027 | 0 |
Relabel gives the best TD3 vx/vy (0.42/0.35). Finer `c` did NOT help tracking →
"too few HL steps" unsupported. Coarser `c` → smoother + better height (the credible signal;
1 seed/c). **Wall is structural** → goal-achievability probe. Also fixed the `gamma_hi`
footgun (was hardcoded `0.99**8`; now derived `0.99**c` unconditionally).

## Goal-achievability probe — structural limiter PINPOINTED (2026-06-16)
Tool `play.py --diagnose-goals` (see `hrl-infra.md`). Decomposes velocity error per HL
window: `(command−achieved) = (command−V*)[HL goal err] + (V*−achieved)[LL reach err]`.
Probe on TD3+relabel c-sweep (`model_3000`), consistent across c∈{4,8,12}:
| run | \|HL err\|vx | \|LL err\|vx | end vx | \|g\|vx (sat) | realized/req vx | fwd | bwd |
|---|---|---|---|---|---|---|---|
| c4  | 0.55 | 0.57 | 0.46 | 0.87 (68%) | −0.01 | 0.58 | 0.29 |
| c8  | 0.51 | 0.61 | 0.40 | 0.84 (58%) | −0.00 | 0.49 | 0.29 |
| c12 | 0.59 | 0.55 | 0.39 | 0.95 (87%) | +0.04 | 0.34 | 0.50 |
- Velocity goal channel is **functionally dead under the learned HL**: it requests ~+0.3–0.4
  m/s vx/window, the LL delivers ~0 (realized/req ≈ 0). `|g|` pinned near the bound.
- **Oracle control (DECISIVE):** SAME warm-started LL + oracle HL → LL reach err **vx 0.048,
  vy 0.038, yaw 0.059**, `|g|` 0.08 (1% sat), fwd ≈ bwd. ⇒ **the LL is a competent
  goal-conditioned tracker when handed sensible goals**; the large learned-HL "LL reach err"
  is a downstream consequence of saturated, unreachable targets, NOT LL incompetence.
- **Symmetric-x control:** flipping lin_vel_x to (−0.5,0.5) left vx err / saturation /
  forward-avoidance UNCHANGED ⇒ **goal-scale / command-asymmetry RULED OUT**.

**Verdict:** limiter is the **command→goal map** — the learned HL collapses to saturated
goals (`|g|→1`), and the co-trained LL never acquires velocity goal-conditioning against
those bad goals (co-adaptation collapse). Forward-avoidance is a pure HL artifact (zero
under oracle). LL, LL intrinsic reward, and command range all CLEARED. Full diagnosis:
`doc/hrl/A1_goal_achievability_probe.md`. → led to the two-lever fix (`doc/hrl/A1_HIRO.md`).

## Absolute-target run → tracking-reward run (the fix chain)
Detailed A/B tables and the SOLVED result are in `doc/hrl/A1_HIRO.md` ("Current results").
Sequence: `absolute` solved LL saturation but flipped the error onto the HL (HL goal err
0.71, refuses forward → penalty-dominated task reward makes value-max = `g≈0`); adding
`tracking` HL reward made the HL ask for the command → **err_vx 0.098 / err_vy 0.091,
A0-level, 0 falls.** Ablation (2026-06-17): `tracking` is the primary lever (alone, delta:
0.14); `absolute` only refines it.

## Estimator-noise DR (#8b) — reliable, clean-baseline quality (2026-06-22)
Train the LL goal channel on a deploy-realistic base-velocity estimate (corrupts ONLY the LL's
observed `V*−s`; HL/reward/critic stay clean — and the HL never observes lin-vel anyway, so
nothing on the HL side *can* be noised). Resolves the "train LL+HL with state-noise DR" todo from
the deploy row above. All clean-eval ≈ ref5k (err_vx ~0.08, ep_len max, 0 falls, no goal saturation):

| variant (seed) | err_vx | outcome |
|---|---|---|
| bias 0.10 (s42, orig) | 0.53 | COLLAPSE — **transient training corruption** (not seed, not config) |
| bias 0.10 (s42, rerun) | 0.078 | success |
| bias 0.10 (s123) | 0.083 | success |
| full = bias + drift .01/.99 + lag 3 (s42) | 0.072 | success |
| full (s123) | 0.083 | success |

Verdict: bias-only and full DR both train **reliably** (2/2 clean each) to clean-baseline quality;
the lone collapse (ep_len 8.9, HL goals 54% saturated — the old A1 wall) was a one-off glitch,
confirmed benign by a successful same-seed-42 rerun (ep_len 724 by it2000). Do NOT report bias
instability. Logs `logs/rsl_rl/h1_2_velocity_a1/2026-06-19_*noise_abs_*`.

## Play/replay fix (F0, 2026-06-11) — all pre-fix qualitative replays are void
`play.py` used `get_inference_policy()` which returned the bare LL actor — nothing fired the
HL or wrote `env.hrl_goal`, so **every pre-F0 A1 replay showed the LL with goal frozen at 0**
(command physically couldn't reach the LL). W&B training metrics (HL in the loop) were always
ground truth. Fix: `HighLevel.act_inference` + `HierarchicalRunner.get_inference_policy`
mirroring the training goal wiring. Plus the structure-restore (so non-default goal spaces /
trained HLs load correctly instead of silently replaying an oracle HL) — see `hrl-infra.md`.
Validated headless on 4 checkpoints.
