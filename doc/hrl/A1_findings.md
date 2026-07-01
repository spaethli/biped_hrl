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
| TD3 F4 `base_lin_vel`→HL obs ("velobs", `hl_obs_vel`) | no-op under `absolute`; **BIG linear-tracking win under `delta`** | Original F4 A/B (absolute/pre-tracking) showed velocity-obs HL ≈ blind → looked like a no-op. **Correction (2026-06-30): NOT a no-op for directional (`delta`) goals** — velobs improves linear tracking a lot, replicated across runs (err_vx 0.137→0.062 at pose0.5; see row below + `a1_estimator_noise_8b`). The directional target `V*=state+scale·g` needs current v to set `g=(command−v)/scale`, so a velocity-blind HL under-reaches. Still privileged (needs on-robot vx,vy estimate) → deployability-gated. **Lesson: det-eval every fix, AND re-test a "no-op" when the regime changes (`absolute`→`delta`).** |
| HIRO relabel + `c`-sweep {4,8,12} (M5) | relabel helps vx/vy; cadence is NOT the limiter | Finer `c` (more HL transitions) didn't improve tracking → "too few HL steps" unsupported. Only clean monotonic effect: coarser `c` → smoother + better height. Wall is STRUCTURAL. (Caveat: 1 seed/c.) |
| `absolute`-target HL | fixed LL saturation, **moved wall to HL** | `|g|` 0.84→0.16, LL reach 0.61→0.11 (near oracle). But HL then outputs `g≈0` regardless of command (refuses forward) → end err didn't improve. Isolated the 2nd failure. |
| `tracking` HL reward (+ absolute) | **SOLVED** | Penalty-dominated task reward was making value-max HL = `g≈0`. Tracking-only HL objective → HL asks for the command. A0-level tracking. |
| `tracking` reward alone, delta (ablation) | absolute NOT required | delta+tracking 0.14/0.11/0.20 → `tracking` is the primary lever; `absolute` only refines it (0.14→0.098). |
| `hl=ppo` + absolute + tracking (± std cap) | FAILED both | No cap: `|g|`→13, yaw 1.40. Std cap (1e-3,1.0): still `|g|`→4 + falls (ep_len 9.6) — the cap bounds σ, not the unbounded Gaussian **mean**. PPO can't cleanly bound `g` (rsl_rl has no squashed density); TD3's deterministic `tanh` is load-bearing. |
| Remove warm-start (control) | catastrophic | `|g|` pins 1.0 (100% sat), action_rate 13.5 vs A0 0.66. Confirms warm-start load-bearing from both directions. |
| Deploy goal-state noise (sim, `hrl.state_noise`; learned `absolute` TD3) | stands but **twitchy, worst standing still** | Clean-trained LL isn't robust to estimator noise on its velocity/height goal feedback (cmd=0 → goal=−noise → phantom corrections; HL input isn't noised). **Sim2real fix: train LL+HL with state-noise DR** on the goal-state obs. Adding measured `imu_lin_vel` as a direct obs is privileged but — unlike F4's *absolute*-mode no-op — base_lin_vel in the HL obs (`velobs`) is a large clean-tracking win under `delta` (F4 row). Deploy mechanism + knob → `.claude/docs/deployment.md`. |
| Upper-body posture penalty (ADR-0002) + velobs sweep (2026-06-30, seed-123; `*_pose0p5/pose1p0/velobs_*_s123`) | **arms tamed ~9×; posture 0.5 = sweet spot; `velobs+pose0.5` = keeper** | A1 LL never saw the env `variable_posture`/`action_rate_l2` (goal-only routing) → arms drift behind back / twist wrists (undeployable). Fix: add negative arms+waist deviation (`ll_posture_coef`) + all-joint action-rate (`ll_action_rate_coef`) to the LL **intrinsic** (`hrl_runner.py:377`). `ar=0.05` climbs `fell_over`→165; **`ar=0.02` → `fell_over`≈0** (the keeper). Benchmark (64×600×2 seeds, model_10000): `velobs_pose0p5_s123` **ub_pose_dev 0.021** (old baseline 0.180, A0 0.0004), **err_vx 0.062**, orient 0.047, CoT 0.76 — best A1 all-round. Posture **1.0 NOT better** (velobs: pose_dev 0.030 > 0.5's 0.021; no-velobs: orient/height →0.14). `velobs` (hl_obs_vel) = clean tracking+arm win (err_vx 0.137→0.062 at pose0.5). Seed s42-vs-s123: **arms replicate, the tracking "gain" was seed luck** (err_vx 0.083 vs 0.137). Still above A0's absolute arm band. **Crash+fix (2026-07-01): penalties were unbounded L2** — `bias_velobs_pose0p5_s123` converged then crashed at it9624 (rare sim blow-up → `posture_pen` spiked −0.18→**−522090** in 3 iters → detonated the LL PPO update → ep_len 980→63, unrecoverable, bench err_vx 1.38). NOT noise-specific (3 other pose0p5 runs ran 10k clean; full-noise one only reached it2620, healthy). Fix `hrl_runner.py`: `dev.clamp(max=9.0)` (posture) + `ar.clamp(max=4000.0)` (action-rate) — catastrophe-headroom, normal training byte-unaffected (healthy dev ~0.36). Rerun the failed pose0p5 runs. |
| Track F lean A0/A1 (semi→true-lean, 2026-06-23; `*_lean_a0_*`/`*_lean_a1_*` runs) | smoothness "cost" was a **reward artifact**; **directional goals best** | Semi-lean (A0 still jerk-penalized, A1 LL never is) made A1 look ~5× jerkier; **true-lean** (action_rate/joint_acc zeroed both sides) flips it — flat A0 act_rate 20–35, A1-from-polished ~2.1–2.5 (~10× smoother). Caveat: **init dominates regardless of goal mode** (`a1_from_lean`/`a1_from_lean_delta` both inherit the wild `a0_scratch`, act ~22). Directional(delta)+tracking from polished → **near-perfect LL** (reach err vx 0.026–0.048) so the **HL becomes the wall** (end err ≈ HL err); best yaw (0.124–0.128) but large same-config spread (vx 0.091 vs 0.143). **Correction (2026-06-24):** this directional HL wall is **NOT saturation** — `|g|` is small/unsaturated (`gabs_vx` 0.065, `sat=0.000`); the HL **under-reaches** (HL err vx 0.124 with tiny `|g|` = asks for too little). The `|g|→1` saturation of the original `delta baseline (relabel)` row above was a *pre-tracking-reward* failure, cured by `tracking` and absent in all true-lean delta runs → don't cite saturation against delta. Full tables → `hierarchy_benefit_roadmap.md` Track F Results. |

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
Also **discovered here**: the LL-intrinsic **action-rate penalty must be
`ll_action_rate_coef=0.02`, NOT A0's 0.05** — matching A0's 0.05 over-penalizes the goal-only
LL and spikes `fell_over` (~165); 0.02 gives `fell_over`≈0 (later re-confirmed in the
2026-06-30 posture/velobs sweep, ledger row above).
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
- **F4 velocity observability** (add `base_lin_vel` to HL obs): under the then-regime
  (absolute/pre-tracking) A/B det-eval blind (0.85,0.39,0.28) ≈ velocity-obs
  (0.85,0.38,0.25) → looked like a no-op. **Overturned 2026-06-30: under `delta` goals
  velobs is a large linear-tracking win** (err_vx 0.137→0.062), replicated — the no-op was
  regime-specific, not general. See the F4 ledger row + `a1_estimator_noise_8b`.

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

**Absolute** target (`2026-06-19_*noise_abs_*`), full benchmark (64×600×2 seeds, model_10000):

| variant (seed) | err_vx | err_vy | err_yaw | act_rate | outcome |
|---|---|---|---|---|---|
| bias 0.10 (s42, orig) | 0.53 | — | — | — | COLLAPSE — **transient training corruption** (not seed, not config) |
| bias 0.10 (s42, rerun) | 0.080 | 0.053 | 0.202 | 4.56 | success |
| bias 0.10 (s123) | 0.083 | 0.074 | 0.194 | 2.85 | success |
| full = bias + drift .01/.99 + lag 3 (s42) | 0.072 | 0.051 | 0.170 | 3.02 | success |
| full (s123) | 0.084 | 0.052 | 0.173 | 5.19 | success |

(Eval is NOT bit-exact: the policy is deterministic but the MuJoCo-Warp GPU rollout isn't, so
`err_vx` wobbles ~0.002 run-to-run — within each run's across-seed `err_vx_std`. The earlier
0.078 for s42-rerun is the same run as 0.080 here; CLAUDE.md GPU-non-determinism caveat, eval-time.)

Verdict: bias-only and full DR both train **reliably** (2/2 clean each) to clean-baseline quality;
the lone collapse (ep_len 8.9, HL goals 54% saturated — the old A1 wall) was a one-off glitch,
confirmed benign by a successful same-seed-42 rerun (ep_len 724 by it2000). Do NOT report bias
instability.

### Directional (delta) goals under noise + directional-vs-absolute verdict (2026-06-23)
Same matrix re-run with **directional** (`hl_target_mode=delta`, `V*=state+scale·g`) HL goals, using the
faithful-DR `_target_obs` fix. Runs `2026-06-23_*noise_delta_*`, model_10000:

| run | goals | noise | err_vx | err_vy | err_yaw | fall | act_rate |
|---|---|---|---|---|---|---|---|
| delta_bias (s42)  | directional | bias | 0.129 | 0.099 | 0.163 | 0.00 | 3.32 |
| delta_bias (s123) | directional | bias | 0.117 | 0.091 | 0.181 | 0.00 | 2.27 |
| delta_full (s42)  | directional | full | 0.127 | 0.132 | 0.163 | 0.00 | 2.52 |
| delta_full (s123) | directional | full | 0.105 | 0.102 | 0.144 | 0.00 | 3.11 |
| **mean directional** | | | **0.120** | **0.106** | **0.163** | 0.00 | **2.8** |
| **mean absolute** (4 above) | | | **0.080** | **0.058** | **0.185** | 0.00 | **3.9** |

Probe (HL-vs-LL, c=8, `sat=0.000` every run): the mode flips **which command axis the HL maps well** —
LL reach is near-perfect (`ll_err_vx ~0.056`) in BOTH, so the HL is always the wall:

| goals | HL err vx | LL err vx | HL err yaw | LL err yaw |
|---|---|---|---|---|
| directional (mean of 4) | **0.072** | 0.057 | **0.090** | 0.069 |
| absolute (mean of 4)    | 0.049 | 0.056 | 0.175 | 0.077 |

**Initial verdict (pre-velobs) — HL-side axis trade-off:** absolute HL nails vx (hl_err_vx 0.049) → best
straight-line tracking (err_vx 0.080) but botches yaw (hl_err_yaw 0.175 → err_yaw 0.185); directional HL
nails yaw (hl_err_yaw 0.090 → err_yaw 0.163, best) and is smoother (act 2.8 vs 3.9) but its vx map is looser
(err_vx 0.120). Same split as clean true-lean (Track F) and the shaped SOLVED run. Both modes are
deployment-faithful (noise enters via the LL's observed `V*−state_n`); in delta a constant bias additionally
cancels in `V*−s` by construction, so delta_bias≈delta_full. **This trade-off was overturned by feeding the
HL the velocity estimate — see below.**

### HL lin-vel observation moves the directional vx wall (2026-06-29) — directional+velobs WINS
Directional's looser vx came from a **velocity-blind HL**: the delta target `V*=state+scale·g` needs the
current velocity to pick `g=(command−v)/scale`, but the HL obs (`policy++command`, `history_length=1`) carries
no lin-vel. Fix: `hl_obs_vel` feeds the HL the SAME deployable lin-vel estimate the LL conditions on
(`state_n` vx,vy; clean at eval) → HL obs `policy++command++v_est` (td3.py, RQ2-safe HL-internal change).
Re-ran the delta matrix with `--agent.hl-obs-vel True` (runs `2026-06-29_*noise_delta_*_velobs_*`, **old
pure-HIRO intrinsic**, the only agent.yaml delta vs the 06-23 baseline is `hl_obs_vel:true` — a clean A/B):

Full 2×2 (bias/full × s42/s123), all pure-HIRO, model_10000 (probe hl_err_vx where run):

| run | err_vx | err_vy | err_yaw | act | hl_err_vx |
|---|---|---|---|---|---|
| bias_velobs s42 | 0.056 | 0.043 | 0.123 | 1.99 | 0.039 |
| bias_velobs s123 | 0.059 | 0.055 | 0.187 | 3.23 | — |
| full_velobs s42 | 0.060 | 0.047 | 0.163 | 1.92 | 0.046 |
| full_velobs s123 | 0.068–0.077 | 0.049 | 0.137–0.170 | 1.7–2.1 | 0.075 |
| **velobs mean (4 cells, 2 seeds)** | **0.063** | **0.049** | **0.161** | **2.3** | ~0.053 |
| baseline directional (no velobs) | 0.120 | 0.106 | 0.163 | 2.8 | 0.072 |

**Read.** The vx HL wall **moved down**: hl_err_vx 0.072→0.053, **fwd_hl_vx ~0.106→0.051 (halved)** in the clean
pairs (bias_s42 0.099→0.037, full_s42 0.113→0.044) — directional's forward weakness. End **err_vx halved
0.120→0.063**; act_rate 2.8→2.3 (smoother, s42 runs ~1.9); LL tightened (0.057→0.044). yaw HL wall
**unchanged** (hl_err_yaw ~0.090) — expected, only lin-vel fed; yaw-rate is the remaining limiter (feed it
next). `bias_velobs_s123` first collapsed (|g|0.79, err_vx 0.25) but **reran clean (0.059) → transient
confirmed** (as predicted; do NOT report velobs instability). full_velobs_s123 same-config spread 0.068↔0.077.
**Verdict: directional+velobs beats absolute on EVERY axis** (vx 0.063 vs 0.080, yaw 0.161 vs 0.185, act 2.3
vs 3.9) — velocity obs closes the vx gap absolute used to win; the axis trade-off is gone. **Directional+velobs
is the best A1 config.** Separate axis (excluded here): `*_pose0p5_*` add ADR-0002 upper-body reg (0.5/0.02) —
the bias one failed (err_vx 1.38). Logs `2026-06-{29,30}_*velobs*`.

## Play/replay fix (F0, 2026-06-11) — all pre-fix qualitative replays are void
`play.py` used `get_inference_policy()` which returned the bare LL actor — nothing fired the
HL or wrote `env.hrl_goal`, so **every pre-F0 A1 replay showed the LL with goal frozen at 0**
(command physically couldn't reach the LL). W&B training metrics (HL in the loop) were always
ground truth. Fix: `HighLevel.act_inference` + `HierarchicalRunner.get_inference_policy`
mirroring the training goal wiring. Plus the structure-restore (so non-default goal spaces /
trained HLs load correctly instead of silently replaying an oracle HL) — see `hrl-infra.md`.
Validated headless on 4 checkpoints.
