# HRL implementation plan — master index

Thesis: 5 HRL locomotion architectures (A0–A4) for the Unitree H1-2, compared for
sim-to-real transfer. Built on `mjlab` + `rsl_rl` + MuJoCo-Warp.

This is the **navigation hub + status dashboard**. The architecture matrix, RQ2
comparison-cleanliness rule, and reproducibility rules live in
`.claude/docs/experiment-design.md` (not duplicated here).

## Status dashboard

| Arch | Task ID | Status | Headline |
|---|---|---|---|
| **A0** | `Unitree-H1_2-Flat` | ✅ v1 done; **v2=option B** (2026-07-09) | Flat PPO baseline. v1 benchmark ref err_vx/vy/yaw 0.09/0.11/0.10 (v1 env only). **Model v2 = option B** (torso 300/3 + arm hold, desired_kl=0.01, `docs/adr/0005`) invalidates v1 checkpoints; baseline retrains on cluster; gate = benchmark + v1/v2 cross-eval + C++ bridge full-vs-split. **WL-E 2026-07-17**: fric-0.1@kl0.01 control (`a0_v2_optB_fric0p1_kl01_s42`) trains ≥ optB (err_vx 0.083, 0 falls) → ADR-0005 correction 1 (fric stall) FALSIFIED; foot drag = training-side clearance shortfall, plant-independent; RULED 2026-07-17: nominal stays fric 0, fric 0.1 = sanctioned option for a later rebase window; audit follow-ups = WL-D arms 7-9 (`docs/adr/0005` Amendment 2). |
| **A1** | `Unitree-H1_2-Flat-A1` | ✅ **tracking wall SOLVED** (2026-06-16) | HIRO: PPO LL + TD3 HL + relabel. `absolute` target + `tracking` HL reward → **A0-level** err 0.098/0.091/0.167, 0 falls. Sim deploy done (two-ONNX C++ `State_RLHRL`); state-noise DR **validated** (#8b `GoalStateNoise`; abs_bias & abs_full both reliable, 2/2 clean, clean-baseline quality). **A1a HL-enrichment** (commanded stride period → cost-of-transport): lever **validated 2026-07-02**; **S1c machinery ✅ 2026-07-02** (HL period action dim + CoT window reward, tested 18/18); d(T) slow-band enabler **parked** (v5/v6/v7 negative, 2026-07-03; v8 repro control ✅ code exonerated); keeper LL = v2 `model_5000`, envelope (0.35, 1.0). **S3 ✅-qualified** (2026-07-03, `a1a_s3_cadence_hl`): frozen-LL learned cadence HL drives CoT below the fixed-0.6 clock at every speed (−8…−22%, 0 falls, gait_match 0.94, tracking held) → **mechanism GO**, but at `hl_cot_coef=1.0` it settled on a near-natural flat ~0.65 stride (not the map optimum); coef = S5's lever (deferred to the full arch). **Co-train = the A1a line (user 2026-07-13)**, coef keeper 0.2 (velgoal+kl01, v2 lineage); **stage D 2026-07-14/15: arms ✅** (per-joint posture weights, pose_dev 10× down). **WL-C 2026-07-16: the "held-command blocker" was an EVAL ARTIFACT — there is no HL hold degeneracy.** `--eval-cmd-vx` collapsed the twist ranges, which the HIRO goal scale was derived from → scale hit its 1e-3 floor → `V* ≈ s` → the goal channel was inert in every hold eval (A0 immune: no goal space). HIRO-relabel suspect falsified (`relabel_frac` 0.03–0.09). **Fixed (F2): the goal scale is now baked into the checkpoint** (`GoalSpace.freeze_scale` + save/load + ONNX metadata; play.py shim for old ckpts) — no retrain. **Tables (e)+(f) re-measured: every A1 run holds ≥ A0** (ss@0.5 0.030–0.048 vs A0 0.057; ss@1.0 0.039–0.097 vs 0.076; 0 falls), D1/D2 are the set's *best* holders (the artifact had inverted the ranking), and fix0p8 walks fine — so **A1 tracking is not a deploy risk and the only surviving A1-vs-A0 gap is ENERGY** (CoT 0.66–1.04 vs 0.537). **A1a cadence premise RE-READ and VALIDATED 2026-09-01:** the earlier "a pinned clock beats every HL-cadence policy on CoT" result held only at `hl_cot_coef=0.2`, which was calibrated on a CoT denominator that changed 2026-07-10 and was never re-swept (~40 runs, 51 days). Re-swept 0.2→20 on the current reward: the sign FLIPS (a larger coef now *lengthens* the stride) and the response is threshold-like (0.2→2 inert, 2→5 flips). At **coef 5** the HL picks an interior 0.775 s and beats A0 on **9/10 walking metrics** — **CoT 0.391 vs 0.507 (−23%)** at `err_vx` 0.61× and 0 falls, clearing **ADR-0004 S4's A1a half** (A0+energy control still unbuilt; 1 seed). Cap is between 5 and 10 (coef 10/20 hold `fall_rate` 0 yet run ~5× A0 `err_vx` — `fall_rate` alone does not detect it). **2026-09-05: `hl_cot_coef=5` + `rel_standing_envs=0.20` meets S4 in BOTH regimes** — walk `act_legs` 0.94× / CoT 0.82× / stride 1.00×, stand `act_legs` **0.67×** with touchdowns on the **128 floor**, 0 falls; unique among 11 arms of a `cot {5,7}`×`rse {0.10..0.20}` grid (`cot 7` never reaches the standing floor). ⚠ Two residuals: the standing **p95** channel stays 3.3-11× (unclosed in every arm ever measured), and the tracking win is **speed-specific** (`ss_err_vx` 4.5× A0 at cmd 0.25, 2.2× at 1.0, 0.92× at 0.5) because `CoT = energy/walked distance` pays the HL to overshoot (+0.152 m/s at cmd 0.25, `hl_err`>`ll_err`); opt-in fix `hl_cot_cap_commanded` implemented, arm pending. → `docs/adr/0004` Amendments + the A1a experiment journal (research KB) 2026-09-01/09-05. |
| **A2** | `Unitree-H1_2-Flat-A2` | ⬜ next — **spec ready** (`doc/hrl/A2_ARMA.md`), now with a **hardware-measured motivation** (2026-08-03) | HIRO + A-RMA (privileged latent z + 1D-CNN adaptation + Phase-3 fine-tune). The H1-2's posture latent is neither single-frame observable (memoryless DR failed on hardware) nor constant across sessions (offset sweep), so only online estimation remains → `A2_ARMA.md` "Hardware motivation". |
| **A3** | `Unitree-H1_2-Flat-A3` | ⬜ planned | NaviGait: offline gait library + RL residual. |
| **A4** | — | ⏸ likely skipped | Trajectory + MPC. Lowest priority. |

## Where things live

- **Shared HRL machinery** (co-train loop, goal space, warm-start, reward decomp,
  benchmark/probe tools, checkpoint/ONNX, gotchas) → **`.claude/docs/hrl-infra.md`**.
  A2/A3 reuse this; read it before starting a new architecture.
- **A2 design spec** (HIRO + A-RMA: e_t, 3 phases, S0–S5, force-perturbation extension) → `doc/hrl/A2_ARMA.md`.
- **A1 design (as-built) + current status + open ablations** → `doc/hrl/A1_HIRO.md`.
- **Exploratory history is NOT in this repo (since 2026-07-28).** The A1 findings
  ledger, the goal-achievability probe writeup, the A1a experiment/deploy
  journals, the hierarchy-benefit roadmap (the ideas backlog answering the
  supervisors' "what does the hierarchy buy?" critique), and the closed-workline
  verdicts and delivered hand-off prompts — every lever tried, what failed, root
  causes, dated session logs and full result tables — live in the author's
  separate research knowledge base, so this repo stays publishable. `doc/hrl/`
  keeps only current design and status: the options in use and how the working
  solution is implemented. Don't re-accumulate narrative here; route it to the
  knowledge base instead.
- **A1a HL-enrichment plan** (the chosen Track-A answer: HL minimizes cost-of-transport via a
  commanded gait cadence the flat command can't express; staged S0–S6) →
  `doc/hrl/A1a_plan.md` (design rationale: `docs/adr/0004-a1a-energy-cadence-hl-enrichment.md`).
- **A0 model delta vs upstream `unitree_rl_mjlab`** (every actuator/plant difference
  from the fork point, ranked by plausible causal weight for the first successful
  hardware deployment; verified via direct `git diff` against `upstream/main`, not
  reconstructed from memory) → `doc/hrl/A0_model_delta_vs_upstream.md`.
- Architecture matrix, RQ2 rule, reproducibility → `.claude/docs/experiment-design.md`.
- File paths / class map / obs dims → `.claude/docs/codebase-map.md`.
- Cluster (SLURM) → `.claude/docs/cluster.md`. C++ deploy/ONNX → `.claude/docs/deployment.md`.

## Building a new architecture (A2/A3)

Each registers as a new task ID `Unitree-H1_2-Flat-A<N>` with its own
`config/h1_2_a<N>/{__init__,env_cfgs,rl_cfg}.py` under `src/tasks/velocity/`. Reuse
`hrl-infra.md` machinery; deviate from A0's env only via explicit, called-out structural
changes (RQ2 rule). Each architecture gets a `doc/hrl/A<N>_*.md` design+status doc that links
back to `hrl-infra.md` rather than re-deriving the shared parts.

## Process (no vibe coding)
Per CLAUDE.md: analysis → feature spec → approval → implement → test, with explicit
go-ahead between stages. Judge policies by the deterministic benchmark, not training err.
