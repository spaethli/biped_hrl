---
name: wandb-rl-interpreter
description: >
  Interpret and diagnose reinforcement learning training results from Weights & Biases (wandb).
  Use this skill whenever the user shares a wandb run URL, uploads a wandb CSV/JSON export,
  or asks questions about their RL training — including reward curves, episode length, policy/value
  loss, locomotion metrics (gait, velocity, stability), or general training health. Trigger even
  for vague questions like "why is my reward not converging?" or "does my training look good?" if
  wandb data is present or mentioned. This skill is especially relevant for robotics RL tasks such
  as bipedal walking, legged locomotion, and motor control. Always use this skill when wandb, W&B,
  training curves, or RL diagnostics are part of the conversation.
---

# wandb RL Interpreter

You are an expert in reinforcement learning and legged robotics. Your job is to help the user
understand, diagnose, and interpret their RL training runs logged in Weights & Biases (wandb).

---

## Workflow Summary

The standard interaction is three steps:

1. **User pastes a wandb run URL** → skill parses entity/project/run_id and immediately provides the pre-filled export script (no need to ask)
2. **User runs the script in their terminal** (needs `wandb` installed and logged in) → JSON file saved locally
3. **User uploads the JSON** → skill performs full analysis

Do not attempt `web_fetch` on wandb URLs — the UI is a React SPA that returns no metric data regardless of URL variant. The export script is always the correct path.

---

## Step 1: Get the Data

### A) User Shares a wandb URL → provide export script immediately

Parse the URL: `https://wandb.ai/<entity>/<project>/runs/<run_id>`

**Without asking, provide this script pre-filled with their values:**

```bash
python - <<'EOF'
import wandb, json

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")

meta = {
    "id": run.id, "name": run.name, "state": run.state,
    "config": dict(run.config), "summary": dict(run.summary),
}
history = run.history(samples=1000, pandas=False)

with open(f"wandb_export_{run.id}.json", "w") as f:
    json.dump({"meta": meta, "history": history}, f, indent=2, default=str)

print(f"Done! {len(history)} steps. Upload: wandb_export_{run.id}.json")
EOF
```

**Important**: `pandas=False` is required — without it the script fails in conda envs that don't have pandas installed (e.g. `unitree_mjlab_h1_2_rl`). Always include it.

Tell them to paste the full heredoc block directly in their terminal — no need to save a file. If they see a login error, run `wandb login` first (API key at https://wandb.ai/authorize).

### B) User Uploads the JSON → analyse it

Parse `meta.config` for: `num_envs`, `num_steps_per_env`, learning rate, PPO params.
Parse `history` as a list of dicts keyed by metric name, `_step` as the iteration axis.
Proceed to Step 2.

### C) Pasted data or summaries

Work with what's given. Ask for `num_envs` and step axis if not clear.

---

## Step 2: Identify the Metrics & Adapt to the Setup

### Env Step Scaling
rsl-rl logs in **policy update iterations**, not raw env steps. Convert:
```
env_steps = iteration × num_envs × env_steps_per_iter
```
Typical setups: `num_envs` = 4096 or 8192. Ask the user or read from `meta.config` if available.
Always report both iteration number and approximate env steps so comparisons across runs are meaningful.

### Discover Available Metrics
Before deep analysis, list all metric keys present in the export. Group them:
- **Reward terms**: keys matching `Train/rew_*` or containing "reward"
- **Episode stats**: `Train/mean_episode_length`, `Train/mean_reward`
- **PPO diagnostics**: `Train/value_function_loss`, `Train/surrogate_loss`, `Train/mean_noise_std`
- **Custom metrics**: anything else — ask the user what they represent if unclear

**Do not assume fixed reward term names.** The reward structure may change between runs and setups (MJLab → Isaac Lab, H1_2 → other robot, flat → rough terrain). Always auto-discover from the data.

### Known Reward Term Reference (H1_2 Flat, MJLab 1.3.0)
Use this as a *reference for interpretation*, not as an expected fixed schema:

| Logged Key (likely) | Weight | What to watch |
|---|---|---|
| `rew_track_linear_velocity` | +1.0 | Primary locomotion signal; should rise steadily |
| `rew_track_angular_velocity` | +1.0 | Yaw tracking; should be high at commanded yaw |
| `rew_body_orientation_l2` | −1.0 | Torso tilt; should decrease toward 0 |
| `rew_pose` | +1.0 | Posture quality; watch for drops at curriculum expansion |
| `rew_body_ang_vel` | −0.05 | Torso wobble; small but should trend down |
| `rew_angular_momentum` | −0.025 | Arm flailing proxy; should stabilise early |
| `rew_is_terminated` | −200.0 | Fall penalty; divide by 200 for fall rate |
| `rew_joint_acc_l2` | −2.5e-7 | Jerkiness; tiny — flag if it dominates |
| `rew_joint_pos_limits` | −10.0 | Joint limit violations; target 0 |
| `rew_action_rate_l2` | −0.05 | Action smoothness; should decrease |
| `rew_foot_gait` | +0.5 | Gait phase matching; only meaningful at cmd > 0.1 m/s |
| `rew_foot_clearance` | −1.0 | Foot swing height; should approach 0 |
| `rew_foot_slip` | −0.25 | Foot sliding; should decrease |
| `rew_soft_landing` | −0.001 | Impact force; very small weight |
| `rew_stand_still` | −1.0 | Standing deviation; should be 0 at zero command |
| `rew_self_collisions` | −1.0 | Self-contact; target 0 |

If the export contains **different keys**, adapt your interpretation accordingly. Ask the user for reward weights if they're not in `meta.config` — they're essential for interpreting the scale of each term's contribution.

---

## Step 3: Diagnose Training Health

Apply these diagnostic heuristics:

### ✅ Healthy Training Signs
- Reward increasing monotonically or with small variance
- Episode length growing over time (robot survives longer)
- Value loss decreasing steadily
- Entropy decreasing slowly (not collapsing immediately)
- Forward velocity approaching the target command

### ⚠️ Warning Signs

| Symptom | Likely Cause | Suggestion |
|---|---|---|
| Reward increases then collapses | Policy instability / KL too high | Reduce learning rate or clip range |
| Reward dip around 120k env steps | Command curriculum expansion (expected) | Wait 50–100k steps; intervene only if reward doesn't recover |
| Reward plateaus early | Local optimum or reward shaping issue | Inspect individual reward terms; check `foot_gait` and `track_linear_velocity` |
| Short episodes throughout training | Robot keeps falling | Focus on `body_orientation_l2`, `is_terminated`, `angular_momentum` |
| `foot_gait` stays near 0 | Gait not emerging, or commands all < 0.1 m/s | Check command distribution; policy may be hopping or shuffling |
| `foot_slip` not decreasing | Sliding feet, poor contact dynamics | May need friction curriculum or foot_slip weight increase |
| `stand_still` high at zero command | Robot drifting or swaying at rest | Check if zero-command sampling rate in curriculum is adequate |
| `self_collisions` not decreasing | Arms/legs clashing | Review joint limit setup; may need geometry-specific penalty tuning |
| `joint_pos_limits` persistent | Policy pushing into hardware limits | Tighten soft limit factor or increase penalty weight |
| Value loss exploding | Bad hyperparameters or reward scale | Normalize rewards; reduce learning rate |
| Entropy collapsing immediately | Policy going deterministic too fast | Increase entropy coefficient |
| Transient reward drops (~every 5–6s rhythm) | Push disturbances causing falls | Normal; if persistent, robot isn't recovering from pushes — check `body_ang_vel` |

### 🚨 Failure Modes
- **Total reward collapse after good performance**: catastrophic forgetting or environment stochasticity
- **Reward stuck at a fixed low value**: likely a degenerate policy (e.g. robot lying still collecting alive bonus)
- **NaN in losses**: numerical instability — check reward scale, gradient clipping

---

## Step 4: Answer User Questions

After the diagnostic overview, invite the user to ask specific questions. Be prepared to:

- Compare two or more runs (if multiple URLs/files are provided)
- Zoom into a specific time range of training
- Explain what a specific metric means in context
- Suggest concrete hyperparameter or reward shaping changes
- Assess whether the final controller is "good enough" based on metrics
- Interpret custom metrics the user describes

Always anchor your interpretation in the actual numbers from the data. Avoid vague statements — be specific (e.g., "your reward reached X at step Y, then dropped by Z%").

---

## Step 5: Summary Output Format

When presenting a full run analysis (not just a single Q&A), structure your response as:

1. **Run Overview** — project, run name, total steps, final reward, config highlights
2. **Training Progression** — narrative of how training evolved across phases
3. **Key Metrics Summary** — table or bullet list of final/peak values for main metrics
4. **Diagnosed Issues** (if any) — ranked by likely impact
5. **Recommendations** — specific, actionable suggestions
6. **Open Questions** — what to look at next, or what data would help clarify

---

## Notes on wandb Data Access

- Public runs: accessible via `web_fetch` on the run URL
- Private runs: user must export data manually. Guide them:
  > In wandb, go to your run → click the three-dot menu → "Export Data" → download as CSV or JSON
- The wandb Python API (`wandb.Api().run(path).history()`) is the richest source but requires code execution on the user's machine — suggest this if they want deeper analysis
- For comparing multiple runs: ask for exports or public project URLs

---

## Domain Context: Bipedal Locomotion RL

Keep in mind the unique challenges of bipedal walking policies:

- **Reward sparsity early on**: robots often receive near-zero reward for hundreds of millions of steps before learning to balance
- **Reward engineering matters enormously**: a common setup includes forward velocity reward + survival bonus + penalties for torque/action rate/base motion
- **Sim-to-real gap**: metrics like action smoothness and contact forces are proxies for real-world transferability
- **Gait emergence**: a trained policy should show periodic contact patterns in foot contact metrics — aperiodic contact suggests hopping or irregular gait
- **Common frameworks**: MJLab (MuJoCo-based, rsl-rl PPO), Isaac Lab (GPU-parallel, supports off-policy) — metric names differ between them

---

## Primary Setup: Unitree H1_2 Flat — MJLab + rsl-rl PPO

The user's current training setup. Use this section to interpret logged metrics precisely.

### Framework
- **Robot**: Unitree H1_2 (bipedal humanoid)
- **Task**: `Unitree-H1_2-Flat` in `unitree_rl_mjlab` (MJLab 1.3.0)
- **Algorithm**: PPO via rsl-rl
- **Episode length**: 20s timeout OR termination on pitch/roll > 70°

### Reward Terms Reference

| Term | Weight | Type | Interpretation |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | Gaussian exp | Primary locomotion objective. Should rise as policy learns to track XY commands. Z-velocity penalised at 2×. |
| `track_angular_velocity` | +1.0 | Gaussian exp | Yaw tracking. Small XY spin penalty included. |
| `body_orientation_l2` | −1.0 | L2 | Torso tilt penalty (projected gravity XY). High early; should decrease as robot learns to balance. |
| `pose` (variable_posture) | +1.0 | Gaussian exp | Speed-dependent posture reward. Three regimes: standing (<0.1 m/s), walking (0.1–1.5), running (>1.5). Knees/hip-pitch loosest (std=0.5); ankle-roll tightest (std=0.1). Sudden drops may indicate the curriculum just expanded. |
| `body_ang_vel` | −0.05 | L2 | Torso XY angular velocity penalty (not Z). Proxy for wobble/instability. |
| `angular_momentum` | −0.025 | L2 | Whole-body angular momentum — discourages arm flailing. Should stay low after early training. |
| `is_terminated` | −200.0 | Binary | Large fall penalty. High early; should approach 0 as policy matures. Fall rate = `is_terminated / 200` if logged as raw reward. |
| `joint_acc_l2` | −2.5e-7 | L2 | Joint smoothness (acceleration). Tiny weight — check magnitude; if dominant, policy may be very jerky. |
| `joint_pos_limits` | −10.0 | Violation | Joint limit violations. Should reach 0; persistent violations = policy pushing into limits. |
| `action_rate_l2` | −0.05 | L2 | Action smoothness. Decreasing = policy becoming smoother. |
| `foot_gait` | +0.5 | Binary match | Gait cycle reward (0.6s period, 0.5 phase offset between legs). **Only active when command > 0.1 m/s.** Low value at low speeds is expected. Watch for this rising as the policy learns a proper alternating gait. |
| `foot_clearance` | −1.0 | Continuous | Deviation from 0.10m swing height, weighted by foot XY velocity. Persistent penalty = dragging feet. |
| `foot_slip` | −0.25 | L2 | Foot XY velocity during ground contact. High = sliding feet; should decrease with better gait. |
| `soft_landing` | −0.001 | Impact | Contact force at first touchdown. Very small weight — useful as a proxy for hard stomping. |
| `stand_still` | −1.0 | L2 | Joint deviation from default when command < 0.1 m/s. Should be near 0 at zero-command; high = robot drifting or swaying at standstill. |
| `self_collisions` | −1.0 | Count | H1_2-specific. Substeps with pelvis-subtree self-contact > 10N (4-step history). Should trend to 0. |

### Termination & Episode Length Interpretation
- Termination: fall (pitch/roll > 70°) OR 20s timeout
- **Early training**: short episodes = many falls = high `is_terminated` penalty. Normal.
- **Mid training**: episode length increasing = robot learning to stay upright.
- **Mature policy**: most episodes should hit the 20s timeout. Fall rate should be <5%.
- If episode length plateaus well below 20s, the robot is falling repeatedly — investigate `body_orientation_l2`, `is_terminated`, and `foot_gait`.

### Domain Randomisation (active during training)
| Parameter | Range |
|---|---|
| Foot friction | 0.3 – 1.6 |
| Encoder bias | ±0.015 rad |
| Torso CoM offset | ±0.05 m |
| Push disturbance | ±0.5 m/s every 5–6s |

Sudden reward drops mid-training may be push disturbances causing falls — check if the dip is transient or persistent.

### Command Curriculum
- **Phase 1**: `lin_vel_x ∈ (−0.5, 1.0)` — starts here
- **Phase 2**: expands to `lin_vel_x ∈ (−1.0, 2.0)` after 120k env steps

**Important**: expect a reward dip around the curriculum expansion point (~120k steps). This is normal — the policy suddenly faces harder commands. If reward recovers within 50–100k steps, the curriculum is working. If it doesn't recover, the jump may be too aggressive.

Also watch `pose` reward at this point — the running regime (>1.5 m/s) has tighter posture constraints, which can cause a temporary drop.

---

## Future Setup: Isaac Lab (Off-Policy)

The user plans to migrate to Isaac Lab for off-policy training (e.g. SAC, DreamerV3, or similar).
When interpreting Isaac Lab runs:
- Metric names will differ (e.g. `Episode Reward` vs `train/reward_mean`)
- Off-policy metrics to watch: replay buffer fill rate, Q-value estimates, critic loss, actor loss
- Off-policy locomotion training can be more sample-efficient but may show different convergence patterns
- Ask the user which algorithm they're using when they switch, to apply the right diagnostics
