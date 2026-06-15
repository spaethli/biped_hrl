# Deployment pipeline (H1-2)

## C++ controller

Binary: `deploy/robots/h1_2/build/h1_2_ctrl`

Setup scripts (**source them, don't execute**):

- `source setup_all_mujoco.sh` → `H1_2_DOMAIN_ID=1`, `NETWORK=lo`,
  `CFG=deploy.yaml` → run `h1_2_sim`
- `source setup_all_robot.sh` → `H1_2_DOMAIN_ID=0`, `NETWORK=enp11s0`,
  `CFG=deploy_real.yaml` → run `h1_2_real`

FSM states + keyboard: `i`=FixStand, `o`=Velocity/walk, `p`=Passive.
Velocity keys: `w/s`=fwd/bwd, `a/d`=strafe, `q/e`=turn.
Config: `deploy/robots/h1_2/config/config.yaml` (`keyboard_transitions`).
Observation assembly: `deploy/robots/h1_2/src/State_RLBase.cpp`
(`keyboard_velocity_commands`).

## Deploy configs

- Sim: `config/policy/velocity/v0/params/deploy.yaml` (keyboard_velocity_commands)
- Real: `config/policy/velocity/v0/params/deploy_real.yaml` (velocity_commands/joystick)

## ONNX export (the correct path)

```bash
python scripts/play.py Unitree-H1_2-Flat \
    --checkpoint-file logs/rsl_rl/h1_2_velocity/<run>/<model>.pt \
    --export-onnx
cp logs/.../policy.onnx deploy/robots/h1_2/config/policy/velocity/v0/exported/policy.onnx
```

`--export-onnx` is a bare flag; it calls `runner.export_policy_to_onnx()` and exits
without the viewer. Training saves also auto-export `policy.onnx` (with metadata)
next to each checkpoint via `VelocityOnPolicyRunner._export_policy_onnx`.

**A1/HRL caveat:** the A1 LL ONNX input is 96-dim = proprio(89) + GOAL(7), vs A0's
92-dim = proprio incl. COMMAND(3). For deployment the C++ must feed the goal delta
where A0 fed the command. Oracle HL: the goal is computable on-robot
(`oracle_target − state`); learned HL: the HL net must be exported and chained
(fires every `c` steps — see plan doc §8, deferred to the deployment milestone).

## Visualize a checkpoint

```bash
python scripts/play.py Unitree-H1_2-Flat \
    --checkpoint-file logs/rsl_rl/h1_2_velocity/<run>/<model>.pt \
    --num-envs 1
```

For A1 runs play.py restores the run's goal space / HL algorithm from its
`params/agent.yaml` automatically (check the `Restored run structure` line).
