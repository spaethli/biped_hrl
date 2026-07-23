#pragma once
// Robot-local observation terms (H1-2). Registered via REGISTER_OBSERVATION, i.e. by
// including this header anywhere in the binary; the shared terms in
// deploy/include/isaaclab/envs/mdp/observations/observations.h stay untouched (standing
// no-isaaclab-edits rule).
//
// Included by BOTH State_RLBase.cpp (A0) and State_RLHRL.cpp (A1) so registration does
// not depend on which translation units the linker happens to keep.

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "FSM/FSMState.h"
#include "isaaclab/envs/mdp/observations/observations.h"

namespace isaaclab
{

// Keyboard velocity commands — copied from deploy/robots/g1/src/State_RLBase.cpp.
// To activate: change "velocity_commands" → "keyboard_velocity_commands" in
// config/policy/velocity/v0/params/deploy.yaml (observations section).
//
// Key mapping (same as G1 and h1v2-Isaac convention):
//   w/s = forward/backward   a/d = strafe left/right   q/e = turn left/right
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // Here you can change the input velocity of the keyboard.
    // s/a/d/q/e clamped to the training command ranges (vx lo -0.5, vy/wz ±0.5) —
    // the old ±1.0 values were outside the trained distribution. Number keys = held
    // forward-speed presets for the deploy-gate battery (held-command tests); 0 = stop.
    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", { 1.0f,  0.0f,  0.0f}},
        {"s", {-0.5f,  0.0f,  0.0f}},
        {"a", { 0.0f,  0.5f,  0.0f}},
        {"d", { 0.0f, -0.5f,  0.0f}},
        {"q", { 0.0f,  0.0f,  0.5f}},
        {"e", { 0.0f,  0.0f, -0.5f}},
        {"0", { 0.0f,  0.0f,  0.0f}},
        {"1", { 0.1f,  0.0f,  0.0f}},
        {"2", { 0.2f,  0.0f,  0.0f}},
        {"3", { 0.3f,  0.0f,  0.0f}},
        {"4", { 0.4f,  0.0f,  0.0f}},
        {"5", { 0.5f,  0.0f,  0.0f}},
        {"6", { 0.6f,  0.0f,  0.0f}},
        {"7", { 0.7f,  0.0f,  0.0f}},
        {"8", { 0.8f,  0.0f,  0.0f}},
        {"9", { 0.9f,  0.0f,  0.0f}},
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    auto it = key_commands.find(key);
    if (it != key_commands.end()) cmd = it->second;
    return cmd;
}

// True if this deploy config takes its twist from the KEYBOARD (sim bridge) rather than
// the joystick (real robot). Handles both YAML layouts: A0 lists observation terms at the
// top level of `observations:`, A1 nests them in groups (`policy:` / `command:`).
// The `const` on every node here is LOAD-BEARING: yaml-cpp's non-const operator[] INSERTS
// an undefined child for a missing key, which mutates `observations` while the
// ObservationManager is iterating it (this term is evaluated from inside that loop) and
// throws YAML::InvalidNode on the next term. Const nodes use the read-only overload.
inline bool uses_keyboard_commands(ManagerBasedRLEnv* env)
{
    const YAML::Node obs = env->cfg["observations"];
    if (!obs) return false;
    if (obs["keyboard_velocity_commands"]) return true;
    for (auto g = obs.begin(); g != obs.end(); ++g) {
        const YAML::Node grp = g->second;
        if (grp.IsMap() && grp["keyboard_velocity_commands"]) return true;
    }
    return false;
}

// Last value returned by gait_phase_cmd (sin, cos) — i.e. exactly what the policy saw,
// post stand-mask. run() executes on the 1 kHz control loop while the observation is
// computed on the policy thread, so the flight recorder reads it from here (same
// cached-for-logging pattern as State_RLHRL's last_cmd_).
inline float g_gait_phase_obs[2] = {0.0f, 0.0f};

// Command-aware gait clock. Identical to isaaclab::mdp::gait_phase (same global_phase
// accumulator, same `period` param, same sin/cos, same <0.1 stand-mask) EXCEPT that the
// stand-mask reads whichever command term this deploy config actually uses.
//
// Why this exists (2026-07-21, defect 0 in doc/hrl/A1a_deploy_plan.md): the shared term
// hardcodes isaaclab::mdp::velocity_commands, i.e. the JOYSTICK. Under keyboard control
// the joystick is neutral, so cmd_norm ~ 0 on every tick and the term returned (0, 0)
// for the whole session — the policy's gait clock was dead in every sim bridge run.
REGISTER_OBSERVATION(gait_phase_cmd)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    // Not cached in a static: A0 and A1 are both constructed in one process, each with its
    // own deploy yaml, so the source is a per-env property. Two yaml lookups at 50 Hz.
    const auto cmd = uses_keyboard_commands(env)
        ? isaaclab::keyboard_velocity_commands(env, params)
        : isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    g_gait_phase_obs[0] = obs[0];
    g_gait_phase_obs[1] = obs[1];
    return obs;
}

} // namespace isaaclab
