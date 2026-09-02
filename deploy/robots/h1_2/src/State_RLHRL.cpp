#include "FSM/State_RLHRL.h"
#include <algorithm>
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "h1_2_observations.h"  // robot-local terms: keyboard_velocity_commands, gait_phase_cmd

#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

// h1_2_limits.h is included unconditionally (not just under SAFETY_FILTER) since
// 2026-08-03: the 1 kHz base-velocity integrator in run() needs H1_2_CONTROL_DT.
#include "h1_2_limits.h"

// Deployable base-state estimator geometry (IMU lever arm, waist yaw, leg FK). Kept in a
// header so test/base_state_estimator_test.cpp can pin it against MuJoCo's own kinematics
// and against the play.py --check-vel-increment reference.
#include "hrl/base_state.h"

using hrl::base_vel_increment;
using hrl::kGravity;
using hrl::kImuOffsetT;
using hrl::lowest_foot_z;
using hrl::rz;

State_RLHRL::State_RLHRL(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // H1_2_DEPLOY_CFG selects sim (keyboard) vs real (gamepad) params (same as State_RLBase).
    const char* cfg_env = std::getenv("H1_2_DEPLOY_CFG");
    std::string deploy_cfg = cfg_env ? cfg_env : "deploy.yaml";
    spdlog::info("[HRL] Using deploy config: {}", deploy_cfg);
    YAML::Node deploy_yaml = YAML::LoadFile(policy_dir / "params" / deploy_cfg);
    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    // ADR-0006 Spec B: optional per-joint encoder-zero correction, absent = 27 zeros (no-op).
    if (deploy_yaml["joint_offset"])
        articulation->joint_offset = deploy_yaml["joint_offset"].as<std::vector<float>>();
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_yaml, articulation);

    // ONNX live here; loaded lazily on first enter() (see ensure_models_loaded).
    exported_dir_ = policy_dir / "exported";

    // HRL structure from deploy.yaml (`hrl:` block) — switch checkpoints by editing this,
    // no recompile. Must match how the policy was trained.
    auto hrl = env->cfg["hrl"];
    c_ = hrl["c"].as<int>();
    hl_target_mode_ = hrl["hl_target_mode"].as<std::string>();
    // hl_algorithm: "oracle" -> analytic V* (command + nominal), no high_level.onnx;
    // anything else (default) -> learned HL from high_level.onnx. Default keeps old behavior.
    oracle_ = hrl["hl_algorithm"] && hrl["hl_algorithm"].as<std::string>() == "oracle";
    auto goal_components = hrl["goal_components"].as<std::vector<std::string>>();
    nominal_h_ = hrl["nominal_root_height"].as<float>();
    goal_space_ = std::make_unique<hrl::GoalSpace>(goal_components, nominal_h_);
    target_ = Eigen::VectorXf::Zero(goal_space_->dim());

    // Keeper structure (2026-07-14): velocity-goals-only HL, HL lin-vel input, HL-owned
    // cadence. All default off -> pre-velgoal checkpoints run byte-identical.
    hl_obs_vel_ = hrl["hl_obs_vel"] && hrl["hl_obs_vel"].as<bool>();
    // WP5d: the HL additionally reads z_hat = phi(history) from adapt_encoder.onnx. Absent
    // key = false = byte-identical to the pre-WP5d path (no third session, no history).
    hl_obs_e_ = hrl["hl_obs_e"] && hrl["hl_obs_e"].as<bool>();
    velgoal_ = hrl["hl_velocity_goals_only"] && hrl["hl_velocity_goals_only"].as<bool>();
    const bool cadence = hrl["hl_cadence"] && hrl["hl_cadence"].as<bool>();
    const std::string cad_src =
        hrl["hl_cadence_source"] ? hrl["hl_cadence_source"].as<std::string>() : "hl";
    cadence_dim_ = (cadence && cad_src == "hl" && !oracle_) ? 1 : 0;
    if (cadence) {
        auto r = hrl["cadence_period_range"];
        period_lo_ = r[0].as<float>();
        period_hi_ = r[1].as<float>();
    }
    pin_period_ = hrl["pin_period"] ? hrl["pin_period"].as<float>() : 0.0f;
    if (hl_obs_vel_ || velgoal_) goal_space_->task_dim();  // enforce velocity-first prefix
    // The LL entrains to its gait-clock obs. The C++ term integrates
    // global_phase += dt/period from its YAML params node, which aliases env->cfg
    // (yaml-cpp handles share storage) — so writing this key retunes the clock
    // phase-continuously, exactly like training's hrl_period buffer. Pin now if asked;
    // the HL rewrites it per window unless pinned (see policy_step).
    // Resolve the term name once: live configs use the robot-local `gait_phase_cmd`
    // (2026-07-21 defect-0 fix), older fixtures still name the shared `gait_phase`.
    // Writing a key that isn't the one the obs manager bound would CREATE it and the
    // cadence would silently stop reaching the clock, so bind the node, don't guess.
    // `const` probe first (yaml-cpp's non-const operator[] would INSERT the missing key);
    // then take the writable node, since the period is written back through it.
    const YAML::Node pol_ro = env->cfg["observations"]["policy"];
    const char* phase_key = pol_ro["gait_phase_cmd"] ? "gait_phase_cmd"
                          : pol_ro["gait_phase"]     ? "gait_phase" : nullptr;
    if (phase_key)
        phase_params_ = env->cfg["observations"]["policy"][phase_key]["params"];
    if (!phase_params_)
        throw std::runtime_error("[HRL] deploy yaml has no gait_phase_cmd/gait_phase term "
                                 "under observations.policy — the LL clock is unbound.");
    if (pin_period_ > 0.0f)
        phase_params_["period"] = pin_period_;
    // Optional goal-state noise (default off) to emulate real-robot estimator noise in sim.
    state_noise_std_ = goal_space_->noise_std(hrl["state_noise"]);
    if (state_noise_std_.cwiseAbs().sum() > 0.0f) {
        std::string s;
        for (int i = 0; i < state_noise_std_.size(); ++i)
            s += std::to_string(state_noise_std_[i]) + " ";
        spdlog::warn("[HRL] goal-state noise ENABLED (per-dim std): [{}] — emulating "
                     "estimator noise; clear hrl.state_noise for clean runs", s);
    }

    // Deployable base-state estimator (2026-08-03; see FSM/State_RLHRL.h). Absent keys keep
    // the SportModeState path, so every pre-2026-08-03 config runs byte-identically.
    vel_from_imu_ = hrl["base_vel_from_imu"] && hrl["base_vel_from_imu"].as<bool>();
    height_from_fk_ = hrl["base_height_from_fk"] && hrl["base_height_from_fk"].as<bool>();
    hl_vel_from_leg_odom_ =
        hrl["hl_vel_from_leg_odom"] && hrl["hl_vel_from_leg_odom"].as<bool>();

#if SAFETY_FILTER
    // FAIL-CLOSED estimator selection (docs/adr/0007). There is deliberately NO default: a
    // missing key must abort, not silently pick a signal. Same class as the 2026-08-05
    // dead-signal splay, where absent keys routed the goal state to a dead topic and no
    // bridge gate could catch it. Every HRL deploy config carries the key; one that does not
    // will refuse to start, which is the intent.
    // (Outside SAFETY_FILTER the bench is compiled out entirely -- a passive arm's only
    // consumer is the flight recorder -- so est_arm_ stays 0, the shipped leg odometry.)
    if (!hrl["base_estimator"]) {
        std::string valid;
        for (int a = 0; a < 7; ++a) valid += std::string(a ? " | " : "") + kEstArmNames[a];
        throw std::runtime_error(
            "[HRL] deploy yaml has no `hrl.base_estimator` key. It selects which base-velocity "
            "estimator feeds obs[\"hl_vel\"] and has no default on purpose (docs/adr/0007). "
            "Set one of: " + valid);
    }
    const std::string arm_name = hrl["base_estimator"].as<std::string>();
    est_arm_ = -1;
    for (int a = 0; a < 7; ++a)
        if (arm_name == kEstArmNames[a]) est_arm_ = a;
    if (est_arm_ < 0) {
        std::string valid;
        for (int a = 0; a < 7; ++a) valid += std::string(a ? " | " : "") + kEstArmNames[a];
        throw std::runtime_error("[HRL] unknown hrl.base_estimator '" + arm_name
                                 + "'. Valid: " + valid);
    }
    spdlog::info("[HRL] base_estimator = {} (arm {}); the other six run passively and are "
                 "logged only", arm_name, est_arm_);
#endif

    // FAIL-CLOSED base-state source check (2026-08-05). "Absent keeps the old path" is a
    // safe default in the BRIDGE and a robot-breaking one on hardware: rt/sportmodestate
    // publishes identical zeros the moment our own low-level controller takes command, so
    // on the real robot the two keys above are not an option, they are the only source.
    // With both absent, the goal-space state read velocity 0 / height 0 and the height goal
    // delta became nominal_root_height - 0 = +1.3076 m against a 0.2 m goal scale; the
    // keeper splayed its legs within ~1 s of entering HRL mode, twice, and needed an
    // emergency stop. NO BRIDGE GATE CAN CATCH THIS CLASS -- in the bridge SportModeState
    // is live, so the identical config is correct there -- which is why the check is here,
    // at construction, and refuses to bring the binary up at all.
    // The sim-vs-real discriminator is the command source (keyboard = bridge, joystick =
    // robot), the same seam gait_phase_cmd already keys off. Checked per goal component,
    // not blanket: `orientation` comes from the IMU and needs neither estimator.
    const bool real_robot = !uses_keyboard_commands(env.get());
    const auto has_component = [&](const char* c) {
        return std::find(goal_components.begin(), goal_components.end(), c)
               != goal_components.end();
    };
    needs_highstate_ = (has_component("velocity") && !vel_from_imu_)
                    || (has_component("height") && !height_from_fk_);
    if (real_robot && needs_highstate_) {
        throw std::runtime_error(
            "[HRL] REFUSING TO START: this is a real-robot (joystick) deploy config whose "
            "goal space needs base velocity/height, but hrl.base_vel_from_imu / "
            "hrl.base_height_from_fk are not both set. rt/sportmodestate reads IDENTICAL "
            "ZEROS on the real H1-2 once this controller has command, so the goal-space "
            "state would be velocity 0 / height 0 and the height goal delta would be "
            "nominal_root_height (1.3076 m) against a 0.2 m scale. That splayed the robot's "
            "legs on 2026-08-05. Set BOTH keys to true in the `hrl:` block "
            "(deploy_real.yaml), or drop `velocity`/`height` from hrl.goal_components.");
    }
    // SECOND fail-closed source check, for the HIGH level (2026-08-06). The check above
    // covers the goal-space state `s`; it cannot see this one, because on a real robot `s`
    // can be perfectly well-formed while the HL's velocity input is still dead.
    // The HL reads an ABSOLUTE base velocity, and on hardware there are only ever two
    // candidates for it: rt/sportmodestate (identically zero, refused above) or the IMU
    // increment (`base_vel_increment(psi, w, 0, lev0) = 0` at every window start, and the HL
    // fires exactly at window starts -- so it reads 0.000000 at EVERY fire, by construction,
    // at every speed). Measured consequence on the bridge, same binary/plant/sequence with
    // the trip hold off: estimator 3/3 falls, privileged ground truth 0/3. On hardware that
    // input has been reading zero all along via the dead sportmodestate.
    // Leg odometry is the only live absolute source, so with hl_obs_vel on it is mandatory.
    // Scoped to real-robot configs like the check above, deliberately: the bridge must stay
    // able to run the no-leg-odom arm as the known-broken control in the A/B.
    if (real_robot && hl_obs_vel_ && !oracle_ && !hl_vel_from_leg_odom_) {
        throw std::runtime_error(
            "[HRL] REFUSING TO START: hrl.hl_obs_vel is set, so the high level reads an "
            "ABSOLUTE base velocity, but hrl.hl_vel_from_leg_odom is not set. On the real "
            "H1-2 that leaves no live source: rt/sportmodestate publishes zeros once this "
            "controller has command, and the IMU increment is 0 at every window start -- "
            "which is every HL fire. The HL would read `stationary` on every single "
            "decision, at every speed (bridge A/B: 3/3 falls vs 0/3 on ground truth). Set "
            "hrl.hl_vel_from_leg_odom: true in the `hrl:` block (deploy_real.yaml), or "
            "clear hrl.hl_obs_vel to run the velocity-blind HL the checkpoint was "
            "trained with.");
    }
    // sdk motor id -> articulation slot for the legs + waist. Everything else in this file
    // indexes by sdk id (joint_ids_map[i]); the FK and the waist encoder need the inverse.
    sdk_slot_.fill(-1);
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i) {
        const int jid = (int)env->robot->data.joint_ids_map[i];
        if (jid >= 0 && jid < (int)sdk_slot_.size()) sdk_slot_[jid] = i;
    }
    // Same encoder-zero correction the articulation applies on read, pre-resolved for the
    // legs + waist. run()'s leg odometry reads lowstate->msg_ directly (the articulation's
    // joint_pos is only refreshed by policy_step(), at 50 Hz, and finite-differencing that
    // staircase would re-create the exact aliasing this estimator lives in run() to avoid),
    // so it does not get the correction for free. joint_offset is indexed by articulation
    // SLOT, matching unitree_articulation.h's loop -- not by sdk id.
    leg_offset_.fill(0.0f);
    if (const auto jo = env->cfg["joint_offset"]) {
        const auto off = jo.as<std::vector<float>>();
        for (int k = 0; k < 13; ++k)
            if (sdk_slot_[k] >= 0 && sdk_slot_[k] < (int)off.size())
                leg_offset_[k] = off[sdk_slot_[k]];
    }
    // FK height at the deploy default pose. No longer an anchor for est_h (that is computed
    // directly in policy_step since 2026-08-04) -- kept as a STARTUP CROSS-CHECK, because
    // the gap between this pose and the goal space's training-referenced nominal is exactly
    // the error class that produced the +0.0325 m height bias and went unnoticed for a day.
    {
        float qn[12];
        for (int k = 0; k < 12; ++k)
            qn[k] = sdk_slot_[k] >= 0 ? env->robot->data.default_joint_pos[sdk_slot_[k]] : 0.0f;
        fk_h_nominal_ = -lowest_foot_z(qn);
    }
    // A large gap means the deploy hold pose sits far from the pose the policy was trained
    // around, which changes what the height goal column means. 0.028 m (the measured keeper
    // value) is expected and silent; 0.05 m is not.
    if (const float gap = nominal_h_ - (fk_h_nominal_ + hrl::kImuOffsetT.z());
        std::abs(gap) > 0.05f) {
        spdlog::warn("[HRL] height nominal gap {:.4f} m: hrl.nominal_root_height={:.4f} vs "
                     "FK-at-default_joint_pos+imu_lever={:.4f}. The deploy default pose is "
                     "far from the training nominal; check default_joint_pos and "
                     "nominal_root_height agree with the checkpoint.",
                     gap, nominal_h_, fk_h_nominal_ + hrl::kImuOffsetT.z());
    }

    // Ground-truth base velocity + height for the goal-space state (the LowState IMU has
    // neither). Provided by the MuJoCo sim bridge's SportModeState publisher
    // (rt/sportmodestate). SIM-ONLY privileged signal; on real hardware it reads ZEROS once
    // our own low-level controller has command — which is what the estimator above replaces.
    highstate_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
    spdlog::info("[HRL] hl_algorithm={} c={} hl_target_mode={} goal_dim={} velgoal={} "
                 "hl_obs_vel={} cadence_dim={} period_range=[{},{}] pin_period={} "
                 "base_vel_from_imu={} base_height_from_fk={} hl_vel_from_leg_odom={} "
                 "fk_h_nominal={:.4f} (HighState=rt/sportmodestate)",
                 oracle_ ? "oracle" : "learned", c_, hl_target_mode_, goal_space_->dim(),
                 velgoal_, hl_obs_vel_, cadence_dim_, period_lo_, period_hi_, pin_period_,
                 vel_from_imu_, height_from_fk_, hl_vel_from_leg_odom_, fk_h_nominal_);
    // The IMU rung reconstructs the WITHIN-WINDOW INCREMENT only, which is exactly what the
    // LL's `delta` goal needs and exactly what the HL's absolute lin-vel input does NOT get:
    // s[0..1] is 0 by construction at every window start, i.e. at every HL fire. Flag it —
    // on real hardware the SportModeState alternative already reads 0 there, so this is not
    // a regression, but in the bridge it IS a behavior change vs the privileged path.
    if (vel_from_imu_ && hl_obs_vel_ && !hl_vel_from_leg_odom_)
        spdlog::warn("[HRL] base_vel_from_imu + hl_obs_vel WITHOUT hl_vel_from_leg_odom: the "
                     "HL's (vx,vy) input is the within-window increment, which is 0 at every "
                     "HL fire step. Measured on the bridge as 3/3 falls vs 0/3 on ground "
                     "truth (2026-08-05). This is the known-broken control arm — intended "
                     "only for A/B attribution. Set hrl.hl_vel_from_leg_odom for the real "
                     "absolute source (a real-robot config is REFUSED without it).");
    if (hl_vel_from_leg_odom_)
        spdlog::info("[HRL] HL velocity from LEG ODOMETRY (stance-foot kinematics, "
                     "c-averaged over each window). The LL's goal delta is untouched and "
                     "still uses {} — the two velocity sources are deliberately split.",
                     vel_from_imu_ ? "the IMU increment" : "rt/sportmodestate");
    if (vel_from_imu_ && hl_target_mode_ != "delta")
        spdlog::warn("[HRL] base_vel_from_imu with hl_target_mode={}: the increment "
                     "substitution is only valid in `delta` mode, where the absolute "
                     "velocity cancels in V* - s. Expect wrong goals.", hl_target_mode_);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
    // Second layer of the 2026-08-05 fail-closed check, keyed off the DATA rather than the
    // config: enter() latches this when the goal state is sourced from SportModeState and
    // that topic is provably dead. It catches the one case the construction check above
    // cannot see -- a real robot started WITHOUT H1_2_DEPLOY_CFG, which silently loads the
    // keyboard/sim deploy.yaml and so looks like a bridge run to the config check.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return base_state_dead_; },
            FSMStringMap.right.at("Passive")
        )
    );
    // WP5d Layer 2. policy_step() latches this if a vector it built ever disagrees with the
    // ONNX-declared input size, instead of letting algorithms.h read past the end of it.
    // Bouncing to Passive rather than throwing is deliberate: the policy thread has no
    // exception handler, so a throw would kill the process and stop lowcmd entirely.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return dim_fault_; },
            FSMStringMap.right.at("Passive")
        )
    );
}

// One-time probe of an ONNX file's flattened input dim. OrtRunner::act builds its input
// tensor from the MODEL's size, so feeding a short obs vector reads out of bounds
// silently — this hard check at load is the only place a wiring mismatch fails loudly.
static int64_t onnx_input_dim(const std::filesystem::path& path)
{
    Ort::Env ort_env(ORT_LOGGING_LEVEL_ERROR, "hrl_dim_probe");
    Ort::SessionOptions so;
    Ort::Session session(ort_env, path.c_str(), so);
    auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t n = 1;
    for (auto d : shape) n *= d;  // [1, obs_dim]
    return n;
}

// One-time read of a comma-separated float list from an ONNX file's custom metadata (written
// by HierarchicalRunner._attach_hrl_metadata). Empty = key absent (pre-2026-07-16 exports).
// Robot-local, same throwaway-session pattern as onnx_input_dim: OrtRunner lives in the
// shared isaaclab header and keeps its Ort::Session private, so we open our own at load. Load
// path only — never the control loop.
static std::vector<float> onnx_metadata_floats(const std::filesystem::path& path, const char* key)
{
    Ort::Env ort_env(ORT_LOGGING_LEVEL_ERROR, "hrl_meta_probe");
    Ort::SessionOptions so;
    Ort::Session session(ort_env, path.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto val = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, alloc);
    std::vector<float> out;
    if (!val) return out;  // absent
    std::stringstream ss(val.get());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stof(tok));
    }
    return out;
}

// Scalar convenience over onnx_metadata_floats (the HL's hl_obs_e / hl_e_dim). `dflt` is
// returned when the key is absent, which is how a pre-WP5d export identifies itself.
static float onnx_metadata_float1(const std::filesystem::path& path, const char* key,
                                  float dflt)
{
    const auto v = onnx_metadata_floats(path, key);
    return v.empty() ? dflt : v[0];
}

// phi's entire baked contract, read through ONE session rather than eleven: the helpers above
// open a throwaway Ort::Session per key, which is fine for the HL's two but not for this.
// Load path only.
static hrl::AdaptEncoderMeta read_encoder_meta(const std::filesystem::path& path)
{
    Ort::Env ort_env(ORT_LOGGING_LEVEL_ERROR, "hrl_enc_meta");
    Ort::SessionOptions so;
    Ort::Session session(ort_env, path.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto md = session.GetModelMetadata();
    auto str = [&](const char* key) -> std::string {
        auto v = md.LookupCustomMetadataMapAllocated(key, alloc);
        return v ? std::string(v.get()) : std::string();
    };
    auto vec = [&](const char* key) {
        std::vector<float> out;
        std::stringstream ss(str(key));
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stof(tok));
        return out;
    };
    auto num = [&](const char* key) {
        const auto v = vec(key);
        return v.empty() ? -1 : (int)v[0];
    };
    hrl::AdaptEncoderMeta m;
    m.history_len  = num("phi_history_len");
    m.input_dim    = num("phi_input_dim");
    m.z_dim        = num("z_dim");
    m.time_order   = str("phi_time_order");
    m.input_layout = str("phi_input_layout");
    m.z_names      = str("z_names");
    m.scale        = vec("z_scale");
    m.center       = vec("z_center");
    m.clip_lo      = vec("z_clip_lo");
    m.clip_hi      = vec("z_clip_hi");
    m.cold         = vec("z_cold");
    return m;
}

void State_RLHRL::ensure_models_loaded()
{
    if (ll_runner_) return;  // already loaded (entered before)

    // The LL ONNX is always needed. The HL ONNX is needed only for a learned HL; the oracle
    // computes V* analytically (command + nominal), so it requires no high_level.onnx.
    const std::filesystem::path hl_path = exported_dir_ / "high_level.onnx";
    const std::filesystem::path ll_path = exported_dir_ / "low_level.onnx";
    if (!std::filesystem::exists(ll_path)) {
        throw std::runtime_error("[HRL] missing ONNX '" + ll_path.string() +
                                 "'. The A1 deploy needs low_level.onnx in exported/.");
    }
    ll_runner_ = std::make_unique<isaaclab::OrtRunner>(ll_path.string());

    // Input-dim guards (see onnx_input_dim). Obs sizes from a live compute.
    const auto obs = env->observation_manager->compute();
    const int policy_dim = (int)obs.at("policy").size();
    const int command_dim = (int)obs.at("command").size();
    const int64_t ll_in = onnx_input_dim(ll_path);
    ll_in_dim_ = ll_in;  // Layer-2 guard compares the BUILT vector against this every step
    if (ll_in != policy_dim + goal_space_->dim()) {
        throw std::runtime_error(
            "[HRL] low_level.onnx input dim " + std::to_string(ll_in) + " != policy(" +
            std::to_string(policy_dim) + ") + goal_dim(" +
            std::to_string(goal_space_->dim()) + ") — obs/goal_components mismatch.");
    }

    if (oracle_) {
        if (hl_obs_e_)
            throw std::runtime_error(
                "[HRL] hrl.hl_obs_e is set with hl_algorithm: oracle. The oracle computes V* "
                "analytically and has no network to read a latent -- phi would be loaded and "
                "never consumed. Pick one.");
        spdlog::info("[HRL] Oracle HL (analytic V*); loaded low_level.onnx from {}",
                     exported_dir_.string());
        return;
    }

    if (!std::filesystem::exists(hl_path)) {
        throw std::runtime_error("[HRL] missing ONNX '" + hl_path.string() +
                                 "'. A learned HL needs high_level.onnx in exported/ "
                                 "(or set hrl.hl_algorithm: oracle).");
    }
    hl_runner_ = std::make_unique<isaaclab::OrtRunner>(hl_path.string());

    // --- WP5d: the adaptation encoder, loaded and validated BEFORE the HL dim check, since
    // its z_dim is a term in that check. Everything geometric travels in phi's own metadata;
    // deploy.yaml contributes exactly one boolean.
    int z_dim = 0;
    if (hl_obs_e_) {
        const std::filesystem::path enc_path = exported_dir_ / "adapt_encoder.onnx";
        if (!std::filesystem::exists(enc_path)) {
            throw std::runtime_error("[HRL] hrl.hl_obs_e is set but '" + enc_path.string() +
                                     "' is missing. The H-adapt HL reads z_hat = phi(history) "
                                     "and cannot run without phi (or clear hrl.hl_obs_e).");
        }
        enc_meta_ = read_encoder_meta(enc_path);
        enc_runner_ = std::make_unique<isaaclab::OrtRunner>(enc_path.string());
        enc_in_dim_ = onnx_input_dim(enc_path);
        // The whole load-time decision is one pure function in hrl/adapt_encoder.h, so it can
        // be exercised offline (test/adapt_encoder_test.cpp) rather than only on a robot.
        const std::string bad = hrl::check_deploy_contract(
            enc_meta_, policy_dim, command_dim, enc_in_dim_,
            (long long)enc_runner_->get_action().size());
        if (!bad.empty()) {
            throw std::runtime_error(
                "[HRL] adapt_encoder.onnx: " + bad + ". phi's geometry and scales are POLICY "
                "properties and must travel baked into the export, never be re-derived at "
                "inference -- same rule as goal_scale, and for the same reason.");
        }
        z_dim = enc_meta_.z_dim;
        hist_.configure(enc_meta_.history_len, enc_meta_.input_dim);
        z_hat_ = enc_meta_.cold;
        std::ostringstream zs;
        for (size_t i = 0; i < enc_meta_.cold.size(); ++i)
            zs << (i ? ", " : "") << enc_meta_.cold[i];
        spdlog::info("[HRL] H-adapt ON: adapt_encoder.onnx H={} D={} -> z_hat({}) [{}]; "
                     "cold-start prior [{}] held for the first {:.2f} s of each entry.",
                     enc_meta_.history_len, enc_meta_.input_dim, z_dim, enc_meta_.z_names,
                     zs.str(), enc_meta_.history_len * env->step_dt);
    }

    // The HL's own metadata declares whether it was TRAINED to read a latent, so a yaml flag
    // that disagrees with the checkpoint is caught here rather than trusted. Unlike
    // goal_scale there is no legacy fallback: a pre-WP5d export simply must not be run with
    // the flag on, and a WP5d export must not be run with it off.
    const float meta_obs_e = onnx_metadata_float1(hl_path, "hl_obs_e", -1.0f);
    if (meta_obs_e >= 0.0f && ((meta_obs_e > 0.5f) != hl_obs_e_)) {
        throw std::runtime_error(
            std::string("[HRL] high_level.onnx was exported with hl_obs_e=") +
            (meta_obs_e > 0.5f ? "true" : "false") + " but deploy yaml says hrl.hl_obs_e: " +
            (hl_obs_e_ ? "true" : "false") + ". The checkpoint decides whether it reads a "
            "latent; fix the yaml.");
    }
    if (hl_obs_e_) {
        const int meta_z = (int)onnx_metadata_float1(hl_path, "hl_e_dim", -1.0f);
        if (meta_z > 0 && meta_z != z_dim) {
            throw std::runtime_error(
                "[HRL] high_level.onnx expects a latent of " + std::to_string(meta_z) +
                " dims but adapt_encoder.onnx emits " + std::to_string(z_dim) +
                " — the HL and phi come from different runs.");
        }
    }

    const int64_t hl_in = onnx_input_dim(hl_path);
    const int hl_in_expect =
        policy_dim + command_dim + (hl_obs_vel_ ? 2 : 0) + (hl_obs_e_ ? z_dim : 0);
    hl_in_dim_ = hl_in;  // Layer-2 guard compares the BUILT vector against this at each fire
    if (hl_in != hl_in_expect) {
        throw std::runtime_error(
            "[HRL] high_level.onnx input dim " + std::to_string(hl_in) + " != policy+" +
            "command" + (hl_obs_vel_ ? "+hl_vel(2)" : "") +
            (hl_obs_e_ ? "+z_hat(" + std::to_string(z_dim) + ")" : "") + " = " +
            std::to_string(hl_in_expect) +
            " — check hrl.hl_obs_vel / hrl.hl_obs_e in deploy yaml.");
    }

    // The exported HL emits the learned goal cols (all, or velocity-only) + the
    // period dim when it owns the cadence.
    const int hl_out_expect =
        (velgoal_ ? goal_space_->task_dim() : goal_space_->dim()) + cadence_dim_;
    if ((int)hl_runner_->get_action().size() != hl_out_expect) {
        throw std::runtime_error(
            "[HRL] high_level.onnx output dim " +
            std::to_string(hl_runner_->get_action().size()) + " != expected " +
            std::to_string(hl_out_expect) +
            " (check hrl.hl_velocity_goals_only / hl_cadence / goal_components).");
    }
    // Pin the g -> V* decode (V* = ref + scale*g) to what the HL TRAINED with, so that
    // `commands.base_velocity.ranges` in deploy.yaml is free to be whatever the operator
    // wants. Those ranges are the joystick safety clamp (isaaclab observations.h
    // `velocity_commands`); until now GoalSpace ALSO derived scale/center from them, so
    // narrowing them for safety (e.g. lin_vel_x -0.5..1.0 -> -0.25..0.5) silently halved
    // scale_vx 0.75 -> 0.375 and gutted the HL's goal authority. A0 has no goal space and was
    // never affected. See doc/hrl/A1_findings.md (WL-C) + .claude/docs/deployment.md.
    const auto meta_scale = onnx_metadata_floats(hl_path, "goal_scale");
    if (meta_scale.empty()) {
        spdlog::warn(
            "[HRL] high_level.onnx carries no 'goal_scale' metadata (pre-2026-07-16 export) "
            "-> falling back to deriving the goal scale from deploy.yaml "
            "commands.base_velocity.ranges. Those ranges MUST then equal the ranges the "
            "policy was TRAINED with, and narrowing them for safety WILL silently rescale "
            "the HL's goals. Re-export the policy (play.py --export-onnx) to fix.");
    } else {
        goal_space_->freeze_scale(
            Eigen::Map<const Eigen::VectorXf>(meta_scale.data(), (Eigen::Index)meta_scale.size()));
        std::ostringstream s;
        for (size_t i = 0; i < meta_scale.size(); ++i) s << (i ? ", " : "") << meta_scale[i];
        spdlog::info("[HRL] goal scale pinned from ONNX metadata [{}] — deploy.yaml command "
                     "ranges are the operator clamp only (safe to narrow).", s.str());
    }
    // `absolute` mode also decodes g against GoalSpace::center(), which stays range-derived
    // (its height column must remain this deploy's imu-site nominal, not training's pelvis z
    // — see goal_space.h). So the clamp is only free to move in `delta` mode.
    if (hl_target_mode_ == "absolute") {
        spdlog::warn("[HRL] hl_target_mode=absolute: the absolute target CENTER is still "
                     "derived from deploy.yaml commands.base_velocity.ranges, so those "
                     "ranges must equal the TRAINED ranges — do NOT narrow them for safety "
                     "with this mode (delta mode is unaffected).");
    }
    spdlog::info("[HRL] Loaded high_level.onnx + low_level.onnx from {}", exported_dir_.string());
}

void State_RLHRL::policy_step()
{
    env->robot->update();
    auto obs = env->observation_manager->compute();
    const auto& policy = obs.at("policy");    // proprio (command removed)
    const auto& command = obs.at("command");  // twist command (HL input only)
    const auto t_step0 = std::chrono::steady_clock::now();

    // WP5d: one history frame per CONTROL step (50 Hz), not per HL fire. phi's window is the
    // last H frames of `policy ++ command` -- the first H*D columns of the HL's own input --
    // pushed BEFORE the fire below, so the window ends on the current frame inclusive. The
    // sim-side Phase-2 rollout must slice its windows the same way; the layout and time order
    // are asserted against phi's metadata at load. No-op with the flag off.
    if (hl_obs_e_) hist_.push(policy, command);

    // Goal-space state s: base lin vel (world->body via the IMU quat) + ang_vel/orientation
    // (IMU) + height (sim HighState). frame_vel is published in the WORLD frame; rotate it.
    // FRAMES, CORRECTED 2026-08-04 (the previous note here claimed both came from the imu
    // SITE, and the gt_dvel below was built on that): the bridge publishes the two sensors
    // at DIFFERENT places --
    //   <framelinvel name="frame_vel" objtype="body" objname="pelvis"/>   (the 2026-07-15
    //   retarget documented in deployment.md)  ->  v_world is the PELVIS velocity
    //   <framepos    name="frame_pos" objtype="site" objname="imu"/>      ->  h_gt is the
    //   imu-SITE height
    // root_quat_w is the IMU (torso) attitude, so v_gt_b is the pelvis velocity expressed
    // in the TORSO frame: it needs the Rz(psi) waist rotation but carries NO lever arm.
    // (h1_2_handless.xml:331,340 in the bridge; the in-repo scene_h1_2.xml is not loaded.)
    const Eigen::Vector3f v_world = highstate_->velocity();
    const Eigen::Vector3f v_gt_b = env->robot->data.root_quat_w.conjugate() * v_world;
    const float h_gt = highstate_->position().z();

    // --- Deployable base-state estimate (2026-08-03) ---------------------------------
    // Mirrors the "waist-corr" rung of play.py --check-vel-increment. The pelvis velocity
    // behind the torso-mounted IMU is
    //     v_pelvis_P = Rz(psi) * (v_site_T - w_T x r)
    // and only the WITHIN-WINDOW increment of v_site_T is observable (integrated accel), so
    //     dv_pelvis = Rz(psi_i) * (dv_int - w_i x r) + Rz(psi_0) * (w_0 x r)
    // dropping the unmodellable [Rz(psi_i) - Rz(psi_0)] * v_site_T(t0) residual (small while
    // the waist barely turns inside one window; the bench's RMS already includes it).
    // dv_int is accumulated by run() at 1 kHz — NOT here at 50 Hz, where the same
    // reconstruction is ~4x worse from aliased foot-impact transients.
    const float psi = sdk_slot_[12] >= 0 ? env->robot->data.joint_pos[sdk_slot_[12]] : 0.0f;
    const Eigen::Vector3f w_T = env->robot->data.root_ang_vel_b;  // gyro, torso frame
    const Eigen::Vector3f lev = w_T.cross(kImuOffsetT);
    const bool window_start = (step_ % c_ == 0);
    Eigen::Vector3f dv;
    {
        std::lock_guard<std::mutex> lock(est_mu_);
        if (window_start) dv_.setZero();  // reset every c steps -> drift cannot accumulate
        dv = dv_;
        // Latch the leg-odometry mean for the window that just ended; the HL fires below
        // and consumes it. This is the bench's `fasthl_*` quantity exactly: the estimate
        // averaged across the window, delivered at the fire, lag included. A window that
        // accumulated nothing keeps the previous latch -- emitting 0 here would recreate
        // the very defect this path removes.
        if (window_start) {
            if (lo_n_ > 0) hl_vel_lo_ = (lo_sum_.head<2>() / (float)lo_n_);
            lo_sum_.setZero();
            lo_n_ = 0;
        }
    }
    if (window_start) {
        lev0_ = rz(psi, lev);
        // NO lever subtraction here (fixed 2026-08-04): v_gt_b is already the PELVIS
        // velocity (see the FRAMES note above), so `- lev` was removing a lever arm the
        // signal never had. Because est_vel carries the identical -(L_i - L_0) term, the
        // spurious one CANCELLED in est-minus-gt and the telemetry was silently scoring
        // the UNCORRECTED, site-frame reconstruction. That is why the bridge read
        // vy 0.205 against a corrected sim prediction of 0.026: 0.205 is the bench's
        // "lever arm (site-pelvis)" rung (0.227), not an estimator failure.
        gt_vel0_ = rz(psi, v_gt_b);
    }
    const Eigen::Vector3f est_vel = base_vel_increment(psi, w_T, dv, lev0_);
    // Lowest-foot FK. No contact sensing: unitree_hg::msg::LowState has no foot-force field.
    float qleg[12];
    for (int k = 0; k < 12; ++k)
        qleg[k] = sdk_slot_[k] >= 0 ? env->robot->data.joint_pos[sdk_slot_[k]] : 0.0f;
    // HEIGHT ANCHOR, CORRECTED 2026-08-04. est_h has to reproduce gt_h -- the imu-SITE
    // world height -- because when base_height_from_fk is on it REPLACES gt_h in the goal
    // state. Upright on a flat floor that height is just (pelvis above sole) + (torso->imu
    // lever), computed directly, with no anchor constant to get wrong:
    const float est_h = -lowest_foot_z(qleg) + hrl::kImuOffsetT.z();
    // The previous form was `nominal_h_ + (P(q) - P(q_def))`, which overloaded ONE constant
    // with TWO different jobs. `nominal_root_height` (1.3076) is the goal space's
    // TRAINING-referenced nominal -- training pelvis 1.03004 + 0.27756 -- and it must stay
    // that, because the deploy height delta cancels the frame offset against gt_h. The FK
    // anchor instead has to be the imu height at the DEPLOY `default_joint_pos`, a crouched
    // PD-hold pose (knee 0.5, hip pitch -0.2) whose pelvis sits at 1.00236. That 0.02768 m
    // pose gap was the dominant term in the +0.0325 m est_h-vs-gt_h bias measured on
    // 2026-08-03 -- reproducible to 0.6 mm across two completely different gaits, i.e.
    // static, not gait noise. GoalSpace is untouched here, so the gt_h path is unchanged.
    // Known residuals, deliberately NOT compensated: the lever is applied along world z
    // without rotating by base attitude (~1 mm at a 5 deg lean), and the FK's sole constant
    // (-0.045, the training capsule) differs from the BRIDGE's mesh foot by -0.013 m -- a
    // bridge modelling difference, not a robot one, so correcting for it here would be
    // wrong on hardware. Pre-registered: this fix alone should take the bias from +0.0325
    // to about +0.005 m; bar is <=5 mm on a cmd-0 stand.

    // Feed the policy the estimate only where the operator opted in; otherwise the
    // privileged bridge reading, byte-identical to the pre-2026-08-03 path.
    const Eigen::Vector3f lin_vel_b = vel_from_imu_ ? est_vel : v_gt_b;
    const float height = height_from_fk_ ? est_h : h_gt;
    // Cached for the flight recorder (run() doesn't otherwise see command/achieved-vel).
    // Deliberately the GROUND TRUTH, not `lin_vel_b`: the SafetyLogger's ach_* columns are
    // defined as the bridge's ground truth and are shared with A0's CSV — keep them comparable.
    last_lin_vel_b_ = v_gt_b;
    if (command.size() == 3) last_cmd_ = Eigen::Vector3f(command[0], command[1], command[2]);
    Eigen::VectorXf s = goal_space_->state(env.get(), lin_vel_b, height);

    // Emulate real-robot state-estimator noise on the goal feedback (no-op if std all 0).
    if (state_noise_std_.size() == s.size()) {
        std::normal_distribution<float> ndist(0.0f, 1.0f);
        for (int i = 0; i < s.size(); ++i)
            if (state_noise_std_[i] > 0.0f) s[i] += state_noise_std_[i] * ndist(rng_);
    }

    // High level fires at the start of each window -> new absolute target V* (held for c steps).
    if (window_start) {
        if (oracle_) {
            // Analytic target: commanded twist + nominal pose (no network).
            const Eigen::Vector3f cmd(command[0], command[1], command[2]);
            target_ = goal_space_->oracle_target(cmd);
        } else {
            std::vector<float> hl_in = policy;
            hl_in.insert(hl_in.end(), command.begin(), command.end());
            // The HL's absolute base velocity (td3._state_vec order: policy, command,
            // hl_vel). THE SPLIT, 2026-08-06: with hl_vel_from_leg_odom this is the ONLY
            // place leg odometry enters — `s` above keeps the IMU increment, so the LL's
            // goal delta V*-s is bit-identical to before. That asymmetry is deliberate and
            // load-bearing in both directions: leg odometry's error (0.074/0.049) is ~5x
            // the increment's (0.0139/0.0288) and would DEGRADE the LL, while the increment
            // is 0 at every fire and is useless to the HL. Training does exactly this split
            // — HlVelJitter is wired into obs["hl_vel"] and never into state_n.
            // Without the flag: the historical path, s[0:2], which under base_vel_from_imu
            // is 0 at every fire. Note V* = s + scale*g is anchored on the increment either
            // way, so the LL still only ever sees scale*g and cannot observe the swap.
            if (hl_obs_vel_) {
                const float vx = hl_vel_from_leg_odom_ ? hl_vel_lo_.x() : s[0];
                const float vy = hl_vel_from_leg_odom_ ? hl_vel_lo_.y() : s[1];
                hl_in.push_back(vx); hl_in.push_back(vy);
            }
            // WP5d: z_hat = phi(history), at the HL rate (every c steps, 6.25 Hz) -- never in
            // run(), which does not sustain 1 kHz. Until the buffer has filled, window()
            // refuses and z_hat_ stays at the baked cold-start prior; that refusal is what
            // makes "phi is never called on a partial buffer" executable rather than a
            // code-reading exercise.
            if (hl_obs_e_) {
                if (hist_.window(hist_win_)) {
                    if (!check_input("adapt_encoder.onnx", hist_win_.size(), enc_in_dim_))
                        return;
                    const auto t_phi0 = std::chrono::steady_clock::now();
                    const auto raw = enc_runner_->act({{"obs", hist_win_}});
                    const double phi_us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t_phi0).count();
                    bucket(phi_hist_, phi_us);
                    if (phi_us > phi_max_us_) phi_max_us_ = phi_us;
                    phi_count_++;
                    z_clipped_last_ = hrl::decode_latent(enc_meta_, raw, z_hat_);
                    z_clip_events_ += z_clipped_last_;
                    if (!z_valid_) {
                        z_valid_ = true;
                        spdlog::info("[WP5d] history full -> z_hat live after {:.2f} s "
                                     "(was holding the cold-start prior).",
                                     hist_.size() * (float)env->step_dt);
                    }
                } else if (!z_cold_cmd_warned_
                           && Eigen::Vector3f(command[0], command[1], command[2]).norm() > 0.1f) {
                    // Deliberately a warning and NOT a command block: refusing the operator's
                    // stick is a new failure mode on a live robot, and the WP7 protocol opens
                    // every cell with a 60 s stand, so the fill normally completes untouched.
                    z_cold_cmd_warned_ = true;
                    spdlog::warn("[WP5d] command given while the phi history is still filling "
                                 "({}/{} frames): the HL is running on the cold-start prior, "
                                 "not an estimate.", hist_.size(), enc_meta_.history_len);
                }
                hl_in.insert(hl_in.end(), z_hat_.begin(), z_hat_.end());
            }
            if (!check_input("high_level.onnx", hl_in.size(), hl_in_dim_)) return;
            const auto g_vec = hl_runner_->act({{"obs", hl_in}});
            const int gd = velgoal_ ? goal_space_->task_dim() : goal_space_->dim();
            const Eigen::VectorXf g = Eigen::Map<const Eigen::VectorXf>(g_vec.data(), gd);
            target_ = goal_space_->to_target(env.get(), s, g, hl_target_mode_, velgoal_);
            // HL-owned stride period: extra tanh dim -> affine map to the range, written
            // to the gait_phase clock for this window (phase-continuous; the term
            // integrates incrementally). Pinned clock was set once in the ctor.
            if (cadence_dim_ && pin_period_ <= 0.0f) {
                const float period =
                    period_lo_ + (g_vec[gd] + 1.0f) * 0.5f * (period_hi_ - period_lo_);
                phase_params_["period"] = period;
            }
        }
    }
    ++step_;

    // Low level observes the remaining delta V* - s and acts.
    const Eigen::VectorXf delta = target_ - s;
    std::vector<float> ll_in = policy;
    ll_in.insert(ll_in.end(), delta.data(), delta.data() + delta.size());
    if (!check_input("low_level.onnx", ll_in.size(), ll_in_dim_)) return;
    const auto action = ll_runner_->act({{"obs", ll_in}});
    env->action_manager->process_action(action);

    if (telemetry_.enabled()) {
        float ar = 0.0f;  // mean |Δaction| vs the previous policy step
        if (last_action_.size() == action.size()) {
            for (size_t i = 0; i < action.size(); ++i)
                ar += std::abs(action[i] - last_action_[i]);
            ar /= action.size();
        }
        last_action_ = action;
        // The live phase clock in every mode (yaml default / ctor pin / HL-written).
        const float period = phase_params_["period"].as<float>();
        // Regression guard: the deployable estimate next to the bridge's ground truth,
        // logged whether or not it is feeding the policy. Both velocity pairs are
        // WITHIN-WINDOW INCREMENTS of the pelvis-frame base velocity, so they are directly
        // comparable; gt_* is 0 on real hardware (dead rt/sportmodestate).
        // See the window_start block: `- lev` removed here too (2026-08-04). v_gt_b is the
        // pelvis velocity, so this is now the true pelvis-frame increment and the residual
        // est-minus-gt finally scores the waist-corrected estimator rather than cancelling
        // its correction away.
        const Eigen::Vector3f gt_dvel = rz(psi, v_gt_b) - gt_vel0_;
        const float est[3] = { est_vel.x(), est_vel.y(), est_h };
        const float gt[3]  = { gt_dvel.x(), gt_dvel.y(), h_gt };
        // Leg odometry next to its OWN reference. rz(psi, v_gt_b) is the absolute
        // pelvis-frame truth: v_gt_b is the pelvis velocity in the TORSO frame (the bridge
        // publishes framelinvel on body `pelvis`, framequat on the imu site), so it needs
        // the waist rotation and no lever arm — the same correction gt_dvel above carries.
        const Eigen::Vector3f v_gt_p = rz(psi, v_gt_b);
        const float lo[4] = { hl_vel_lo_.x(), hl_vel_lo_.y(), v_gt_p.x(), v_gt_p.y() };
        telemetry_.record(step_ * (float)env->step_dt, command.data(), s, target_, period,
                          env->robot->data.joint_pos[1], env->robot->data.joint_pos[7], ar,
                          est, gt, lo,
                          hl_obs_e_ ? z_hat_.data() : nullptr, z_valid_, z_clipped_last_);
    }

    // [WP5d] policy-thread budget. The [T5] counters all bucket inside run(), so nothing
    // measured THIS loop -- the one that actually runs the networks -- until now.
    {
        const double step_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t_step0).count();
        bucket(step_hist_, step_us, kStepBucketUs);
        if (step_us > step_max_us_) step_max_us_ = step_us;
        step_count_++;
    }
}

void State_RLHRL::run()
{
    // [T5] tick start. Bucketed at the end; nothing is printed from inside this loop.
    const auto t_tick0 = std::chrono::steady_clock::now();
    if (have_last_tick_)
        bucket(period_hist_,
               std::chrono::duration<double, std::micro>(t_tick0 - last_tick_).count());
    last_tick_ = t_tick0;
    have_last_tick_ = true;

    auto action = env->action_manager->processed_actions();

    // Split deploy (ADR-0005): joints listed in hold_joint_ids track default_joint_pos
    // instead of the policy action (upper body held; obs still see all joints).
    static const auto hold_ids = env->cfg["hold_joint_ids"]
        ? env->cfg["hold_joint_ids"].as<std::vector<int>>() : std::vector<int>{};

    // ADR-0006 Spec B: same per-joint encoder-zero correction the articulation applied on
    // read (unitree_articulation.h). q_cmd below is in TRUE joint coordinates; convert back
    // to reported coordinates right before the wire write, AFTER the safety clamp, so the
    // clamp keeps protecting in the coordinates the firmware itself compares against.
    static const auto joint_offset = env->cfg["joint_offset"]
        ? env->cfg["joint_offset"].as<std::vector<float>>() : std::vector<float>{};

    // [ESTIMATOR] Integrate the gravity-free IMU acceleration at the 1 kHz CONTROL rate.
    // This must live here and NOT in policy_step(): at the 50 Hz policy rate the same
    // reconstruction is ~4x worse, because foot-impact transients alias
    // (play.py --check-vel-increment prints both rungs: "waist-corr @50Hz" vs "@200Hz",
    // and 200 Hz is only a conservative stand-in for what this loop gets).
    // The accelerometer includes gravity like any real IMU, so removal is a + proj_grav*|g|.
    // BOTH are already in the TORSO frame: the H1-2 IMU — and the bridge's `imu` site
    // (scene_h1_2.xml framequat/gyro/accelerometer) — sit on torso_link, so root_quat_w IS
    // the torso attitude. The python reference has to undo the waist yaw first only because
    // its projected_gravity_b is the PELVIS frame; applying Rz(-psi) here would be a bug.
    // This takes its OWN lowstate snapshot rather than sharing the safety filter's read
    // below: msg_ is republished asynchronously by the DDS callback thread (Subscription.h),
    // so folding the two reads into one would silently change WHICH sample the fall detector
    // sees. An extra uncontended lock per 1 ms tick is the cheaper price.
    Eigen::Vector3f est_acc_b, est_gyro_T;
    Eigen::Quaternionf est_quat;
    float est_q[13], est_dq[13], est_dpsi;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        est_acc_b = Eigen::Vector3f(
            lowstate->msg_.imu_state().accelerometer()[0],
            lowstate->msg_.imu_state().accelerometer()[1],
            lowstate->msg_.imu_state().accelerometer()[2]
        );
        est_quat = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        // Leg encoders + waist for the leg odometry below, from the SAME sample as the IMU
        // above — the two are combined in one cross product, so they must describe one
        // instant of the robot (the same reason the safety filter keeps its own snapshot).
        est_gyro_T = Eigen::Vector3f(
            lowstate->msg_.imu_state().gyroscope()[0],
            lowstate->msg_.imu_state().gyroscope()[1],
            lowstate->msg_.imu_state().gyroscope()[2]
        );
        for (int k = 0; k < 13; ++k) {
            est_q[k] = lowstate->msg_.motor_state()[k].q() + leg_offset_[k];
            // No offset on dq: leg_offset_ is a constant encoder-zero correction, so it
            // drops out of the derivative. Logged only (2026-08-11) -- nothing in run()
            // consumes it; it exists so the offline Jacobian arm can run at the DDS rate.
            est_dq[k] = lowstate->msg_.motor_state()[k].dq();
        }
        est_dpsi = est_dq[12];
    }
    // [ESTIMATOR] Leg odometry — the HL's ABSOLUTE velocity, which `delta` does NOT cancel.
    // Here for the same reason the integrator above is: the bench measured the identical
    // reconstruction 2.4x worse when d(p_foot)/dt was taken at the 50 Hz control rate
    // (per-step vx 0.374 -> 0.176, c-averaged 0.147 -> 0.072).
    //
    // FRAMES. leg_odom_velocity returns a PELVIS-frame velocity already (leg FK is
    // pelvis-referenced), so it takes NO Rz(psi) on the way out — unlike the IMU path,
    // whose whole lever-arm correction exists because the imu site is on torso_link. The
    // waist enters only here, converting the TORSO-frame gyro into the pelvis angular
    // velocity: w_P = Rz(psi)*w_T - psi_dot*z. Dropping psi_dot is worth ~0.1 m/s per rad/s
    // of waist rate (|z x p_foot| ~ 0.1 m), so it is not optional.
    const float psi = est_q[12];
    Eigen::Vector3f w_P = rz(psi, est_gyro_T);
    w_P.z() -= est_dpsi;
    const Eigen::Vector3f g_T = est_quat.conjugate() * env->robot->data.GRAVITY_VEC_W;
    const Eigen::Vector3f grav_P = rz(psi, g_T);

    // Arm A (legodom), the SHIPPED path: unchanged, still inside the lock, still guarded by
    // lo_prev_valid_ so it never differences across a stance switch or an FSM re-entry.
    Eigen::Vector3f v_a(0, 0, 0);
    bool a_ok = false;
    {
        Eigen::Vector3f p_now[2] = { hrl::foot_site_b(est_q, 0), hrl::foot_site_b(est_q, 1) };
        std::lock_guard<std::mutex> lock(est_mu_);
        // dv_ and the leg-odom accumulator are the only state crossing the thread boundary;
        // policy_step() reads and zeroes both under this mutex. Uncontended.
        dv_ += (est_acc_b + g_T * kGravity) * H1_2_CONTROL_DT;
        if (lo_prev_valid_) {
            v_a = hrl::leg_odom_velocity(p_now, lo_p_prev_, H1_2_CONTROL_DT, w_P, grav_P);
            a_ok = true;
            if (est_arm_ == 0) {          // selected: byte-identical to the pre-bench build
                lo_sum_ += v_a;
                lo_n_++;
            }
        }
        lo_p_prev_[0] = p_now[0];
        lo_p_prev_[1] = p_now[1];
        lo_prev_valid_ = true;
    }

    // [ESTIMATOR BENCH] arms B-F, via the shared bank (hrl/base_estimators.h). Outside the
    // lock: the bank touches only this thread's own state and est_sample_, never
    // dv_/lo_sum_/lo_p_prev_. Arm A is passed in rather than recomputed so the shipped
    // guarded path stays the single source for index 0.
    float v_arm[7][2] = {{0}};
    const auto t_est0 = std::chrono::steady_clock::now();
    {
        Eigen::Vector3d out[7];
        est_bank_.step(est_q, est_dq, est_acc_b, est_gyro_T, g_T,
                       est_quat.toRotationMatrix(), a_ok ? &v_a : nullptr,
                       H1_2_CONTROL_DT, out);
        for (int a = 0; a < 7; ++a) {
            v_arm[a][0] = (float)out[a].x();
            v_arm[a][1] = (float)out[a].y();
        }
        // A passive arm reaches the log and nothing else. Only the SELECTED arm moves
        // lo_sum_, and arm 0 already did so above under its own validity guard.
        if (est_arm_ != 0) {
            std::lock_guard<std::mutex> lock(est_mu_);
            lo_sum_ += out[est_arm_].cast<float>();
            lo_n_++;
        }
    }
    {
        const double est_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t_est0).count();
        bucket(est_hist_, est_us);
        if (est_us > est_max_us_) est_max_us_ = est_us;
    }

    // [ESTIMATOR] Hand the snapshot to the flight recorder (2026-08-10). Every other sensor
    // column on that row comes from env->robot->data, which policy_step() refreshes at 50 Hz,
    // so without this an offline replay of THIS estimator is impossible from its own log --
    // measured 2026-08-10 on the 2026-08-07_11-17-20 session (meas_q/quat 56.5 Hz effective;
    // the gyro was never a column at all, and reconstructing it from 56.5 Hz attitude left
    // the w x p term carrying half the estimate's magnitude). Costs no lock and no extra
    // read: these are the values just taken above, under a lock already released.
#if SAFETY_FILTER
    for (int k = 0; k < 3; ++k) {
        est_sample_.acc[k] = est_acc_b[k];
        est_sample_.gyro[k] = est_gyro_T[k];
    }
    est_sample_.quat[0] = est_quat.w(); est_sample_.quat[1] = est_quat.x();
    est_sample_.quat[2] = est_quat.y(); est_sample_.quat[3] = est_quat.z();
    for (int k = 0; k < 13; ++k) { est_sample_.q[k] = est_q[k]; est_sample_.dq[k] = est_dq[k]; }
    est_sample_.dpsi = est_dpsi;
    for (int a = 0; a < 7; ++a) {
        est_sample_.v_arm[a][0] = v_arm[a][0];
        est_sample_.v_arm[a][1] = v_arm[a][1];
    }
#endif

#if SAFETY_FILTER
    // IMU tilt check — uses snapshot already captured by pre_run(), no extra lock needed
    const auto& q = env->robot->data.root_quat_w;
    float pitch = std::asin(std::clamp(2.0f*(q.w()*q.y() - q.z()*q.x()), -1.0f, 1.0f));
    float roll  = std::atan2(2.0f*(q.w()*q.x() + q.y()*q.z()), 1.0f - 2.0f*(q.x()*q.x() + q.y()*q.y()));
    bool tilt_safety = (std::abs(pitch) > H1_2_TILT_LIMIT || std::abs(roll) > H1_2_TILT_LIMIT);
    if (tilt_safety)
        spdlog::warn("[Safety] Tilt: pitch={:.2f} roll={:.2f} rad", pitch, roll);

    // Joint limits: per-joint COMMAND CLAMP, not a whole-body hold (changed 2026-07-16).
    // The A1a gait rides its ankle/hip/knee stops by 0.01-0.2 rad every stride; the old
    // any-joint->freeze response held all 27 joints mid-step and CAUSED the fall it
    // guarded against (bridge session 2026-07-16: alpha=1.0 at tilt 0.196, fall after).
    // Real firmware clamps the offending command; tilt/fall below keep the ramped hold.
    bool joint_hold = false;  // joint violations no longer feed the hold ramp

    // Vertical acceleration fall detection — catches pelvis sinking while body stays upright.
    // The measured motor torques are read from the SAME snapshot (2026-08-05): both feed the
    // safety filter this tick, so they must describe one instant of the robot.
    Eigen::Vector3f lin_acc_b;
    std::array<float, 27> tau_est{};
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        lin_acc_b = Eigen::Vector3f(
            lowstate->msg_.imu_state().accelerometer()[0],
            lowstate->msg_.imu_state().accelerometer()[1],
            lowstate->msg_.imu_state().accelerometer()[2]
        );
        for (int jid = 0; jid < 27; ++jid)
            tau_est[jid] = lowstate->msg_.motor_state()[jid].tau_est();
    }
    float a_world_z = (env->robot->data.root_quat_w * lin_acc_b).z() - 9.81f;
    if (a_world_z < H1_2_FALL_ACC_THRESH) fall_acc_counter_ = std::min(fall_acc_counter_ + 1, H1_2_FALL_ACC_WINDOW);
    else                                  fall_acc_counter_ = std::max(fall_acc_counter_ - 1, 0);
    bool fall_detected = (fall_acc_counter_ >= H1_2_FALL_ACC_WINDOW);
    if (fall_detected)
        spdlog::warn("[Safety] Fall: a_world_z={:.2f} m/s²", a_world_z);

    // Ramp hold counter up/down over H1_2_RAMP_CYCLES ticks to avoid torque spikes
    bool hold_active = joint_hold || tilt_safety || fall_detected;
    if (hold_active) hold_counter_ = std::min(hold_counter_ + 1, H1_2_RAMP_CYCLES);
    else             hold_counter_ = std::max(hold_counter_ - 1, 0);
    float alpha = static_cast<float>(hold_counter_) / H1_2_RAMP_CYCLES;

    // Per-joint torque / joint-velocity trip (2026-08-05). Scoped and ramped exactly like
    // the position clamp above -- per joint, not whole body -- so a joint that is genuinely
    // running away is pulled back to its measured position while the rest keep tracking the
    // policy. Thresholds and their derivation live in h1_2_limits.h.
    //
    // The 50 ms ramp doubles as the debounce: a single-sample sensor spike buys alpha 0.02
    // for one tick, which is nothing, while a real trip reaches a full hold in 50 ms. On the
    // 2026-08-05 splay that is alpha=1 by t+50 ms, with hip roll still at ~4 deg -- it
    // reached -67 deg by t+240 ms unrestrained. So no separate debounce counter is needed.
    //
    // This is why it exists at all: the 2026-08-05 splay never tripped tilt (peak 22.8 deg
    // against the 25 deg limit) and never tripped the vertical-acceleration fall detector.
    // The filter as it stood had no channel that could see the failure.
    // trip_ratio/trip_jid are logged ALWAYS, not only while tripping (2026-08-05, second
    // pass). Booleans alone made the channel unmeasurable: a clean session says "did not
    // fire" and nothing about whether it cleared by 3x or by 2%. That mattered immediately
    // -- thresholds derived from 389 s of hardware walking (which is A0-dominated) turned
    // out to sit UNDER this A1 candidate's own bridge gait on three joints, and the boolean
    // columns could not have shown that coming. With the ratio logged, max(trip_ratio) over
    // a clean run IS the headroom, per joint, directly.
    bool trig_torque = false, trig_dq = false;
    int trip_jid = -1;
    float trip_ratio = 0.0f;
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        const int jid = (int)env->robot->data.joint_ids_map[i];
        const float at = std::abs(tau_est[jid]), ad = std::abs(env->robot->data.joint_vel[i]);
        const bool ht = at > h1_2_joint_trips[jid].tau;
        const bool hd = ad > h1_2_joint_trips[jid].dq;
        trig_torque = trig_torque || ht;
        trig_dq = trig_dq || hd;
        // Compare joints by overshoot RATIO, so a 15 Nm wrist and a 385 Nm knee stay
        // commensurable in the logged attribution.
        const float r = std::max(at / h1_2_joint_trips[jid].tau,
                                 ad / h1_2_joint_trips[jid].dq);
        if (r > trip_ratio) { trip_ratio = r; trip_jid = jid; }
        trip_counter_[i] = (ht || hd) ? std::min(trip_counter_[i] + 1, H1_2_RAMP_CYCLES)
                                      : std::max(trip_counter_[i] - 1, 0);
    }
    if (trig_torque || trig_dq)
        spdlog::warn("[Safety] Trip: {} at {:.2f}x threshold ({} joints ramping)",
                     h1_2_joint_names[trip_jid], trip_ratio,
                     std::count_if(trip_counter_.begin(), trip_counter_.end(),
                                   [](int c){ return c > 0; }));

    bool any_clamp = false;
    float alpha_trip_max = 0.0f;
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        // Whole-body tilt/fall hold and this joint's own trip hold compose by max: whichever
        // says "hold harder" wins. Tilt/fall keeps its existing whole-body reach untouched.
        // H1_2_TRIP_HOLD 0 keeps the ramp COMPUTED (so alpha_trip is still logged and the
        // channel stays measurable) but never lets it reach the command. See h1_2_limits.h.
        const float alpha_trip = static_cast<float>(trip_counter_[i]) / H1_2_RAMP_CYCLES;
        alpha_trip_max = std::max(alpha_trip_max, alpha_trip);
        const float a_i = H1_2_TRIP_HOLD ? std::max(alpha, alpha_trip) : alpha;
        // alpha=0: pure policy output  |  alpha=1: hold at current measured position
        float q_cmd = (1.0f - a_i) * action[i] + a_i * q_meas;
        // per-joint safety clamp (commands past the mechanical stop are pointless and
        // trip the real firmware; measured grazes are the plant's business, not ours)
        const float lo = h1_2_joint_limits[jid].min, hi = h1_2_joint_limits[jid].max;
        if (q_cmd < lo || q_cmd > hi) { q_cmd = std::clamp(q_cmd, lo, hi); any_clamp = true; }
        if (std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end())
            q_cmd = env->robot->data.default_joint_pos[i];
        const float offset = (i < (int)joint_offset.size()) ? joint_offset[i] : 0.0f;
        lowcmd->msg_.motor_cmd()[jid].q() = q_cmd - offset;
    }
    joint_hold = any_clamp;  // recorded as trig_joint in the flight log (no hold effect)

    if (safety_logger_.enabled()) {
        float quat[4] = { q.w(), q.x(), q.y(), q.z() };
        float acc[3]  = { lin_acc_b.x(), lin_acc_b.y(), lin_acc_b.z() };
        // cmd/ach_vel are the values cached by the last policy_step() (2026-07-21, WL-B0) —
        // A1 already computed both for the goal-space state s; this just also logs them.
        float cmd[3]     = { last_cmd_.x(), last_cmd_.y(), last_cmd_.z() };
        float ach_vel[3] = { last_lin_vel_b_.x(), last_lin_vel_b_.y(), last_lin_vel_b_.z() };
        safety_logger_.record(action,
                              env->robot->data.joint_pos.data(),
                              env->robot->data.joint_vel.data(),
                              quat, acc, alpha, joint_hold, tilt_safety, fall_detected,
                              cmd, ach_vel, isaaclab::g_gait_phase_obs,
                              trig_torque, trig_dq, trip_jid, alpha_trip_max, trip_ratio,
                              &est_sample_);
    }
#else
    for(int i(0); i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        bool held = std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end();
        float q_cmd = held ? env->robot->data.default_joint_pos[i] : action[i];
        const float offset = (i < (int)joint_offset.size()) ? joint_offset[i] : 0.0f;
        lowcmd->msg_.motor_cmd()[jid].q() = q_cmd - offset;
    }
#endif

    // [T5] tick end. Single exit path (verified: run() has no early returns), so one
    // measurement covers the whole tick. Counters only -- the report is emitted from exit().
    {
        const double work_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t_tick0).count();
        bucket(work_hist_, work_us);
        if (work_us > work_max_us_) work_max_us_ = work_us;
        if (work_us > 800.0) overrun_count_++;
        tick_count_++;
    }
}
