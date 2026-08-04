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
    // sdk motor id -> articulation slot for the legs + waist. Everything else in this file
    // indexes by sdk id (joint_ids_map[i]); the FK and the waist encoder need the inverse.
    sdk_slot_.fill(-1);
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); ++i) {
        const int jid = (int)env->robot->data.joint_ids_map[i];
        if (jid >= 0 && jid < (int)sdk_slot_.size()) sdk_slot_[jid] = i;
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
                 "base_vel_from_imu={} base_height_from_fk={} fk_h_nominal={:.4f} "
                 "(HighState=rt/sportmodestate)",
                 oracle_ ? "oracle" : "learned", c_, hl_target_mode_, goal_space_->dim(),
                 velgoal_, hl_obs_vel_, cadence_dim_, period_lo_, period_hi_, pin_period_,
                 vel_from_imu_, height_from_fk_, fk_h_nominal_);
    // The IMU rung reconstructs the WITHIN-WINDOW INCREMENT only, which is exactly what the
    // LL's `delta` goal needs and exactly what the HL's absolute lin-vel input does NOT get:
    // s[0..1] is 0 by construction at every window start, i.e. at every HL fire. Flag it —
    // on real hardware the SportModeState alternative already reads 0 there, so this is not
    // a regression, but in the bridge it IS a behavior change vs the privileged path.
    if (vel_from_imu_ && hl_obs_vel_)
        spdlog::warn("[HRL] base_vel_from_imu + hl_obs_vel: the HL's (vx,vy) input is the "
                     "within-window increment, which is 0 at every HL fire step. The HL's "
                     "ABSOLUTE velocity input is NOT solved by this estimator (leg odometry "
                     "— play.py --check-leg-odometry — is the intended source).");
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
    if (ll_in != policy_dim + goal_space_->dim()) {
        throw std::runtime_error(
            "[HRL] low_level.onnx input dim " + std::to_string(ll_in) + " != policy(" +
            std::to_string(policy_dim) + ") + goal_dim(" +
            std::to_string(goal_space_->dim()) + ") — obs/goal_components mismatch.");
    }

    if (oracle_) {
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

    const int64_t hl_in = onnx_input_dim(hl_path);
    const int hl_in_expect = policy_dim + command_dim + (hl_obs_vel_ ? 2 : 0);
    if (hl_in != hl_in_expect) {
        throw std::runtime_error(
            "[HRL] high_level.onnx input dim " + std::to_string(hl_in) + " != policy+" +
            "command" + (hl_obs_vel_ ? "+hl_vel(2)" : "") + " = " +
            std::to_string(hl_in_expect) + " — check hrl.hl_obs_vel in deploy yaml.");
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
            // Same (noised) estimate the LL's goal uses — training feeds the HL the
            // identical reading (td3._state_vec order: policy, command, hl_vel).
            if (hl_obs_vel_) { hl_in.push_back(s[0]); hl_in.push_back(s[1]); }
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
        telemetry_.record(step_ * (float)env->step_dt, command.data(), s, target_, period,
                          env->robot->data.joint_pos[1], env->robot->data.joint_pos[7], ar,
                          est, gt);
    }
}

void State_RLHRL::run()
{
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
    Eigen::Vector3f est_acc_b;
    Eigen::Quaternionf est_quat;
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
    }
    {
        // Only dv_ crosses the thread boundary; policy_step() reads and zeroes it under the
        // same mutex. Three floats, uncontended — nothing else here is shared.
        std::lock_guard<std::mutex> lock(est_mu_);
        dv_ += (est_acc_b + (est_quat.conjugate() * env->robot->data.GRAVITY_VEC_W) * kGravity)
               * H1_2_CONTROL_DT;
    }

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
    Eigen::Vector3f lin_acc_b;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        lin_acc_b = Eigen::Vector3f(
            lowstate->msg_.imu_state().accelerometer()[0],
            lowstate->msg_.imu_state().accelerometer()[1],
            lowstate->msg_.imu_state().accelerometer()[2]
        );
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

    bool any_clamp = false;
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        // alpha=0: pure policy output  |  alpha=1: hold at current measured position
        float q_cmd = (1.0f - alpha) * action[i] + alpha * q_meas;
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
                              cmd, ach_vel, isaaclab::g_gait_phase_obs);
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
}
