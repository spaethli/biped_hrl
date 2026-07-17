// A1 (HIRO) deploy goal space — robot-local C++ mirror of
//   src/tasks/velocity/rl/hrl/goal_space.py
// The high level emits a bounded goal g in [-1,1]^dim; this maps it to the absolute
// window target V* and the low level observes the remaining delta V* - s. The component
// list is declared in deploy.yaml (`hrl.goal_components`), so goal_dim is derived, never
// hardcoded — exactly like the training side.
//
// Supported components (the 7-dim default = velocity+orientation+height):
//   velocity(3)    = [lin_vel_b.x, lin_vel_b.y, ang_vel_b.z]
//                    scale = half command range, center = command midpoint
//   orientation(3) = projected_gravity_b
//                    scale = 1, center/nominal = (0,0,-1) upright
//   height(1)      = base height (world z)
//                    scale = 0.2 m, center/nominal = nominal_root_height
//
// SCALE/CENTER SOURCE (changed 2026-07-16, WL-C). The g -> V* decode is a property of the
// TRAINED policy, so it is pinned from the HL ONNX metadata (`goal_scale` / `goal_center`,
// written by HierarchicalRunner._attach_hrl_metadata) via freeze_scale/freeze_center, which
// State_RLHRL does at load. `commands.base_velocity.ranges` in deploy.yaml stays the
// OPERATOR's knob — the joystick safety clamp (isaaclab observations.h `velocity_commands`)
// — and is now free to be narrowed for safety without touching policy semantics.
//
// Legacy exports (no metadata) fall back to deriving scale/center from those ranges, which
// is the old, coupled behavior: then the ranges MUST equal the ranges the policy was trained
// with (the final curriculum stage) or V* is reconstructed on the wrong scale, and narrowing
// them for safety silently rescales the HL's goals (that coupling is what produced the
// 2026-07-15 phantom "HL hold degeneracy" in sim; see doc/hrl/A1_findings.md WL-C).
// Re-export the policy to get the metadata and lift that restriction.
#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace hrl
{

class GoalSpace
{
public:
    enum Comp { VELOCITY, ORIENTATION, HEIGHT };

    // names: ordered component list from deploy.yaml; nominal_height: the training
    // default_root_state z (used as the height center / nominal target).
    GoalSpace(const std::vector<std::string>& names, float nominal_height)
    : nominal_height_(nominal_height)
    {
        for (const auto& n : names) {
            if      (n == "velocity")    comps_.push_back(VELOCITY);
            else if (n == "orientation") comps_.push_back(ORIENTATION);
            else if (n == "height")      comps_.push_back(HEIGHT);
            else throw std::runtime_error("hrl::GoalSpace: unknown goal component '" + n + "'");
        }
    }

    int dim() const
    {
        int d = 0;
        for (auto c : comps_) d += comp_dim(c);
        return d;
    }

    // Pin the goal SCALE to the value the policy TRAINED with (from the HL ONNX metadata).
    // Empty vector = leave on the legacy range-derived path. See the header note.
    // Only the scale is pinnable: every scale entry is a *difference* (m/s, rad/s, m of
    // height deviation) and so is frame-independent, whereas the ONNX `goal_center` holds
    // absolute values in the TRAINING frame — its height column is pelvis z (1.02) while
    // this deploy measures height at the imu site (nominal_root_height 1.3076). Adopting it
    // would inject that 0.29 m frame offset. See center().
    void freeze_scale(const Eigen::VectorXf& v) { check_dim(v, "goal_scale"); frozen_scale_ = v; }
    bool scale_frozen() const { return frozen_scale_.size() > 0; }

    // Learned (task) goal columns for hl_velocity_goals_only — the velocity component,
    // which the training side requires to be the contiguous prefix (goal_space.py).
    int task_dim() const
    {
        if (comps_.empty() || comps_[0] != VELOCITY)
            throw std::runtime_error(
                "hrl::GoalSpace: velocity must be the first goal component "
                "(training assumes the task prefix).");
        return comp_dim(VELOCITY);
    }

    // Current goal-space state s. lin_vel_b = base linear velocity in the BODY frame (xy
    // used); ang_vel_b.z + projected_gravity_b come from the IMU (robot->data); height is
    // the base world z (from the sim HighState). Caller supplies lin_vel_b/height so this
    // module stays agnostic to the HighState message type.
    Eigen::VectorXf state(isaaclab::ManagerBasedRLEnv* env,
                          const Eigen::Vector3f& lin_vel_b, float height) const
    {
        const auto& d = env->robot->data;
        Eigen::VectorXf s(dim());
        int i = 0;
        for (auto c : comps_) {
            if (c == VELOCITY) {
                s[i + 0] = lin_vel_b.x();
                s[i + 1] = lin_vel_b.y();
                s[i + 2] = d.root_ang_vel_b.z();
            } else if (c == ORIENTATION) {
                s.segment<3>(i) = d.projected_gravity_b;
            } else { // HEIGHT
                s[i] = height;
            }
            i += comp_dim(c);
        }
        return s;
    }

    // Per-dim HIRO delta scale (V* = ref + scale .* g). Pinned to the trained value when the
    // HL ONNX carried `goal_scale`; otherwise derived from the deploy.yaml command ranges
    // (legacy — see the header note: then those ranges must equal the trained ones).
    Eigen::VectorXf scale(isaaclab::ManagerBasedRLEnv* env) const
    {
        if (frozen_scale_.size()) return frozen_scale_;
        const auto r = env->cfg["commands"]["base_velocity"]["ranges"];
        Eigen::VectorXf sc(dim());
        int i = 0;
        for (auto c : comps_) {
            if (c == VELOCITY) {
                sc[i + 0] = half(r["lin_vel_x"]);
                sc[i + 1] = half(r["lin_vel_y"]);
                sc[i + 2] = half(r["ang_vel_z"]);
            } else if (c == ORIENTATION) {
                sc.segment<3>(i).setOnes();
            } else { // HEIGHT
                sc[i] = 0.2f;
            }
            i += comp_dim(c);
        }
        return sc;
    }

    // Absolute-target center (used only by `absolute` mode): velocity = command midpoint,
    // orientation = upright (0,0,-1), height = nominal_root_height.
    //
    // NOT pinnable from the ONNX (unlike scale): the height column must stay this deploy's
    // imu-site nominal_root_height (1.3076), not training's pelvis z (1.02) — see
    // freeze_scale. The velocity columns therefore remain range-derived, so **for
    // `hl_target_mode: absolute` the command ranges must still equal the TRAINED ranges**
    // (narrowing them for safety would shift the absolute reference). `delta` — the A1
    // default and what is deployed — never calls this, so the safety clamp is free there.
    // State_RLHRL warns loudly if absolute mode is configured.
    Eigen::VectorXf center(isaaclab::ManagerBasedRLEnv* env) const
    {
        const auto r = env->cfg["commands"]["base_velocity"]["ranges"];
        Eigen::VectorXf ce(dim());
        int i = 0;
        for (auto c : comps_) {
            if (c == VELOCITY) {
                ce[i + 0] = mid(r["lin_vel_x"]);
                ce[i + 1] = mid(r["lin_vel_y"]);
                ce[i + 2] = mid(r["ang_vel_z"]);
            } else if (c == ORIENTATION) {
                ce[i + 0] = 0.0f; ce[i + 1] = 0.0f; ce[i + 2] = -1.0f;
            } else { // HEIGHT
                ce[i] = nominal_height_;
            }
            i += comp_dim(c);
        }
        return ce;
    }

    // Map the bounded goal g to the absolute window target V*.
    //   delta    (HIRO default): V* = state  + scale .* g
    //   absolute               : V* = center + scale .* g
    // task_only (hl_velocity_goals_only): g carries only the task (velocity) prefix;
    // orientation/height targets are pinned to nominal — mirrors goal_space.py to_target.
    Eigen::VectorXf to_target(isaaclab::ManagerBasedRLEnv* env, const Eigen::VectorXf& state,
                              const Eigen::VectorXf& g, const std::string& mode,
                              bool task_only = false) const
    {
        const Eigen::VectorXf ref = (mode == "absolute") ? center(env) : state;
        if (!task_only) return ref + scale(env).cwiseProduct(g);
        const int td = task_dim();
        // oracle_target(0) = nominal non-task targets (velocity cols overwritten below).
        Eigen::VectorXf v = oracle_target(Eigen::Vector3f::Zero());
        v.head(td) = ref.head(td) + scale(env).head(td).cwiseProduct(g.head(td));
        return v;
    }

    // Oracle (no-network) target: the commanded twist for the velocity (task) component,
    // nominal for the rest (upright / nominal height). Mirrors goal_space.py oracle_target.
    // `command` is the [vx, vy, yaw] twist. Independent of g / hl_target_mode / scale.
    Eigen::VectorXf oracle_target(const Eigen::Vector3f& command) const
    {
        Eigen::VectorXf v(dim());
        int i = 0;
        for (auto c : comps_) {
            if (c == VELOCITY) {
                v[i + 0] = command.x(); v[i + 1] = command.y(); v[i + 2] = command.z();
            } else if (c == ORIENTATION) {
                v[i + 0] = 0.0f; v[i + 1] = 0.0f; v[i + 2] = -1.0f;
            } else { // HEIGHT
                v[i] = nominal_height_;
            }
            i += comp_dim(c);
        }
        return v;
    }

    // Per-dim Gaussian-noise std from a {component: std} YAML map (missing comp -> 0),
    // broadcast across each component's dims. For mimicking the real-robot state-estimator
    // noise on the goal-space feedback s (sim s is ground truth). Returns zeros if cfg null.
    Eigen::VectorXf noise_std(const YAML::Node& cfg) const
    {
        Eigen::VectorXf sd = Eigen::VectorXf::Zero(dim());
        if (!cfg) return sd;
        int i = 0;
        for (auto c : comps_) {
            const char* key = (c == VELOCITY) ? "velocity"
                            : (c == ORIENTATION) ? "orientation" : "height";
            if (cfg[key]) sd.segment(i, comp_dim(c)).setConstant(cfg[key].as<float>());
            i += comp_dim(c);
        }
        return sd;
    }

private:
    static int comp_dim(Comp c) { return c == HEIGHT ? 1 : 3; }

    // A metadata vector that disagrees with the declared goal space means the ONNX and the
    // deploy yaml describe different policies -> refuse rather than decode g on wrong dims.
    void check_dim(const Eigen::VectorXf& v, const char* what) const
    {
        if (v.size() && v.size() != dim())
            throw std::runtime_error(
                std::string("hrl::GoalSpace: ") + what + " from ONNX metadata has dim " +
                std::to_string(v.size()) + " but goal_components declare " +
                std::to_string(dim()) + " — the ONNX and deploy.yaml disagree.");
    }

    // half = max((hi-lo)/2, 1e-3); mid = (lo+hi)/2  (mirror goal_space.py).
    static float half(const YAML::Node& n)
    {
        return std::max((n[1].as<float>() - n[0].as<float>()) / 2.0f, 1e-3f);
    }
    static float mid(const YAML::Node& n)
    {
        return (n[0].as<float>() + n[1].as<float>()) / 2.0f;
    }

    std::vector<Comp> comps_;
    float nominal_height_;
    Eigen::VectorXf frozen_scale_;  // empty -> derive from the deploy.yaml command ranges
};

} // namespace hrl
