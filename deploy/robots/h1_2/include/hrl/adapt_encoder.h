// WP5d — the deploy side of H-adapt's adaptation module phi.
//
// The HL trained in WP5 Phase 1 reads a privileged environment latent
// e = (payload_kg, com_dx, com_dy, com_dz, friction) that only exists in the simulator
// (mdp/observations.py reads body_mass/body_ipos/geom_friction). On hardware that channel
// is replaced by z_hat = phi(history), computed from signals the robot actually has: the
// last H control-rate frames of the HL's OWN observation vector, minus the tails the HL
// appends for itself.
//
// THE PINNED CONTRACT (doc/hrl/A1a_deploy_plan.md "WP5d"). phi's per-step vector is
// exactly `obs["policy"] ++ obs["command"]` = 92 floats:
//     ang_vel(3) | gravity(3) | phase(2) | q_rel(27) | qd_rel(27) | last_action(27) | cmd(3)
// which is the first 92 columns of the HL input State_RLHRL already builds. Sim side:
// td3.py `_state_vec` = [policy, command, hl_vel?, hl_e?], and h1_2_a1/env_cfgs.py builds
// `policy` by popping `command` out of the actor group ORDER-PRESERVED.
// WARNING: the A0 flat 92-vector is a DIFFERENT permutation of the same terms (command sits
// mid-vector at cols 6:9 there, last here). Nothing about a length check can tell them
// apart, which is why the layout string travels in the ONNX metadata and is asserted.
//
// Everything geometric or dimensional about phi is a POLICY property and travels baked into
// adapt_encoder.onnx, never in deploy.yaml — same rule and same reason as `goal_scale`
// (see State_RLHRL.cpp): re-deriving a policy quantity at inference from an operator-editable
// file is the bug class that produced a whole family of phantom eval results.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace hrl
{

// The contract exported alongside phi's weights, read once at load from the ONNX custom
// metadata map. Deploy validates against THIS, not against a yaml claim.
struct AdaptEncoderMeta
{
    int history_len{0};             // phi_history_len   (H, control-rate frames)
    int input_dim{0};               // phi_input_dim     (D, floats per frame)
    int z_dim{0};                   // z_dim             (latent width)
    std::string time_order;         // phi_time_order    ("oldest_first")
    std::string input_layout;       // phi_input_layout  ("policy89+command3")
    std::string z_names;            // z_names           (comma-separated, for the log)
    // z_hat = clamp(center + scale * raw_output, clip_lo, clip_hi). Written by the exporter
    // even when phi already emits raw physical units (scale 1, center 0), so the deploy
    // never has to ASSUME which convention an export used.
    std::vector<float> scale, center, clip_lo, clip_hi;
    // The cold-start prior: what the HL reads until the history buffer has filled. Baked
    // rather than defaulted in code because a zero vector is NOT nominal for this latent —
    // payload and CoM read as deltas (zero is nominal) but friction reads as an ABSOLUTE
    // coefficient whose training support is 0.3-1.6, so a zero fill hands the HL a
    // frictionless floor: not merely wrong, but off-distribution in the worst direction.
    std::vector<float> cold;

    // Fail-closed completeness check. Returns "" when usable, else why not.
    std::string validate() const
    {
        if (history_len <= 0 || input_dim <= 0 || z_dim <= 0)
            return "phi_history_len / phi_input_dim / z_dim missing or non-positive";
        if (time_order != "oldest_first")
            return "phi_time_order is '" + time_order + "', expected 'oldest_first' — a "
                   "reversed time axis is a perfectly valid tensor that returns garbage";
        if (input_layout.empty())
            return "phi_input_layout missing — cannot prove the export used the A1 HL "
                   "column order rather than the A0 flat one";
        const size_t z = (size_t)z_dim;
        if (scale.size() != z || center.size() != z || clip_lo.size() != z
            || clip_hi.size() != z || cold.size() != z)
            return "z_scale / z_center / z_clip_lo / z_clip_hi / z_cold must each carry "
                   "z_dim entries";
        for (int i = 0; i < z_dim; ++i) {
            if (clip_lo[i] > clip_hi[i]) return "z_clip_lo > z_clip_hi";
            if (cold[i] < clip_lo[i] || cold[i] > clip_hi[i])
                return "z_cold lies outside [z_clip_lo, z_clip_hi] — the cold-start prior "
                       "must itself be in the training support";
        }
        return "";
    }
};

// Fixed-capacity ring of the last H frames, pushed once per CONTROL step (50 Hz) and read
// at each HL fire (6.25 Hz). Policy-thread-only: State_RLHRL::policy_step() is the sole
// writer AND the sole reader, so no mutex — the same argument the FSM header already makes
// for est_sample_, and unlike dv_/lo_sum_ which cross to the 1 kHz run() thread.
class HistoryBuffer
{
public:
    void configure(int history_len, int input_dim)
    {
        h_ = history_len;
        d_ = input_dim;
        buf_.assign((size_t)h_ * (size_t)d_, 0.0f);
        reset();
    }

    // Cleared on FSM re-entry, alongside dv_/lo_sum_/est_bank_ and for the same reason: a
    // buffer carried across an exit spans the gap and describes a robot that was doing
    // something else — usually falling, since that is what a re-entry follows.
    void reset() { n_ = 0; head_ = 0; }

    bool full() const { return n_ >= h_; }
    int size() const { return n_; }

    // One frame, supplied as the two pieces the caller already holds (policy ++ command),
    // so the hot path never materialises a temporary. Silently no-op on a wrong total
    // length: the caller checks that at LOAD time, where it can still refuse to run.
    void push(const std::vector<float>& a, const std::vector<float>& b)
    {
        if ((int)(a.size() + b.size()) != d_) return;
        float* slot = &buf_[(size_t)head_ * (size_t)d_];
        std::copy(a.begin(), a.end(), slot);
        std::copy(b.begin(), b.end(), slot + a.size());
        head_ = (head_ + 1) % h_;
        if (n_ < h_) ++n_;
    }

    // Flattens to row-major [1, H, D], OLDEST FRAME FIRST — out[t*D + d]. Returns false
    // (leaving `out` untouched) unless the ring is full, which is what makes "phi is never
    // called on a partial buffer" an executable invariant rather than a code-reading
    // exercise.
    bool window(std::vector<float>& out) const
    {
        if (!full()) return false;
        out.resize((size_t)h_ * (size_t)d_);
        const int oldest = head_;  // full ring: the next slot to overwrite IS the oldest
        for (int t = 0; t < h_; ++t) {
            const float* src = &buf_[(size_t)((oldest + t) % h_) * (size_t)d_];
            std::copy(src, src + d_, out.begin() + (size_t)t * (size_t)d_);
        }
        return true;
    }

private:
    int h_{0}, d_{0}, n_{0}, head_{0};
    std::vector<float> buf_;
};

// Everything the deploy can check about an export BEFORE it is allowed to command the robot,
// as one pure function so the whole fail-closed decision is testable without a robot, a
// bridge or onnxruntime (test/adapt_encoder_test.cpp). State_RLHRL calls this at load and
// throws on a non-empty return; the caller supplies the two ONNX-declared sizes it read from
// the session. Returns "" when the export is usable with THIS deploy.
inline std::string check_deploy_contract(const AdaptEncoderMeta& m, int policy_dim,
                                         int command_dim, long long onnx_input_size,
                                         long long onnx_output_size)
{
    const std::string bad = m.validate();
    if (!bad.empty()) return bad;
    // The layout string is the ONLY thing that can tell the A1 HL column order (command
    // LAST) from the A0 flat one (command mid-vector at cols 6:9). Both are 92 floats per
    // frame, so no dimension check anywhere can separate them.
    const std::string want = "policy" + std::to_string(policy_dim) + "+command"
                           + std::to_string(command_dim);
    if (m.input_layout != want)
        return "phi_input_layout '" + m.input_layout + "' != this deploy's '" + want
             + "'; phi's per-step vector must be obs[policy] ++ obs[command] in the training "
               "column order";
    if (m.input_dim != policy_dim + command_dim)
        return "phi_input_dim " + std::to_string(m.input_dim) + " != policy("
             + std::to_string(policy_dim) + ") + command(" + std::to_string(command_dim) + ")";
    const long long want_in = (long long)m.history_len * m.input_dim;
    if (onnx_input_size != want_in)
        return "the ONNX declares an input of " + std::to_string(onnx_input_size)
             + " floats but its own metadata says H*D = " + std::to_string(m.history_len)
             + "*" + std::to_string(m.input_dim) + " = " + std::to_string(want_in);
    if (onnx_output_size != m.z_dim)
        return "the ONNX output dim " + std::to_string(onnx_output_size) + " != z_dim "
             + std::to_string(m.z_dim);
    return "";
}

// z_hat = clamp(center + scale * raw, clip_lo, clip_hi), written into `z`. Returns how many
// components the clamp actually bit.
//
// The clamp cannot make a wrong estimate right. What it guarantees is that a wrong estimate
// is never OUT-OF-DISTRIBUTION wrong: phi trains on sim histories, and inferring a hidden
// property from hardware proprioception is exactly what this thesis is about, so a
// confidently-wrong z_hat is the plausible failure. Bounding it to the training support
// keeps the HL evaluated on inputs it was trained on, and the returned count is logged so
// sustained clipping reports "phi does not transfer" in the DATA rather than in the robot's
// behaviour.
inline int decode_latent(const AdaptEncoderMeta& m, const std::vector<float>& raw,
                         std::vector<float>& z)
{
    z.resize((size_t)m.z_dim);
    int clipped = 0;
    for (int i = 0; i < m.z_dim; ++i) {
        const float v = m.center[i] + m.scale[i] * raw[i];
        z[i] = std::clamp(v, m.clip_lo[i], m.clip_hi[i]);
        if (v != z[i]) ++clipped;
    }
    return clipped;
}

} // namespace hrl
