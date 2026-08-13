#pragma once
// [SAFETY FILTER] flight recorder — delete this file when reverting.
//
// Two-file output (base path from H1_2_SAFETY_LOG):
//   <base>.csv        standard CSV, one header row + one row per control tick
//                      (decimated to ~500 Hz; a tick with trig_joint/trig_tilt/
//                      trig_fall set is NEVER decimated, so every raw joint-limit
//                      violation and every tilt/fall event is still captured at
//                      the full control rate)
//   <base>_meta.json  static config written once at init (limits, names, dt)
//
// Hot-path safe: record() only appends a formatted line to an in-RAM string
// buffer (no syscall). The file is written on periodic flush (~5 s) and on exit.
// Floats are logged at 4 significant digits (2026-07-27, file-size reduction) --
// plenty for joint angles/velocities, well past the sensor noise floor.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "h1_2_limits.h"

// Process-wide (NOT per-instance) entry counter: State_RLBase (A0) and State_RLHRL (A1)
// each own a SEPARATE SafetyLogger object, so a per-instance counter would restart at 0
// the first time the OTHER FSM type is entered, colliding with entries already written
// by the first type to the same shared file. 2026-07-17 finding (2nd round): a session
// that ran A0 first, then switched to A1, needs entry ids that stay unique across that
// switch, not just across repeated entries into the same type.
inline int g_safety_logger_entry = -1;

// One tick of the estimator's OWN lowstate read, for offline estimator replay.
//
// WHY THIS IS A SEPARATE CHANNEL. Every other sensor column on a row comes from
// `env->robot->data`, the articulation cache, and `env->robot->update()` is called exactly
// once per policy_step() -- at 50 Hz. So meas_q/meas_dq/quat are a 50 Hz STAIRCASE in a
// 1 kHz file, and differencing them reproduces the exact aliasing State_RLHRL::run()'s
// estimator exists to avoid. Measured 2026-08-10: those columns hold for 94.4% of rows
// (17.70 rows between changes = 50.2 Hz) at the current 4-sig-fig/500 Hz setting AND for
// 95.0% (19.99 rows = 50.0 Hz) in a pre-dd8ecfc capture at 6 sig figs and a true 1000 Hz
// -- so this was never a precision or a decimation artifact, and reverting dd8ecfc does
// not fix it. run() already takes the right snapshot at the DDS rate and then discards it;
// this channel just routes it to the file. No extra lock, no extra read, no extra rows.
//
// The gyro is here because NO version of this recorder has ever logged one, which forced
// the 2026-08-10 offline replay to reconstruct it from 56.5 Hz attitude -- where the
// `w x p` term carried half the leg-odometry magnitude (rms 0.896 vs 0.890), i.e. the
// dominant error source in the replay was the missing column.
//
// `dq` joined on 2026-08-11 for the same reason and at the same zero cost: MotorState::dq()
// is already inside the lock this snapshot holds, so it is 13 more columns and no extra DDS
// read. Without it the Jacobian leg-odometry arm (`v = -(J qdot + dp/dpsi * dpsi) - w x p`,
// arm E of doc/hrl/h1_2_ekf_design.md) can only run off the 50 Hz articulation cache, which
// is not comparable to the arms that run at the DDS rate. NOTE it does not make that arm
// derivative-free: dq is differentiated on the motor board, so arm E measures WHERE the
// differentiation happens, not its absence. The EKF itself consumes positions only.
struct EstSample
{
    float acc[3];   // raw accelerometer, TORSO frame, gravity INCLUDED (as the IMU reports)
    float quat[4];  // torso attitude (w, x, y, z)
    float gyro[3];  // torso gyro
    float q[13];    // leg encoders + waist yaw, sdk order, encoder-zero offset APPLIED
    float dq[13];   // the SAME 13 joints' velocities (2026-08-11) -- see note below
    float dpsi;     // waist yaw rate == dq[12]; kept for back-compat with the 08-10 columns
    // Every estimator arm's PELVIS-frame velocity, vx/vy only (the HL consumes head<2>).
    // All seven run every tick; exactly one is selected to feed obs["hl_vel"], the rest are
    // passive and reach this log and nothing else. NaN is written as-is: a diverging arm is
    // evidence and is never reset (docs/adr/0007). Order is kEstArmNames.
    float v_arm[7][2];
};

// Fixed column order for EstSample::v_arm, and the accepted values of the deploy config's
// `base_estimator:` key. Changing this order silently reinterprets every logged session, so
// it is a one-way door: append, never reorder.
static constexpr const char* kEstArmNames[7] = {
    "legodom", "compl", "ekf", "ekf_grav", "ekf_rot", "jacobian", "ekf_att"
};

class SafetyLogger
{
public:
    // Enabled only if base_path is non-empty (H1_2_SAFETY_LOG set).
    bool enabled() const { return !base_.empty(); }

    // Called once per State ENTRY (every `h`/`o` press), not once per process, and A0/A1
    // use SEPARATE SafetyLogger instances (see g_safety_logger_entry above). Whether to
    // truncate is decided by whether the file already exists on disk (robust across BOTH
    // repeated entries into one FSM type AND switching between A0 and A1), not by this
    // instance's own memory of having run before -- an instance that has never run yet
    // (e.g. A1's, the first time you press `h` after already having used A0) must still
    // append, not wipe out what the other type already wrote this session.
    // `with_estimator` adds the EstSample columns (A1 only -- A0 has no such snapshot to
    // log). It is fixed at init because the header is written here and every row must match
    // it; record() writes the block unconditionally once enabled, so a tick that somehow
    // arrives without a sample still emits a full-width row.
    // `joint_offset` is recorded into the meta json because `est_q` is logged in whatever
    // encoder convention the running config applies (`true = reported + joint_offset`), and
    // the two live deploy configs DISAGREE: velocity/v0 (A0) ships
    // [0,0.012,0,0.012,0.012,0,...] while velocity_hrl/v0 (A1) ships all zeros. Those
    // offsets move the foot site 16.3 mm in x, so a replay that assumes the wrong one is
    // silently in the wrong coordinates. Same failure class as the 2026-08-05
    // `deploy_est.yaml` default: recoverable only if the convention travels with the data.
    void init(const std::string& base_path, const std::vector<float>& joint_ids_map,
              float tilt_limit, float fall_acc_thresh, float control_dt,
              bool with_estimator = false,
              const std::vector<float>& joint_offset = {})
    {
        base_ = base_path;
        if (base_.empty()) return;
        n_ = (int)joint_ids_map.size();
        dt_ = control_dt;
        with_est_ = with_estimator;
        buf_ << std::setprecision(4);  // significant digits, not decimal places
        // Decimate quiet ticks to ~500 Hz (control loop runs at 1/control_dt, e.g. 1000 Hz).
        // record() ignores this factor and always logs a tick when a safety trigger fires,
        // so this only shrinks the file -- it never drops a joint clamp or a tilt/fall event.
        constexpr float kTargetLogHz = 500.0f;
        log_every_ = std::max(1, (int)std::lround((1.0f / kTargetLogHz) / control_dt));
        g_safety_logger_entry++;  // every entry gets a new id, incl. the first (starts at 0)
        std::ifstream existing(base_ + ".csv");
        bool first_entry = !existing.good();
        existing.close();
        if (!first_entry) return;  // re-entry (this type or the other): append, keep tick_
        tick_ = 0;

        // --- meta json (single source of truth for the analyzer) ---
        std::ofstream meta(base_ + "_meta.json");
        meta << "{\n";
        meta << "  \"num_joints\": " << n_ << ",\n";
        meta << "  \"control_dt\": " << control_dt << ",\n";
        meta << "  \"tilt_limit\": " << tilt_limit << ",\n";
        meta << "  \"fall_acc_thresh\": " << fall_acc_thresh << ",\n";
        meta << "  \"with_estimator\": " << (with_est_ ? "true" : "false") << ",\n";
        meta << "  \"joint_offset\": [";
        for (size_t i = 0; i < joint_offset.size(); i++)
            meta << (i ? "," : "") << joint_offset[i];
        meta << "],\n";
        meta << "  \"joints\": [\n";
        for (int i = 0; i < n_; i++) {
            int jid = (int)joint_ids_map[i];
            // tau_trip/dq_trip ride along per joint (2026-08-05) so the analyzer can score
            // the trip channel without re-deriving the table: tau_est is NOT a CSV column
            // (27 more at 500 Hz), so unlike the position clamp the trip cannot be
            // reconstructed from the row alone -- the thresholds have to travel with it.
            meta << "    {\"slot\": " << i << ", \"hw_id\": " << jid
                 << ", \"name\": \"" << h1_2_joint_names[jid] << "\""
                 << ", \"min\": " << h1_2_joint_limits[jid].min
                 << ", \"max\": " << h1_2_joint_limits[jid].max
                 << ", \"tau_trip\": " << h1_2_joint_trips[jid].tau
                 << ", \"dq_trip\": " << h1_2_joint_trips[jid].dq << "}"
                 << (i < n_ - 1 ? "," : "") << "\n";
        }
        meta << "  ]\n}\n";
        meta.close();

        // --- csv header ---
        buf_.str("");
        // t      = tick counter x nominal control_dt. NOT a clock: it advances one dt per
        //          run() iteration regardless of how long that iteration actually took.
        // t_wall = real elapsed seconds (steady_clock) since the first record() of this
        //          process. Added 2026-08-04: the loop rate had never been directly
        //          measurable from this file, so a print artifact in `t` was mistaken for
        //          a control stall and had to be disproved indirectly, by comparing
        //          bridge_session.py's scripted hold durations against `t`. With both
        //          columns present, rate = d(t_wall)/d(t) is read straight off the CSV.
        buf_ << "t,t_wall";
        for (int i = 0; i < n_; i++) buf_ << ",raw_q" << i;
        for (int i = 0; i < n_; i++) buf_ << ",meas_q" << i;
        for (int i = 0; i < n_; i++) buf_ << ",meas_dq" << i;
        buf_ << ",quat_w,quat_x,quat_y,quat_z,acc_x,acc_y,acc_z";
        // cmd_* = commanded twist (keyboard/joystick, both FSMs). ach_* = ground-truth body-
        // frame base velocity from the sim bridge's SportModeState (2026-07-21, WL-B0) — SIM
        // ONLY, added for sim-to-sim validation (e.g. A0 achieved-vx) and for correlating a
        // fall with what was commanded at the time (the flight recorder previously had no
        // command column at all). Reads 0 on real hardware: same estimator gap as E1, the
        // SportModeState publisher goes silent once a custom low-level controller has command.
        buf_ << ",cmd_vx,cmd_vy,cmd_wz,ach_vx,ach_vy,ach_vz";
        // phase_sin/phase_cos = the gait-clock observation AS THE POLICY SAW IT (post
        // stand-mask), from the robot-local gait_phase_cmd term (2026-07-21, defect 0).
        // (0,0) under cmd 0 is correct; (0,0) while a command is held means the clock is
        // dead, which is precisely the bug the joystick-hardcoded shared term caused.
        buf_ << ",phase_sin,phase_cos";
        // trig_joint MEANING (A1 since 2026-07-16, A0 since 2026-07-23): "at least one
        // commanded joint was clamped to its h1_2_limits.h bound this tick", with NO hold
        // effect. It no longer contributes to alpha, which is now a pure tilt/fall
        // indicator for both FSM types. Older CSVs mean the opposite ("a MEASURED joint
        // left range -> whole-body hold engaged"), so trig_joint=1 with alpha>0 in an A0
        // capture dated before 2026-07-23 is the OLD semantics — do not mix the two eras.
        // trig_torque / trig_dq (2026-08-05): at least one joint over its |tau_est| or |dq|
        // trip threshold this tick (h1_2_limits.h). Unlike trig_tilt/trig_fall these drive a
        // PER-JOINT ramped hold, like trig_joint's clamp -- so `alpha` (whole-body tilt/fall)
        // stays exactly what it always meant and `alpha_trip` carries the new channel: the
        // max per-joint trip ramp this tick. trip_jid/trip_ratio name the closest joint to
        // its threshold and how close, logged on EVERY tick rather than only while tripping:
        // tau_est is not a column, so without the ratio a clean session cannot tell you
        // whether it cleared by 3x or by 2%. max(trip_ratio) over a clean run is the
        // headroom. Per-joint thresholds are in <base>_meta.json.
        // Columns absent from any CSV written before 2026-08-05.
        buf_ << ",alpha,trig_joint,trig_tilt,trig_fall";
        buf_ << ",trig_torque,trig_dq,trip_jid,trip_ratio,alpha_trip,entry";
        // Estimator channel (2026-08-10, A1 only) -- see EstSample. These are the ONLY
        // columns on the row sampled at the DDS rate rather than the 50 Hz policy rate,
        // and the only ones written at full precision; everything above stays at 4 sig
        // figs so dd8ecfc's file-size fix is preserved.
        if (with_est_) {
            buf_ << ",est_acc_x,est_acc_y,est_acc_z";
            buf_ << ",est_quat_w,est_quat_x,est_quat_y,est_quat_z";
            buf_ << ",est_gyro_x,est_gyro_y,est_gyro_z";
            for (int i = 0; i < 13; i++) buf_ << ",est_q" << i;
            for (int i = 0; i < 13; i++) buf_ << ",est_dq" << i;
            buf_ << ",est_dpsi";
            for (int a = 0; a < 7; a++)
                buf_ << ",est_v_" << kEstArmNames[a] << "_x"
                     << ",est_v_" << kEstArmNames[a] << "_y";
        }
        buf_ << '\n';
        // Truncate/create the file and write the header now (first entry only).
        std::ofstream(base_ + ".csv", std::ios::trunc) << buf_.str();
        buf_.str("");
        rows_since_flush_ = 0;
    }

    // Append one tick. Cheap: string formatting only, no file I/O until flush.
    void record(const std::vector<float>& raw_q,
                const float* meas_q, const float* meas_dq,
                const float quat[4], const float acc[3],
                float alpha, bool trig_joint, bool trig_tilt, bool trig_fall,
                const float cmd[3], const float ach_vel[3], const float phase[2],
                bool trig_torque = false, bool trig_dq = false, int trip_jid = -1,
                float alpha_trip = 0.0f, float trip_ratio = 0.0f,
                const EstSample* est = nullptr)
    {
        if (base_.empty()) return;
        long this_tick = tick_++;
        // alpha > 0 covers the hold ramp-down tail (up to H1_2_RAMP_CYCLES ticks after
        // trig_joint/trig_tilt/trig_fall all clear) -- without it, decimation could drop
        // rows from that tail and undercount engaged_ticks/engaged_time_s downstream.
        // alpha_trip does the same job for the per-joint trip ramp; without it the decimator
        // would sample the trip channel at 500 Hz and could miss a short trip entirely.
        bool trig = trig_joint || trig_tilt || trig_fall || alpha > 0.0f
                    || trig_torque || trig_dq || alpha_trip > 0.0f;
        if (!trig && (this_tick % log_every_) != 0) return;  // decimated quiet tick, skip
        // `t` needs its OWN precision, not the buffer's 4 SIGNIFICANT digits (set in
        // init()): 4 sig digits quantises time to 0.01 s once t passes 10 s, which made
        // every dt read off this column past that point either 0.00 or exactly 10.00 ms.
        // That artifact was misread as a 100 Hz control stall on 2026-08-03 and briefly
        // "falsified" the 1 kHz assumption; the loop was measured at 989 Hz afterwards by
        // comparing scripted wall-clock holds against this column. Fixed decimal places
        // here, restored to significant digits for every other float on the row.
        // NOTE this column is still a TICK COUNTER x nominal dt, not a clock -- see t_wall.
        if (!wall0_set_) { wall0_ = std::chrono::steady_clock::now(); wall0_set_ = true; }
        const double t_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0_).count();
        buf_ << std::fixed << std::setprecision(4) << (this_tick * dt_) << ',' << t_wall
             << std::defaultfloat << std::setprecision(4);
        for (int i = 0; i < n_; i++) buf_ << ',' << raw_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_dq[i];
        buf_ << ',' << quat[0] << ',' << quat[1] << ',' << quat[2] << ',' << quat[3];
        buf_ << ',' << acc[0]  << ',' << acc[1]  << ',' << acc[2];
        buf_ << ',' << cmd[0] << ',' << cmd[1] << ',' << cmd[2];
        buf_ << ',' << ach_vel[0] << ',' << ach_vel[1] << ',' << ach_vel[2];
        buf_ << ',' << phase[0] << ',' << phase[1];
        buf_ << ',' << alpha << ',' << (trig_joint?1:0) << ',' << (trig_tilt?1:0)
             << ',' << (trig_fall?1:0);
        buf_ << ',' << (trig_torque?1:0) << ',' << (trig_dq?1:0) << ',' << trip_jid
             << ',' << trip_ratio << ',' << alpha_trip;
        buf_ << ',' << g_safety_logger_entry;
        if (with_est_) {
            // PRECISION IS LOAD-BEARING HERE, unlike every other column. This block exists
            // to be differenced offline at the DDS rate (~500 Hz, measured): 4 sig figs on
            // a ~0.5 rad joint angle is a 1e-4 rad quantum, which differences to ~0.1 rad/s
            // and, through the ~0.8 m leg lever, to ~0.04 m/s of pure quantisation noise --
            // the same order as the signal being measured. 9 sig figs (float round-trip) is
            // the only setting that cannot become the dominant error term later.
            static const EstSample kZero{};
            const EstSample& e = est ? *est : kZero;
            buf_ << std::setprecision(9);
            for (int i = 0; i < 3; i++) buf_ << ',' << e.acc[i];
            for (int i = 0; i < 4; i++) buf_ << ',' << e.quat[i];
            for (int i = 0; i < 3; i++) buf_ << ',' << e.gyro[i];
            for (int i = 0; i < 13; i++) buf_ << ',' << e.q[i];
            for (int i = 0; i < 13; i++) buf_ << ',' << e.dq[i];
            buf_ << ',' << e.dpsi;
            for (int a = 0; a < 7; a++)
                buf_ << ',' << e.v_arm[a][0] << ',' << e.v_arm[a][1];
            buf_ << std::setprecision(4);
        }
        buf_ << '\n';
        // Periodic flush as crash insurance (~5 s at the quiet-tick 500 Hz logged rate; more
        // often during a safety-trigger burst, since those log every tick). Infrequent
        // blocking write.
        if (++rows_since_flush_ >= 2500) flush();
    }

    // Append the RAM buffer to the CSV and clear it. Called periodically and on exit.
    void flush()
    {
        if (base_.empty() || buf_.tellp() <= 0) return;
        std::ofstream(base_ + ".csv", std::ios::app) << buf_.str();
        buf_.str("");
        rows_since_flush_ = 0;
    }

private:
    std::string base_;
    int n_{0};
    float dt_{0.001f};
    long tick_{0};
    int log_every_{1};
    bool with_est_{false};  // estimator channel present (fixed at init, header must match)
    // Wall clock is anchored at the first record(), not at init(): init() runs on FSM
    // entry and a re-entry must not restart the clock while tick_ keeps counting, or
    // d(t_wall)/d(t) goes negative across the seam.
    std::chrono::steady_clock::time_point wall0_{};
    bool wall0_set_{false};
    std::ostringstream buf_;
    int rows_since_flush_{0};
};
