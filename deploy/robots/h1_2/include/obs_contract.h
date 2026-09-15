#pragma once
// A0 (State_RLBase) deploy I/O contract, checked ONCE at load (2026-09-15).
//
// THE DEFECT. isaaclab::OrtRunner::act (deploy/include/isaaclab/algorithms/algorithms.h)
// builds each input tensor from the observation vector's DATA POINTER but the ONNX's
// declared SIZE. A vector shorter than the model expects is therefore not an error: ORT reads
// the missing floats out of adjacent heap, the policy emits confident garbage, and the robot
// goes limp via the safety hold with nothing in the log naming the cause. The output side is
// the same shape: ActionManager::process_action slices `action.begin() + idx + action_dim`, so
// a model emitting fewer actions than the joint map iterates past the end of the vector.
//
// WHY AT LOAD, AND ONLY AT LOAD. The HRL state has a second, per-step layer because its HL
// input can change shape at run time (hl_obs_e adds the latent). A0's cannot: the observation
// terms are fixed by the deploy yaml and the model is loaded once, in the constructor. So the
// realistic failure -- a hand-copied policy.onnx of the wrong width, which is how this slot is
// actually filled -- is fully visible here, before any thread starts. A load-time throw is also
// the correct fail-closed action (see State_RLHRL.h): the constructor runs before the policy
// thread exists, and OrtRunner's own Ort::Session already throws from this exact spot on a
// missing or corrupt file, so this adds no new class of refusal, only a sharper one.
// A per-step check on A0 would need the shared isaaclab layer edited (the thread, step() and
// act() all live there), and calling compute() from anywhere else is unsafe: it re-runs every
// term function, gait_phase_cmd included.
//
// Dependency-free by design (no onnxruntime, no Eigen, no yaml) so test/obs_contract_test.cpp
// can pin it with a bare g++, the same way adapt_encoder_test.cpp pins the WP5d contract.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace h1_2
{

struct OnnxTensor
{
    std::string name;
    int64_t size;  // flattened: [1, 92] -> 92
};

using ObsMap = std::unordered_map<std::string, std::vector<float>>;

// Empty string = the contract holds. Otherwise a message that says what was built, what the
// model declares, and what would have happened -- the one thing the heap read never says.
inline std::string io_contract_violation(const std::vector<OnnxTensor>& inputs,
                                         const ObsMap& obs,
                                         int64_t onnx_output_size,
                                         int64_t action_dim)
{
    if (inputs.empty())
        return "policy.onnx declares no inputs";
    for (const auto& in : inputs) {
        const auto it = obs.find(in.name);
        if (it == obs.end()) {
            // OrtRunner::act would THROW this from the policy thread, which has no handler:
            // std::terminate, lowcmd stops, robot left to the DDS timeout. Fail it here instead.
            std::string have;
            for (const auto& kv : obs) have += (have.empty() ? "" : ", ") + kv.first;
            return "policy.onnx input '" + in.name + "' has no observation group of that name "
                   "(built: " + have + ") -- the policy thread would throw it with no handler";
        }
        const int64_t built = (int64_t)it->second.size();
        if (built != in.size)
            return "policy.onnx input '" + in.name + "' declares " + std::to_string(in.size) +
                   " floats but the deploy yaml's observation terms build " +
                   std::to_string(built) + (built < in.size
                       ? " -- ORT would read " + std::to_string(in.size - built) +
                         " floats of ADJACENT HEAP as observations"
                       : " -- ORT would silently drop the last " + std::to_string(built - in.size)) +
                   ". Wrong export in the A0 slot, or the yaml's obs terms changed.";
    }
    if (onnx_output_size != action_dim)
        return "policy.onnx emits " + std::to_string(onnx_output_size) +
               " actions but the action manager expects " + std::to_string(action_dim) +
               " -- process_action would slice past the end of the action vector";
    return {};
}

}  // namespace h1_2
