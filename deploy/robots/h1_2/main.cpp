#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_RLHRL.h"  // emit REGISTER_FSM(State_RLHRL) registrar into main.o so the
                              // static-lib object isn't dropped (else "Unknown FSM type RLHRL")
#include <filesystem>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
// Always initialize keyboard (same as G1). When a gamepad is connected the
// keyboard simply stays idle; when there is no gamepad it is the only input.
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     H1-2 Controller \n";

    // Unitree DDS Config
    // Priority: H1_2_DOMAIN_ID env var (set by setup scripts) > config.yaml > default 0
    const char* domain_env = std::getenv("H1_2_DOMAIN_ID");
    int domain_id = domain_env ? std::stoi(domain_env)
                  : (param::config["domain_id"] ? param::config["domain_id"].as<int>() : 0);
    // Priority: H1_2_NETWORK env var (set by setup scripts) > --network CLI arg > ""
    const char* network_env = std::getenv("H1_2_NETWORK");
    std::string network = network_env ? network_env : vm["network"].as<std::string>();
    spdlog::info("DDS domain={} network={}", domain_id, network.empty() ? "(auto)" : network);
    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network);

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 6;
    // H1-2's ankle is a PARALLEL A/B mechanism: indices 4/5 (and 10/11) mean ankle
    // pitch/roll in PR mode and motor B/A in AB mode. We command pitch/roll, so PR is
    // required. Set explicitly -- we were previously relying on the IDL default
    // (mode_pr_ = 0 == PR), which is not a contract (ADR-0006 Spec C).
    FSMState::lowcmd->msg_.mode_pr() = 0;  // PR
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    bool gamepad = std::filesystem::exists("/dev/input/js0");
    const char* deploy_cfg = std::getenv("H1_2_DEPLOY_CFG");
    bool keyboard_mode = deploy_cfg && std::string(deploy_cfg).find("real") == std::string::npos;
    if (gamepad && !keyboard_mode) {
        std::cout << "Input: Unitree controller  (L2+Up=FixStand, R2+A=Velocity, L2+B=Passive)\n";
    } else {
        std::cout << "Input: Keyboard\n";
        std::cout << "  FSM:      i=FixStand(stand)  o=Velocity(walk)  p=Passive(limp)\n";
        std::cout << "  Velocity: w/s=fwd/bwd  a/d=strafe  q/e=turn\n";
    }

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

