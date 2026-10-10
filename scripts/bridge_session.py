"""Scripted LIVE-bridge session: drives /opt/unitree_mujoco + the real `h1_2_ctrl` binary.

The counterpart to `bridge_replica.py`, and NOT a substitute for it — they test different
halves. The replica re-implements the controller side in python, so it is cheap and
deterministic but structurally blind to C++ bugs. This script runs the ACTUAL shipped
binary over DDS against the actual bridge, which is the only way to validate a change to
`State_RLBase` / `State_RLHRL` (safety filter, observation assembly, FSM) before hardware.
Born 2026-07-23 while porting the A0 joint-limit clamp, where the replica could not help.

What it automates (previously a human at the keyboard):
  1. launches the bridge and the controller, controller stdin fed from a FIFO
  2. presses the FSM keys (`i` FixStand -> `o`/`h` policy -> `p` Passive)
  3. releases the sim elastic band (bridge GUI key `9`) via XTEST, since the band is a
     GLFW keypress on the mujoco window and nothing else can toggle it
  4. holds each velocity command for a fixed duration, then runs `deploy_gate_analyzer.py`
  5. records the _Joint telemetry log_ beside the flight recorder (2026-10-02): `read_all_joints`
     is the only source of `tau_est`, and a Run without it loses `cot` / `mech_power_w` to a
     scorer WARN. Started here, gated BEFORE the command sequence, `--no-joint-telemetry` skips.

KEYBOARD LATCH (deploy plan "Defect 1") — the reason `--seq` holds rather than presses:
`Keyboard::_read()` (deploy/include/isaaclab/devices/keyboard/keyboard.h) clears `_key` to
"" after an 80 ms select() timeout, so a velocity key must be RE-SENT continuously or
`keyboard_velocity_commands` reads "" and returns [0,0,0]. A real keyboard's X key-repeat
does this for you; a piped one-shot char does not. FSM keys are exempt (the transition
fires on the transient). Confirmed the hard way: the first automated run transitioned
states correctly and logged cmd [0,0,0] for the entire session.

Requires `python-xlib` for the band release (`pip install python-xlib`); use `--no-band`
to skip it (the robot then stays on the harness and cannot walk — FixStand is not
self-stable on any plant, so the band is what holds it up until policy takeover).

Examples:
  python scripts/bridge_session.py --tag g2_postfix
  python scripts/bridge_session.py --policy hrl --seq "0:10,5:15,0:5" --tag hrl_check
"""

import argparse
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESTAND_FALL_HEIGHT = 0.9  # matches deploy_gate_analyzer.FALL_HEIGHT
BRIDGE = '/opt/unitree_mujoco/simulate/build/unitree_mujoco'
SETUP = Path.home() / 'ramlab_ws' / 'setup_all_mujoco.sh'
CTRL_DIR = REPO / 'deploy' / 'robots' / 'h1_2' / 'build'
TRAJ_DIR = Path.home() / 'ramlab_ws' / 'trajectories'   # where read_all_joints writes
# Full G2.2 + G2.3 battery (2026-08-04). ~253 s. Every command change is a scored
# TRANSITION window in deploy_gate_analyzer -- steady holds alone miss decel failures, and
# the 2026-08-03 fall happened on the 0.5 -> 0 deceleration, not inside any hold.
#   line 1: G2.2 forward parity holds (>=30 s each) + G2.3's 0->1.0-from-stand, the case
#           the plan calls the historical bridge killer
#   line 2: backward + strafe both ways        line 3: yaw both ways
#   line 4: G2.3 walk -> turn -> stop
# DEVIATION ON RECORD: the keyboard map (h1_2_observations.h:36-53) has no +-0.3 lateral
# preset -- s=-0.5, a/d=+-0.5, q/e=+-0.5 -- so G2.2's backward/strafe legs run at +-0.5,
# STRICTER than the gate specifies. Not silently weakened, and not "fixed" by editing the
# key map, which would change what every historical capture means.
# The 3:30 hold is also what the analyzer's band-release detector needs (>=10 s at >=0.3).
DEFAULT_SEQ = ('0:8,3:30,0:6,5:30,0:6,w:30,0:8,'
               's:15,0:6,a:15,0:6,d:15,0:6,'
               'q:15,0:6,e:15,0:6,'
               '5:12,q:8,0:10')


OBS_H = REPO / 'deploy' / 'robots' / 'h1_2' / 'include' / 'h1_2_observations.h'


def deploy_limit_rows():
  """`(lo, hi, name)` per DEPLOY slot, in slot order, from the audited limit header."""
  hdr = (REPO / 'deploy' / 'robots' / 'h1_2' / 'include' / 'h1_2_limits.h').read_text()
  blk = hdr.split('h1_2_joint_limits = {{', 1)[1].split('}};', 1)[0]
  rows = re.findall(r'\{\s*(-?[\d.]+)f?,\s*(-?[\d.]+)f?\s*\}\s*,?\s*//\s*(\d+)\s+(\w+)', blk)
  assert [int(i) for _, _, i, _ in rows] == list(range(len(rows))), 'limit header slots out of order'
  return [(float(a), float(b), n.lower()) for a, b, _, n in rows]


def key_commands():
  """The controller's keyboard presets, `key -> (vx, vy, wz)`, read from the header the binary
  is compiled from so a schedule replayed elsewhere cannot drift from what the bridge runs."""
  rows = re.findall(r'\{"(\w)",\s*\{\s*(-?[\d.]+)f,\s*(-?[\d.]+)f,\s*(-?[\d.]+)f\s*\}\}',
                    OBS_H.read_text())
  return {k: (float(x), float(y), float(z)) for k, x, y, z in rows}


def seq_commands(spec, step_dt=0.02):
  """A `--seq` spec -> one `(vx, vy, wz)` per control step: the command schedule a bridge Run
  holds, for `play.py --eval-cmd-seq` to replay at the training bench."""
  keys = key_commands()
  out = []
  for item in spec.split(','):
    k, secs = item.split(':')
    out += [keys[k]] * round(float(secs) / step_dt)
  return out


def joints_logging(proc, before, wait_s=2.0):
  """The csv `read_all_joints` is writing, or None when it is not alive and growing.

  The node takes no path argument and its INFO stream is unusable as a source (it prints all
  27 joints of every message: 207 MB of log for a 70 s session), so it runs at WARN and its
  file is the ONE `all_joints_*.csv` that did not exist in `before`. That is only sound
  because stray nodes are cleared at session start: an orphan from an aborted session kept
  logging on 2026-10-02 and bench_flight_recorder paired the next Run with ITS file.
  """
  new = sorted(set(TRAJ_DIR.glob('all_joints_*.csv')) - before)
  if len(new) != 1:
    return None
  n0 = new[0].stat().st_size
  time.sleep(wait_s)
  return new[0] if proc.poll() is None and new[0].stat().st_size > n0 else None


def setup_env():
  """Re-export what `source setup_all_mujoco.sh` sets (DDS domain/network/config)."""
  out = subprocess.run(['bash', '-c', f'source {SETUP} >/dev/null 2>&1; env'],
                       capture_output=True, text=True).stdout
  env = dict(os.environ)
  for line in out.splitlines():
    if '=' in line:
      k, v = line.split('=', 1)
      env[k] = v
  return env


def _focus_mujoco_window(d, X, label, pid=None):
  """Focus the mujoco window WITHOUT raising it (2026-08-04): XTEST keys are GLFW key
  callbacks, which only fire for the FOCUSED window, but a forced raise would steal the
  desktop during an unattended run. Returns the window on success, None (having already
  printed why) on failure -- the caller must treat None as "key NOT sent", not retry blind:
  the toggle/reset is otherwise silent (no log line anywhere from the bridge itself), so a
  swallowed keypress used to produce a session that looked perfect -- correct FSM
  transitions, CSV written, commands held -- while the robot hung on the harness the whole
  time. An ICONIFIED window is unmapped and cannot hold focus, which is why this asserts
  focus landed rather than trusting the XTEST call to have worked.
  """
  def walk(win, out):
    try:
      if win.get_wm_name():
        out.append((win.get_wm_name(), win))
    except Exception:
      pass
    try:
      for c in win.query_tree().children:
        walk(c, out)
    except Exception:
      pass

  wins = []
  walk(d.screen().root, wins)
  # The simulator's title is 'MuJoCo : Unitree H1-2'. A substring match also hit any editor or
  # terminal titled with the repo path (unitree_rl_mjlab) and typed the key into it (2026-10-09).
  # 2026-10-10: a title prefix is not enough either -- another window titled 'MuJoCo...' took the
  # key with focus verified on IT, and 9 Runs hung on the band. `pid` pins the window to the
  # simulator this session launched (GLFW sets _NET_WM_PID).
  def owner(w):
    try:
      prop = w.get_full_property(d.intern_atom('_NET_WM_PID'), X.AnyPropertyType)
      return prop.value[0] if prop else None
    except Exception:
      return None
  hit = next((w for n, w in wins if str(n).lower().startswith('mujoco') and (pid is None or owner(w) == pid)),
             None)
  if hit is None:
    print(f'[{label}] no mujoco window{f" of pid {pid}" if pid else ""} found -- key NOT sent')
    return None
  hit.set_input_focus(X.RevertToParent, X.CurrentTime)
  d.sync()
  time.sleep(0.4)
  focused = d.get_input_focus().focus
  try:
    ok = focused.id == hit.id or focused.query_tree().parent.id == hit.id
  except Exception:
    ok = False
  if not ok:
    print(f'[{label}] FOCUS DID NOT LAND on the mujoco window -- key NOT sent. '
          'Is the window iconified? It must be mapped (backgrounded is fine).')
    return None
  return hit


def release_band(key='9', pid=None):
  """Toggle the bridge's elastic band: a GLFW key on the mujoco window, so XTEST it. NB
  this is a TOGGLE, not a release: sending it twice re-attaches the band. The analyzer's
  travel check is the backstop that catches a failed release (and every other cause)."""
  try:
    from Xlib import display, X
    from Xlib.ext import xtest
  except ImportError:
    print('[band] python-xlib not installed -- band NOT released (pip install python-xlib)')
    return False
  d = display.Display()
  hit = _focus_mujoco_window(d, X, 'band', pid)
  if hit is None:
    return False
  code = d.keysym_to_keycode(ord(key))
  xtest.fake_input(d, X.KeyPress, code); d.sync(); time.sleep(0.05)
  xtest.fake_input(d, X.KeyRelease, code); d.sync()
  print(f'[band] released (key {key}, focus verified)')
  return True


def reset_sim(pid=None):
  """Backspace on the mujoco window -> `mj_resetData(m, d); mj_forward(m, d)`
  (main.cc:632-634, `user_key_cb`): an actual physics-state reset to the model's initial
  keyframe, not a joint-position hold. This is what makes restand-after-a-real-fall work at
  all -- FixStand ('i') only drives a joint configuration, it cannot move the floating base,
  so it cannot recover a robot that is actually lying on the ground (confirmed 2026-08-24:
  a p->i->h retry loop against a face-planted robot cycled for 150s at gt_h~0.15 without
  ever standing back up). Does NOT touch the elastic band's `enable_` state -- a separate
  global in the bridge process, untouched by mj_resetData -- so a caller that already
  released the band must re-attach it (another `release_band()` toggle) before this leaves
  the robot self-supporting; otherwise the reset robot just topples again during FixStand,
  same as it would at a fresh session start with `--no-band`.
  """
  try:
    from Xlib import display, X, XK
    from Xlib.ext import xtest
  except ImportError:
    print('[reset] python-xlib not installed -- sim NOT reset (pip install python-xlib)')
    return False
  d = display.Display()
  hit = _focus_mujoco_window(d, X, 'reset', pid)
  if hit is None:
    return False
  code = d.keysym_to_keycode(XK.string_to_keysym('BackSpace'))
  xtest.fake_input(d, X.KeyPress, code); d.sync(); time.sleep(0.05)
  xtest.fake_input(d, X.KeyRelease, code); d.sync()
  print('[reset] sim reset (Backspace, focus verified)')
  return True


def walking_ratio(csv):
  """Median true / commanded sagittal speed over the newest forward-walking stretch of a growing
  flight recorder (`ach_vx` is the sim's true velocity), its first 3 s and anything after a tilt
  trigger dropped; None below 2 s of such rows. ~1 walking, ~0 on the harness (2026-10-10: held
  Runs -0.16..0.11, free 0.79..1.07)."""
  import pandas as pd
  try:
    d = pd.read_csv(csv, usecols=['t_wall', 'cmd_vx', 'cmd_vy', 'cmd_wz', 'ach_vx', 'trig_tilt'],
                    on_bad_lines='skip').apply(pd.to_numeric, errors='coerce').dropna()
  except Exception:
    return None
  if (d.trig_tilt != 0).any():
    d = d.loc[:d.index[(d.trig_tilt != 0).to_numpy()][0] - 1]
  fwd = (d.cmd_vx >= 0.2) & (d.cmd_vy.abs() < 0.05) & (d.cmd_wz.abs() < 0.05)
  if not fwd.any():
    return None
  seg = (fwd != fwd.shift()).cumsum()
  w = d[fwd & (seg == seg[fwd].iloc[-1])]
  w = w[w.t_wall - w.t_wall.iloc[0] > 3.0]
  if len(w) == 0 or w.t_wall.iloc[-1] - w.t_wall.iloc[0] < 2.0:
    return None
  return float((w.ach_vx / w.cmd_vx).median())


class FallWatcher:
  """Polls a growing `<base>_hrl.csv` for `gt_h` (sim-bridge ground-truth height,
  from `hrl::Telemetry` -- reads 0 on real hardware, which is what keeps this class
  sim-bridge-only structurally, not just by convention) dropping below
  RESTAND_FALL_HEIGHT, and flags `fallen` so the caller can restand before the next
  phase. Opt-in only (bridge_session.py's default behaviour is unchanged when this
  is never constructed): recovery buys per-phase evidence after a fall, it does not
  touch deploy_gate_analyzer.py's fall gate, which stays blocking regardless.
  """

  def __init__(self, hrl_csv: Path, poll_s: float = 0.5):
    self.hrl_csv = hrl_csv
    self.poll_s = poll_s
    self.fallen = threading.Event()
    self._stop = threading.Event()
    self._gt_h_idx = None
    self._thread = threading.Thread(target=self._run, daemon=True)

  def start(self):
    self._thread.start()

  def stop(self):
    self._stop.set()
    self._thread.join(timeout=2)

  def _run(self):
    while not self._stop.is_set():
      try:
        self._poll_once()
      except Exception:
        pass  # a torn read on a mid-flush file is expected; just retry next tick
      time.sleep(self.poll_s)

  def _poll_once(self):
    if not self.hrl_csv.exists():
      return
    with open(self.hrl_csv) as f:
      lines = f.read().splitlines()
    if len(lines) < 2:
      return
    if self._gt_h_idx is None:
      header = lines[0].split(',')
      if 'gt_h' not in header:
        return
      self._gt_h_idx = header.index('gt_h')
    row = lines[-1].split(',')
    if len(row) <= self._gt_h_idx:
      return  # last line torn mid-flush; the next poll re-reads a complete one
    gt_h = float(row[self._gt_h_idx])
    if gt_h < RESTAND_FALL_HEIGHT:
      self.fallen.set()


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--policy', choices=['a0', 'hrl'], default='a0')
  ap.add_argument('--tag', default='session', help='appended to the flight-recorder base name')
  ap.add_argument('--seq', default=DEFAULT_SEQ,
                  help='comma-separated KEY:SECONDS holds, e.g. "0:10,5:12,0:5"')
  ap.add_argument('--stand-s', type=float, default=6.0, help='FixStand settle before takeover')
  ap.add_argument('--no-band', action='store_true', help='leave the elastic band attached')
  ap.add_argument('--no-analyze', action='store_true')
  ap.add_argument('--auto-restand', action='store_true',
                  help='sim-bridge only, default off. Poll <base>_hrl.csv\'s gt_h; on a '
                       f'fall (gt_h < {RESTAND_FALL_HEIGHT}) hold zero command briefly, '
                       'then Backspace (a real mj_resetData physics reset, not a joint '
                       'hold -- FixStand alone cannot recover a robot actually lying on '
                       'the ground) and resume, before the next command phase, so one '
                       'fall does not cost every later phase. The policy is never taken '
                       'out of control (no FSM cycling). Recovery buys per-phase '
                       'evidence, not forgiveness -- it does not touch '
                       'deploy_gate_analyzer.py, so any fall anywhere still means NO-GO.')
  ap.add_argument('--deploy-cfg', default=None,
                  help='override H1_2_DEPLOY_CFG (default: whatever setup_all_mujoco.sh '
                       'exports, deploy.yaml). Filename must exist under the policy_dir\'s '
                       'params/ directory. For testing a params variant against the sim '
                       'bridge without touching the shipped deploy.yaml/deploy_real.yaml.')
  ap.add_argument('--no-joint-telemetry', action='store_true',
                  help='do not launch `read_all_joints` beside the session. The Run then has '
                       'no tau_est, so bench_flight_recorder omits cot and mech_power_w.')
  args = ap.parse_args()

  # A stale controller holds the DDS lowcmd channel and the new one refuses to start
  # ("The other process is using the lowcmd channel") -- it then sits in Passive for the
  # whole session and writes no CSV. Clear both before every run.
  subprocess.run(['pkill', '-f', 'h1_2_ctrl'])
  subprocess.run(['pkill', '-f', 'unitree_mujoco'])
  if not args.no_joint_telemetry:
    subprocess.run(['pkill', '-INT', '-f', 'read_all_joints'])   # see joints_logging()
  time.sleep(2)

  env = setup_env()
  env['DISPLAY'] = env.get('DISPLAY', ':0')
  if args.deploy_cfg:
    env['H1_2_DEPLOY_CFG'] = args.deploy_cfg
    print(f'[cfg] H1_2_DEPLOY_CFG overridden -> {args.deploy_cfg}')
  ts = time.strftime('%Y-%m-%d_%H-%M-%S')
  base = REPO / 'logs' / 'deploy_safety' / f'{ts}_{args.tag}'
  base.parent.mkdir(parents=True, exist_ok=True)
  env['H1_2_SAFETY_LOG'] = str(base)
  print(f'[log] {base}.csv')

  fifo = Path(f'/tmp/bridge_session_{os.getpid()}.fifo')
  if fifo.exists():
    fifo.unlink()
  os.mkfifo(fifo)

  logs = {}
  for name in ('bridge', 'ctrl'):
    logs[name] = open(f'{base}_{name}.log', 'w')

  bridge = subprocess.Popen([BRIDGE], env=env, stdout=logs['bridge'], stderr=subprocess.STDOUT)
  time.sleep(6)
  joints = None
  if not args.no_joint_telemetry:
    # Own process group: `ros2 run` does not forward a SIGINT sent to it alone (it relies on
    # the terminal signalling the whole group), so the node would outlive the session.
    logs['joints'] = open(f'{base}_joints.log', 'w')
    joints_before = set(TRAJ_DIR.glob('all_joints_*.csv'))
    try:
      joints = subprocess.Popen(['ros2', 'run', 'h1_2_low_level_controller', 'read_all_joints',
                                 '--ros-args', '--log-level', 'warn'],
                                env=env, stdout=logs['joints'], stderr=subprocess.STDOUT,
                                start_new_session=True)
    except FileNotFoundError:
      bridge.kill()
      raise SystemExit('[joints] ABORTING: `ros2` is not on PATH after setup_all_mujoco.sh')
  fin = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
  ctrl = subprocess.Popen(['./h1_2_ctrl'], cwd=CTRL_DIR, env=env, stdin=fin,
                          stdout=logs['ctrl'], stderr=subprocess.STDOUT)
  fout = os.open(fifo, os.O_WRONLY)
  time.sleep(4)

  def press(k, label):
    os.write(fout, k.encode())
    print(f'[{time.strftime("%T")}] key {k!r}  ({label})')

  def hold(k, secs, label, watcher=None):
    print(f'[{time.strftime("%T")}] hold {k!r} {secs}s  ({label})')
    end = time.time() + secs
    while time.time() < end:      # 50 Hz re-send: beats the 80 ms _key timeout
      if watcher is not None and watcher.fallen.is_set():
        print(f'[{time.strftime("%T")}] fall detected mid-hold -- cutting this phase short')
        return
      os.write(fout, k.encode())
      time.sleep(0.02)

  policy_key = 'o' if args.policy == 'a0' else 'h'
  restand_count = [0]

  def restand():
    # 0 (settle) -> Backspace (mj_resetData, a real physics reset -- see reset_sim()) ->
    # settle -> resume. The policy is NEVER taken out of control (no p/i/band cycling):
    # it stayed in State_RLHRL the whole time, so a fresh reset pose under a held zero
    # command is just a stand it -- the same thing a clean cmd-0 hold does at session
    # start. NOT p -> i -> policy re-entry: FixStand only holds a joint configuration, it
    # cannot move the floating base, so it cannot recover a robot that is actually lying on
    # the ground -- confirmed 2026-08-24, that approach cycled 15x over 150s at gt_h~0.15
    # and never stood up. hrl::Telemetry keeps logging through the reset (no FSM re-entry
    # here, so `entry` does not bump -- the reset is invisible to it by design, same as any
    # other mid-episode teleport), and `t`/`t_wall` just show the same discontinuity a
    # cmd-0 hold would.
    restand_count[0] += 1
    print(f'[{time.strftime("%T")}] [restand] #{restand_count[0]}: 0 -> reset -> resume')
    hold('0', 3.0, 'zero cmd before reset (auto-restand)')
    reset_sim(bridge.pid)
    time.sleep(2)
    watcher.fallen.clear()

  watcher = None
  joints_csv = None
  try:
    if joints is not None:
      # FAIL-CLOSED before the sequence, same reasoning as the band below: a dead node is
      # otherwise discovered by the scorer, after the Run was spent.
      joints_csv = joints_logging(joints, joints_before)
      if joints_csv is None:
        print(f'[joints] ABORTING before the command sequence: read_all_joints is not logging '
              f'(alive={joints.poll() is None}); see {base}_joints.log')
        raise SystemExit(1)
    if ctrl.poll() is not None:
      # A controller that died at start-up (e.g. a deploy config missing from one of the two
      # policy dirs, 2026-10-03) used to leave the session holding keys into a dead FIFO for
      # the whole schedule and exit 0 with no flight recorder.
      raise SystemExit(f'[ctrl] ABORTING: h1_2_ctrl exited (rc={ctrl.returncode}) before the '
                       f'schedule; see {base}_ctrl.log')
    press('i', 'FixStand')
    time.sleep(args.stand_s)
    press(policy_key, f'policy takeover ({args.policy})')
    time.sleep(3)
    # FAIL-CLOSED (2026-08-06). release_band() already verifies focus landed and returns
    # False when it did not -- but the return value used to be discarded, so a failed
    # release ran the FULL sequence anyway and produced a session with the robot hanging on
    # the harness: correct FSM transitions, CSV written, commands held, and not one word of
    # it interpretable. That happened during the leg-odometry A/B (a click stole focus
    # inside release_band's 0.4 s window) and cost a run of a 9-run battery. The analyzer's
    # travel check catches it afterwards, which is the backstop; this stops us paying 83 s
    # and a slot to learn nothing.
    if not args.no_band and not release_band(pid=bridge.pid):
      raise SystemExit(
        '[band] ABORTING before the command sequence: the elastic band was NOT released, '
        'so the robot is still on the harness and nothing it does is interpretable. '
        'Keep the mujoco window mapped and do not click away while the session starts '
        '(the release needs focus for ~0.5 s). Re-run this session; pass --no-band only '
        'if you deliberately want a harnessed run.')
    if args.auto_restand:
      # gt_h only exists in <base>_hrl.csv (hrl::Telemetry, A1 only) and only reads
      # meaningfully on the sim bridge -- see FallWatcher's docstring. For an A0 session
      # the file never appears, so the watcher polls harmlessly and never fires.
      watcher = FallWatcher(Path(f'{base}_hrl.csv'))
      watcher.start()
    travel_ok = args.no_band
    for item in args.seq.split(','):
      k, secs = item.split(':')
      hold(k, float(secs), 'commanded', watcher=watcher)
      if not travel_ok and (r := walking_ratio(Path(f'{base}.csv'))) is not None:
        # The backstop for every way a release can fail silently (2026-10-10): focus can be verified
        # on the right window and the band still be on. Checked after the first forward hold.
        print(f'[band] travel check: true / commanded sagittal speed {r:.2f}')
        if r < 0.5:
          raise SystemExit('[band] ABORTING: the robot did not walk forward under a forward command, '
                           'so it is still on the harness. Re-run this session.')
        travel_ok = True
      if watcher is not None and watcher.fallen.is_set():
        restand()
    press('p', 'Passive')
    time.sleep(2)
  finally:
    if watcher is not None:
      watcher.stop()
    for p in (ctrl, bridge):
      p.send_signal(signal.SIGINT)
    if joints is not None and joints.poll() is None:
      os.killpg(joints.pid, signal.SIGINT)
    time.sleep(2)
    if joints is not None and joints.poll() is None:
      os.killpg(joints.pid, signal.SIGKILL)
    for p in (ctrl, bridge):
      if p.poll() is None:
        p.kill()
    os.close(fout); os.close(fin)
    fifo.unlink(missing_ok=True)
    for f in logs.values():
      f.close()
  if not travel_ok:
    print('[band] WARNING: travel check never ran (no forward hold, or a fall before one); '
          'the band release is unverified')
  if restand_count[0]:
    print(f'[restand] {restand_count[0]} recovery/recoveries this session -- the run is '
          f'still NO-GO if any phase fell (recovery buys per-phase evidence, not '
          f'forgiveness); see deploy_gate_analyzer.py output above.')

  if joints_csv is not None:
    with open(joints_csv) as f:
      rows = sum(1 for _ in f) - 1
    print(f'[joints] {joints_csv.name} rows={rows}')
    if rows <= 0:
      raise SystemExit(f'[joints] {joints_csv.name} has no rows -- the Run has no tau_est')
  print(f'[done] {base}.csv')
  if not args.no_analyze:
    # Both CSVs, not just the base one (D1, fixed 2026-08-04). An HRL session writes TWO
    # files and this used to hand over only `<base>.csv`, so `_hrl.csv` -- the one with
    # height, the pre-computed act_rate/period, and since 2026-08-03 the est_*/gt_*
    # estimator columns -- was never read at all. They are complementary, not
    # alternatives: the base CSV carries the safety triggers and the A0-comparable
    # columns, the hrl CSV carries everything the hierarchy adds.
    for csv_path in (f'{base}_hrl.csv', f'{base}.csv'):
      if Path(csv_path).exists():
        print(f'\n=== analyzing {Path(csv_path).name}')
        subprocess.run(['python', str(REPO / 'scripts' / 'deploy_gate_analyzer.py'), csv_path])


if __name__ == '__main__':
  main()
