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
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BRIDGE = '/opt/unitree_mujoco/simulate/build/unitree_mujoco'
SETUP = Path.home() / 'ramlab_ws' / 'setup_all_mujoco.sh'
CTRL_DIR = REPO / 'deploy' / 'robots' / 'h1_2' / 'build'
# G2.1 stand -> G2.2 held 0.3/0.5/1.0 -> G2.3 return to stand. `w` is the vx=1.0 range edge.
DEFAULT_SEQ = '0:10,3:12,0:4,5:12,0:4,w:12,0:8'


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


def release_band(key='9'):
  """Toggle the bridge's elastic band: a GLFW key on the mujoco window, so XTEST it."""
  try:
    from Xlib import display, X
    from Xlib.ext import xtest
  except ImportError:
    print('[band] python-xlib not installed -- band NOT released (pip install python-xlib)')
    return False
  d = display.Display()

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
  hit = next((w for n, w in wins if 'mujoco' in n.lower() or 'unitree' in n.lower()), None)
  if hit is None:
    print('[band] no mujoco window found -- band NOT released')
    return False
  hit.set_input_focus(X.RevertToParent, X.CurrentTime)
  hit.configure(stack_mode=X.Above)
  d.sync()
  time.sleep(0.4)
  code = d.keysym_to_keycode(ord(key))
  xtest.fake_input(d, X.KeyPress, code); d.sync(); time.sleep(0.05)
  xtest.fake_input(d, X.KeyRelease, code); d.sync()
  print(f'[band] released (key {key})')
  return True


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
  args = ap.parse_args()

  # A stale controller holds the DDS lowcmd channel and the new one refuses to start
  # ("The other process is using the lowcmd channel") -- it then sits in Passive for the
  # whole session and writes no CSV. Clear both before every run.
  subprocess.run(['pkill', '-f', 'h1_2_ctrl'])
  subprocess.run(['pkill', '-f', 'unitree_mujoco'])
  time.sleep(2)

  env = setup_env()
  env['DISPLAY'] = env.get('DISPLAY', ':0')
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
  fin = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
  ctrl = subprocess.Popen(['./h1_2_ctrl'], cwd=CTRL_DIR, env=env, stdin=fin,
                          stdout=logs['ctrl'], stderr=subprocess.STDOUT)
  fout = os.open(fifo, os.O_WRONLY)
  time.sleep(4)

  def press(k, label):
    os.write(fout, k.encode())
    print(f'[{time.strftime("%T")}] key {k!r}  ({label})')

  def hold(k, secs, label):
    print(f'[{time.strftime("%T")}] hold {k!r} {secs}s  ({label})')
    end = time.time() + secs
    while time.time() < end:      # 50 Hz re-send: beats the 80 ms _key timeout
      os.write(fout, k.encode())
      time.sleep(0.02)

  try:
    press('i', 'FixStand')
    time.sleep(args.stand_s)
    press('o' if args.policy == 'a0' else 'h', f'policy takeover ({args.policy})')
    time.sleep(3)
    if not args.no_band:
      release_band()
    for item in args.seq.split(','):
      k, secs = item.split(':')
      hold(k, float(secs), 'commanded')
    press('p', 'Passive')
    time.sleep(2)
  finally:
    for p in (ctrl, bridge):
      p.send_signal(signal.SIGINT)
    time.sleep(2)
    for p in (ctrl, bridge):
      if p.poll() is None:
        p.kill()
    os.close(fout); os.close(fin)
    fifo.unlink(missing_ok=True)
    for f in logs.values():
      f.close()

  print(f'[done] {base}.csv')
  if not args.no_analyze and (Path(f'{base}.csv')).exists():
    subprocess.run(['python', str(REPO / 'scripts' / 'deploy_gate_analyzer.py'), f'{base}.csv'])


if __name__ == '__main__':
  main()
