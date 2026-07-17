"""Headless replica of the /opt/unitree_mujoco C++ bridge (deploy plan, born 2026-07-15).

Reproduces the bridge loop without the GUI: explicit torque PD at the scene physics dt,
optional feedback delay (the bridge's DDS/thread latency, the destabilizer that killed the
training-nominal plant), the elastic band (k=200 d=100 anchor (0,0,3)), and the deployed
policy ONNX in the loop (A0 single net, or the A1 two-ONNX hierarchy incl. velgoal +
HL-owned cadence + pelvis-frame goal state). Gains/scales/hrl params come from the deploy
YAMLs and the exported/ ONNX, so this tests what actually ships.

Chain per run: FixStand(band) -> policy takeover(band) -> stand(free) -> step to
--step-vx from stand -> stop -> walk --walk-vx. Prints one line per phase +
`[REPLICA] {json}`.

Uses (validated 2026-07-15/16): plant A/B tests, latency sweeps, pre-session sanity on a
new candidate checkpoint, velocity-source/wiring discriminators. NOT a substitute for the
real-bridge gates (it re-implements the controller side in python; the C++ path is what
G2 validates).

Per-phase swing clearance (WL-E, 2026-07-16): steps = completed swings (either foot,
>0.1 s air time), clr = mean/max swing apex rise of the ankle_roll body since liftoff.
`--plant nominal` overrides joint armature/frictionloss/damping in-memory to the mjlab
training nominal (per-motor armature, fric 0) without touching the /opt scene XMLs.

Examples:
  python scripts/bridge_replica.py --policy a0 --delay-ms 2
  python scripts/bridge_replica.py --policy hrl --delay-ms 2 --walk-vx 0.5 --step-vx 1.0
  python scripts/bridge_replica.py --policy hrl --scene scene_stress.xml
"""

import argparse
import json

import numpy as np
import mujoco
import onnxruntime as ort
import yaml

SCENE_DIR = '/opt/unitree_mujoco/unitree_robots/h1_2'
DEPLOY = 'deploy/robots/h1_2/config/policy'
FIX_KP = np.array([250,250,250,400,80,40, 250,250,250,400,80,40, 600,
                   250,400,80,80,20,20,20, 250,400,80,80,20,20,20], float)
FIX_KD = np.array([6.5,6.5,6.5,10,1.5,0.4, 6.5,6.5,6.5,10,1.5,0.4, 6,
                   6,10,2,2,0.5,0.5,0.5, 6,10,2,2,0.5,0.5,0.5], float)
FIX_Q = np.array([0,-0.3,0,0.5,-0.2,0, 0,-0.3,0,0.5,-0.2,0, 0,
                  0.28,0,0,0.52,0,0,0, 0.28,0,0,0.52,0,0,0], float)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--policy', choices=('a0', 'hrl'), default='a0')
  p.add_argument('--scene', default='scene.xml', help='scene.xml (vendor) | scene_stress.xml')
  p.add_argument('--delay-ms', type=float, default=2.0, help='feedback latency on q/dq (PD + obs)')
  p.add_argument('--walk-vx', type=float, default=0.5)
  p.add_argument('--step-vx', type=float, default=1.0, help='held command stepped straight from stand')
  p.add_argument('--onnx-dir', default=None, help='override exported/ dir (candidate checkpoints)')
  p.add_argument('--plant', choices=('scene', 'nominal'), default='scene',
                 help='nominal = override joint armature/frictionloss/damping in-memory '
                      'to the mjlab training nominal (WL-E instrument; scene XMLs untouched)')
  args = p.parse_args()

  sub = 'velocity_hrl' if args.policy == 'hrl' else 'velocity'
  cfg = yaml.safe_load(open(f'{DEPLOY}/{sub}/v0/params/deploy.yaml'))
  kp = np.array(cfg['stiffness'], float)
  kd = np.array(cfg['damping'], float)
  offset = np.array(cfg['actions']['JointPositionAction']['offset'], float)
  scale = np.array(cfg['actions']['JointPositionAction']['scale'], float)
  onnx_dir = args.onnx_dir or f'{DEPLOY}/{sub}/v0/exported'
  prov = ['CPUExecutionProvider']
  if args.policy == 'hrl':
    hrl = cfg['hrl']
    c = int(hrl['c'])
    plo, phi = (map(float, hrl['cadence_period_range'])) if hrl.get('hl_cadence') else (0.35, 1.0)
    plo, phi = float(plo), float(phi)
    pin = float(hrl.get('pin_period', 0.0))
    nomh = float(hrl['nominal_root_height'])
    r = cfg['commands']['base_velocity']['ranges']
    gscale = np.array([(r['lin_vel_x'][1]-r['lin_vel_x'][0])/2,
                       (r['lin_vel_y'][1]-r['lin_vel_y'][0])/2,
                       (r['ang_vel_z'][1]-r['ang_vel_z'][0])/2], float)
    ll = ort.InferenceSession(f'{onnx_dir}/low_level.onnx', providers=prov)
    hl = ort.InferenceSession(f'{onnx_dir}/high_level.onnx', providers=prov)
  else:
    net = ort.InferenceSession(f'{onnx_dir}/policy.onnx', providers=prov)

  m = mujoco.MjModel.from_xml_path(f'{SCENE_DIR}/{args.scene}')
  if args.plant == 'nominal':  # h1_2_constants.py training nominal, per-motor armature
    nom_arm = (('hip', 0.025), ('torso', 0.025), ('knee', 0.04), ('ankle', 0.005),
               ('shoulder_yaw', 0.002), ('shoulder', 0.005), ('elbow', 0.002), ('wrist', 0.002))
    for j in range(m.njnt):
      if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
      dof = m.jnt_dofadr[j]
      name = m.joint(j).name
      m.dof_armature[dof] = next(v for k, v in nom_arm if k in name)
      m.dof_frictionloss[dof] = 0.0
      m.dof_damping[dof] = 0.001
  d = mujoco.MjData(m)
  base = m.body('pelvis').id
  imu = m.site('imu').id
  floor = m.geom('floor').id
  foot_b = {s: m.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')}
  foot_g = {s: [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == foot_b[s] and m.geom_contype[g]] for s in foot_b}
  decim = max(1, round(0.02 / m.opt.timestep))
  delay = max(0, round(args.delay_ms / 1000.0 / m.opt.timestep))
  d.qpos[7:] = offset
  d.qpos[2] = 1.05
  mujoco.mj_forward(m, d)
  lowest = min(float(d.geom_xpos[g][2]) - float(m.geom_size[g][0])
               for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] != 0)
  d.qpos[2] -= lowest - 0.002
  mujoco.mj_forward(m, d)
  q0 = d.qpos[7:].copy()

  from collections import deque
  hist = deque([(d.qpos[7:].copy(), d.qvel[6:].copy())] * (delay + 1), maxlen=delay + 1)
  last_a = np.zeros(27)
  tgt = offset.copy()
  target = np.zeros(7)
  phase_acc, period, ks = [0.0], [0.6 if pin <= 0 else pin] if args.policy == 'hrl' else [0.6], [0]

  def frames():
    w, x, y, z = d.qpos[3:7]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    return R, np.array([0, 0, -1.0]) @ R

  def pol(cmd):
    nonlocal last_a, tgt, target
    R, grav = frames()
    q_d, dq_d = hist[0]
    ph = phase_acc[0] if args.policy == 'hrl' else (d.time % 0.6) / 0.6
    pha = np.zeros(2) if np.linalg.norm(cmd) < 0.1 else np.array(
      [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)])
    if args.policy == 'a0':
      obs = np.concatenate([d.qvel[3:6], grav, cmd, pha, q_d-offset, dq_d, last_a]).astype(np.float32)
      a = net.run(None, {'obs': obs[None]})[0][0]
    else:
      v_b = d.qvel[0:3] @ R  # PELVIS root velocity -> body (the 2026-07-15 fix)
      s = np.array([v_b[0], v_b[1], d.qvel[5], grav[0], grav[1], grav[2],
                    float(d.site_xpos[imu][2])])
      policy = np.concatenate([d.qvel[3:6], grav, pha, q_d-offset, dq_d, last_a]).astype(np.float32)
      if ks[0] % c == 0:
        g = hl.run(None, {'obs': np.concatenate([policy, cmd, s[:2]]).astype(np.float32)[None]})[0][0]
        target[:3] = s[:3] + gscale * g[:3]  # delta mode, velocity goal prefix
        target[3:6] = [0, 0, -1.0]
        target[6] = nomh
        if pin <= 0:
          period[0] = plo + (g[3] + 1) * 0.5 * (phi - plo)
      ks[0] += 1
      phase_acc[0] = (phase_acc[0] + 0.02 / period[0]) % 1.0
      a = ll.run(None, {'obs': np.concatenate([policy, target - s]).astype(np.float32)[None]})[0][0]
    last_a = a.copy()
    tgt = offset + scale * a

  results = []
  min_swing = round(0.1 / m.opt.timestep)  # sub-0.1s air time = contact chatter, not a step
  sw = {s: {'air': 0, 'lift_z': 0.0, 'max_z': 0.0} for s in foot_b}
  phases = [('fixstand', 6.0, 'fix', True, (0, 0, 0)),
            ('takeover', 3.0, 'pol', True, (0, 0, 0)),
            ('stand', 5.0, 'pol', False, (0, 0, 0)),
            (f'step{args.step_vx}', 6.0, 'pol', False, (args.step_vx, 0, 0)),
            ('stop', 3.0, 'pol', False, (0, 0, 0)),
            (f'walk{args.walk_vx}', 8.0, 'pol', False, (args.walk_vx, 0, 0))]
  for label, dur, mode, band, cmd in phases:
    n = int(dur / m.opt.timestep)
    qv = []
    apex = []  # swing apex heights (either foot) completed in this phase
    for i in range(n):
      q_d, dq_d = hist[0]
      if mode == 'fix':
        a_ = min(d.time / 3.0, 1.0)
        d.ctrl[:] = FIX_KP * ((1-a_)*q0 + a_*FIX_Q - q_d) + FIX_KD * (0 - dq_d)
      else:
        if i % decim == 0:
          pol(np.array(cmd, float))
        d.ctrl[:] = kp * (tgt - q_d) + kd * (0 - dq_d)
      if band:
        dx = np.array([0, 0, 3]) - d.qpos[:3]
        dist = np.linalg.norm(dx)
        u = dx / dist
        d.xfrc_applied[base, :3] = (200 * dist - 100 * float(d.qvel[:3] @ u)) * u
      else:
        d.xfrc_applied[base, :3] = 0
      mujoco.mj_step(m, d)
      hist.append((d.qpos[7:].copy(), d.qvel[6:].copy()))
      onfloor = set()
      for k in range(d.ncon):
        con = d.contact[k]
        if con.geom1 == floor:
          onfloor.add(con.geom2)
        elif con.geom2 == floor:
          onfloor.add(con.geom1)
      for s, t in sw.items():
        z = float(d.xpos[foot_b[s]][2])
        if any(g in onfloor for g in foot_g[s]):
          if t['air'] >= min_swing:
            apex.append(t['max_z'] - t['lift_z'])
          t['air'] = 0
        else:
          if t['air'] == 0:
            t['lift_z'], t['max_z'] = z, z
          t['air'] += 1
          t['max_z'] = max(t['max_z'], z)
      if i % 10 == 0:
        qv.append(np.sqrt((d.qvel[6:] ** 2).mean()))
    w, x, y, z = d.qpos[3:7]
    pitch = float(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))
    fell = bool(d.qpos[2] < 0.7) or abs(pitch) > 0.5
    rec = {'phase': label, 'qvel_rms': round(float(np.mean(qv)), 3),
           'pitch': round(pitch, 3), 'height': round(float(d.qpos[2]), 3), 'fell': fell,
           'steps': len(apex),
           'clr_mean': round(float(np.mean(apex)), 4) if apex else None,
           'clr_max': round(float(np.max(apex)), 4) if apex else None}
    results.append(rec)
    clr = f"steps={rec['steps']:3d} clr={rec['clr_mean']:.3f}/{rec['clr_max']:.3f}" \
        if apex else 'steps=  0'
    print(f"  {label:12s} qvel_rms={rec['qvel_rms']:.3f} pitch={rec['pitch']:+.3f} "
          f"h={rec['height']:.3f} {clr} {'FELL' if fell else 'ok'}")
    if fell:
      break
  out = {'policy': args.policy, 'scene': args.scene, 'plant': args.plant,
         'delay_ms': args.delay_ms,
         'pass': not any(r['fell'] for r in results), 'phases': results}
  print(f'[REPLICA] {json.dumps(out)}')


if __name__ == '__main__':
  main()
