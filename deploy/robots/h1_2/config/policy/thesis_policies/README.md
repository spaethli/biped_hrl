# Thesis policies

The nine policies evaluated in the thesis and deployed on the sim bridge and the robot: three
architectures, three training seeds each (42, 123, 7).

| Folder | Report name | Code tag | Nets |
|---|---|---|---|
| `flat_s42`, `flat_s123`, `flat_s7` | Flat | A0 | `policy.onnx` |
| `hierarchical_s42`, `hierarchical_s123`, `hierarchical_s7` | Hierarchical | A1a | `high_level.onnx`, `low_level.onnx` |
| `hierarchical_dr_s42`, `hierarchical_dr_s123`, `hierarchical_dr_s7` | Hierarchical + DR | A1a_DR | `high_level.onnx`, `low_level.onnx` |

`hierarchical_s42` is the A1a keeper, the policy deployed on the robot (ADR-0013). Each folder's
`MANIFEST.json` names the training run, seed, training task, md5 of each net, the deploy directory
it goes into and the deploy config it ran with (`deploy_est.yaml` for all nine).

## Deploying one

The C++ controller reads its nets from the policy directory's `exported/` folder. Copy a policy in:

```bash
P=deploy/robots/h1_2/config/policy
cp $P/thesis_policies/flat_s123/policy.onnx              $P/velocity/v0/exported/       # flat
cp $P/thesis_policies/hierarchical_s7/*_level.onnx       $P/velocity_hrl/v0/exported/   # hierarchical
```

then run the controller as in the top-level README (sim first). The hierarchy's goal scale and
structure travel inside the ONNX metadata, so no yaml edit is needed to switch seeds.

## Provenance

Each net is the training run's own export, which is exactly what `scripts/deploy_provenance.py
--stage` copied into `exported/` for every bridge and robot Run (`data/2026-10-02-chain-seq/
driver_bridge.sh`). On 2026-11-09 every checkpoint was re-exported from a scratch copy and compared:
the weights are identical to `model_10000.pt` for all nine. The files differ from a fresh
`play.py --export-onnx` only in metadata (the hierarchy's training-time exports predate the
`hl_obs_e`/`hl_e_dim` keys, whose absence means off; the flat training-time export carries joint
and gain metadata that `play.py` does not attach).
