# Dreamer (r2dreamer + highway-env)

The world-model agent uses the DreamerV3 implementation in
[r2dreamer](https://github.com/NM512/r2dreamer) (Morihira et al., ICLR 2026),
commit `becf338`, with plain DreamerV3 selected (`model.rep_loss=dreamer`).

`highway.patch` adds highway-env to r2dreamer. Dreamer's learning code is not modified:

| File in r2dreamer | Change |
|---|---|
| `envs/highway.py` | new: wrapper that turns highway-env into r2dreamer's environment format |
| `configs/env/highway.yaml` | new: settings for `highway-fast-v0` |
| `envs/__init__.py` | 4 added lines that register `highway` as an environment |

## Setup

```bash
git clone https://github.com/NM512/r2dreamer.git
cd r2dreamer
git checkout becf338
pip install -e .
pip install "highway-env==1.12.1" "pandas<3"
git apply /path/to/this/repo/dreamer/highway.patch
```

## Training

```bash
python train.py env=highway model.rep_loss=dreamer buffer.storage_device=cpu \
    env.train_ratio=512 seed=0 trainer.eval_every=5000 logdir=./logdir/highway_dreamer_seed0
```