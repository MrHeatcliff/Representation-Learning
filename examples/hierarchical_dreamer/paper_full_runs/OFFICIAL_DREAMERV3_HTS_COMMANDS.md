# Official DreamerV3 HTS Commands

HTS is implemented inside the cloned official DreamerV3 repo:

```text
external_baselines/dreamerv3-official/dreamerv3/hts.py
external_baselines/dreamerv3-official/dreamerv3/hts_agent.py
external_baselines/dreamerv3-official/dreamerv3/main_hts.py
```

The original official baseline entrypoint remains:

```bash
python -m dreamerv3.main
```

The HTS entrypoint is:

```bash
python -m dreamerv3.main_hts
```

## Alien 10K Smoke

Use this before the full run. It uses the same official Atari100K protocol and `size12m` model as the official DreamerV3 baseline, with HTS auxiliary losses enabled.

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=HTS-DreamerV3-official-size12m-atari100k-alien-smoke10k-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-hts-atari100k-size12m-smoke \
WANDB_TAGS=official-dreamerv3,hts,atari100k,alien,size12m,smoke,seed0 \
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts \
  --configs hts_atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts/alien_smoke10k_size12m_seed0 \
  --run.steps 10000 \
  --logger.outputs jsonl,scope,wandb
```

Expected W&B/log keys:

```text
episode/score
train/loss/hts_hier
train/loss/hts_sdyn
train/loss/hts_temp
train/loss/hts_vc
train/loss/hts_sparse
train/hts/active_ratio
train/hts/mean_abs
train/hts/sdyn_l*
train/hts/hier_l*
```

## Alien Atari100K Full

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=HTS-DreamerV3-official-size12m-atari100k-alien-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-hts-atari100k-size12m \
WANDB_TAGS=official-dreamerv3,hts,atari100k,alien,size12m,seed0 \
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts \
  --configs hts_atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts/alien_atari100k_size12m_seed0 \
  --logger.outputs jsonl,scope,wandb
```

## Conservative-Loss Probe

If the default HTS auxiliary losses hurt the baseline curve early, run this lower-weight probe:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance/external_baselines/dreamerv3-official

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=HTS-WM-HarmonyDream-Alien-Curve \
WANDB_RUN_NAME=HTS-DreamerV3-official-size12m-atari100k-alien-lowaux-seed0 \
WANDB_MODE=online \
WANDB_GROUP=official-dreamerv3-hts-atari100k-size12m-lowaux \
WANDB_TAGS=official-dreamerv3,hts,atari100k,alien,size12m,lowaux,seed0 \
/mnt/disk1/backup_user/dat.tt2/xuance/.venv/bin/python -m dreamerv3.main_hts \
  --configs hts_atari100k size12m \
  --task atari100k_alien \
  --seed 0 \
  --logdir /mnt/disk1/backup_user/dat.tt2/xuance/logs/external_baselines/dreamerv3_official_hts/alien_atari100k_size12m_lowaux_seed0 \
  --agent.hts.l_hier 0.03 \
  --agent.hts.l_sdyn 0.03 \
  --agent.hts.l_temp 0.003 \
  --agent.hts.l_vc 0.003 \
  --agent.hts.l_sparse 0.000003 \
  --logger.outputs jsonl,scope,wandb
```

## Notes

Current HTS hook is auxiliary-only:

```text
RSSM repfeat -> HTS sparse hierarchy losses
World-model decoder/reward/continue/policy/value paths remain official DreamerV3.
```

This is intentional for the first parity stage: compare against the official baseline without changing the policy input path.
