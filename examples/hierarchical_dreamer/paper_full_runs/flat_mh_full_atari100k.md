# Flat-MH Full Atari100K

Runs the paper-locked same-code `flat_mh` control on all 26 Atari100K games
with seeds `0..4`.

`flat_mh` uses:

```text
horizons = [1, 2, 4, 8, 16, 32]
no hierarchy
no nested prefix decoder
no TopK / L1
no temporal contrastive loss
no VC
```

This fills the Flat-MH rows/columns for:

- `tab:atari-task-results`
- `tab:matched-controls`
- `tab:ablation-plan`
- Atari part of `tab:main-results`
- Flat-MH rows in `tab:compute`

## Command

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE=cuda:0
export WANDB_MODE=online
export PROJECT_NAME=HTS-WM-Paper-Final-FlatMH

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  for SEED in 0 1 2 3 4; do
    ROM_NAME="${GAME}-v5"
    CONFIG_FILE=config/generated_configs/flat_mh.yaml \
    RUN_NAME=flat-mh-${GAME}-seed${SEED}-100k \
    ENV_ID="ALE/${GAME}-v5" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    WANDB_MODE="$WANDB_MODE" \
    PROJECT_NAME="$PROJECT_NAME" \
    RUNNING_STEPS=100000 \
    REPLAY_RATIO=0.125 \
    BATCH_SIZE=16 \
    SEQ_LEN=64 \
    CHECKPOINT_RULE=final \
    INTERMEDIATE_TEST_EPISODE=20 \
    TEST_EPISODE=100 \
    examples/hierarchical_dreamer/train_ablation.sh
  done
done
```

## Tmux

```bash
tmux new -s flatmh-atari26
```

Then paste the command block above.
