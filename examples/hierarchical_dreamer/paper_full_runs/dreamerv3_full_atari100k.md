# DreamerV3 Full Atari100K

Runs the DreamerV3 anchor on all 26 Atari100K games with seeds `0..4`.

This fills the DreamerV3 column for:

- `tab:atari-task-results`
- `tab:backbone-reproduction`
- Atari part of `tab:main-results`
- anchor rows for `tab:matched-controls`
- DreamerV3 rows in `tab:compute`

## Command

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance

export DEVICE=cuda:0
export WANDB_MODE=online
export PROJECT_NAME=HTS-WM-Paper-Final-DreamerV3

GAMES=(
  Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
  ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher
  Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye
  Qbert RoadRunner Seaquest UpNDown
)

for GAME in "${GAMES[@]}"; do
  ENV_ID="ALE/${GAME}-v5" \
  METHODS=dreamer \
  SEEDS=0,1,2,3,4 \
  DEVICE="$DEVICE" \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT_NAME="$PROJECT_NAME" \
  RUNNING_STEPS=100000 \
  REPLAY_RATIO=1.0 \
  BATCH_SIZE=16 \
  SEQ_LEN=64 \
  CHECKPOINT_RULE=final \
  EVAL_PROTOCOL=final \
  INTERMEDIATE_TEST_EPISODE=20 \
  TEST_EPISODE=100 \
  RENDER_EVAL_VIDEO=true \
  RENDER_INTERMEDIATE_VIDEO=false \
  examples/hierarchical_dreamer/run_paper_final_samecode_atari100k.sh
done
```

## Tmux

```bash
tmux new -s dreamerv3-atari26
```

Then paste the command block above.
