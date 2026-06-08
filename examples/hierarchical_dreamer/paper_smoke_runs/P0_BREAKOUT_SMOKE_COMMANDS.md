# P0 Breakout Smoke Queue

Run this queue before any paper-final Atari/DMC/external campaign. It only runs
Breakout smoke jobs and is intended to catch code/config issues after the P0
corrections.

## Default Protocol

```text
env: ALE/Breakout-v5
seed: 1
steps: 10K agent actions
replay_ratio: 0.125
batch: 16 x 64
eval_interval: 2500
test_episode: 3
final eval video: enabled
eval protocol: final only, no periodic separate eval
wandb project: HTS-WM-P0-Smoke
```

## Run Everything In Order

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
bash examples/hierarchical_dreamer/paper_smoke_runs/run_p0_breakout_smoke_queue.sh
```

To use another GPU:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
DEVICE=cuda:1 bash examples/hierarchical_dreamer/paper_smoke_runs/run_p0_breakout_smoke_queue.sh
```

To make the smoke lighter:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
RUNNING_STEPS=1000 EVAL_INTERVAL=500 DEVICE=cuda:1 \
bash examples/hierarchical_dreamer/paper_smoke_runs/run_p0_breakout_smoke_queue.sh
```

To disable final eval GIF upload:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
RENDER_EVAL_VIDEO=false DEVICE=cuda:1 \
bash examples/hierarchical_dreamer/paper_smoke_runs/run_p0_breakout_smoke_queue.sh
```

To restore the old development behavior with separate periodic eval:

```bash
cd /mnt/disk1/backup_user/dat.tt2/xuance
EVAL_PROTOCOL=periodic DEVICE=cuda:1 \
bash examples/hierarchical_dreamer/paper_smoke_runs/run_p0_breakout_smoke_queue.sh
```

## Queue Order

1. `smoke-dreamerv3`
2. `smoke-hts-full`
3. `smoke-flat-sae`
4. `smoke-flat-mh`
5. `smoke-sgf-style-flat-same-code`
6. `smoke-recon-only-hierarchy`
7. `smoke-dense-multistride-no-sparse`
8. `smoke-hts-no-hier`
9. `smoke-hts-no-sdyn`
10. `smoke-hts-no-temp`
11. `smoke-hts-no-vc`
12. `smoke-larger-flat-param`
13. `smoke-xuance-harmonydream`
14. `smoke-tsae-style`

## What To Report Back

For each run, send:

```text
run name:
wandb URL:
finished or failed:
final/test episode reward mean:
Videos_Test present on W&B: yes/no
any traceback:
```

If one command fails, the script stops at that run. Send me the terminal output
from the failing section and do not continue to full runs yet.

## Not Included Yet

This queue intentionally does not run:

- full 26-game Atari100K campaigns
- DMC / DMC-GB2
- official external-code campaigns for SGF, DyMoDreamer, HarmonyDream, or EAWM
- MiniHack/KeyCorridor
- Synthetic fixed-buffer paper diagnostics

Those should start only after this smoke queue passes.
