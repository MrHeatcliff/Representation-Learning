import argparse
import json
from pathlib import Path

import numpy as np


FACTOR_NAMES = ["fast", "mid", "slow", "context", "nuisance"]
BOUNDARY_NAMES = ["fast", "mid", "slow", "context", "macro"]
ACTION_VALUES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
FACTOR_CARDINALITIES = {
    "fast": 8,
    "mid": 8,
    "slow": 8,
    "context": 4,
    "nuisance": 16,
}


def _one_hot(values: np.ndarray, cardinality: int) -> np.ndarray:
    return np.eye(cardinality, dtype=np.float32)[values]


def _sample_actions(rng: np.random.Generator, count: int, length: int) -> np.ndarray:
    actions = np.empty((count, length), dtype=np.int64)
    block_structured = rng.random(count) >= 0.7
    random_count = int((~block_structured).sum())
    if random_count:
        actions[~block_structured] = rng.integers(0, len(ACTION_VALUES), size=(random_count, length))

    for row in np.flatnonzero(block_structured):
        cursor = 0
        while cursor < length:
            block = int(rng.choice([4, 8, 16]))
            value = int(rng.integers(0, len(ACTION_VALUES)))
            end = min(cursor + block, length)
            actions[row, cursor:end] = value
            cursor = end
    return actions


def _simulate_factors(rng: np.random.Generator, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, length = actions.shape
    factors = np.zeros((count, length, len(FACTOR_NAMES)), dtype=np.int16)
    boundaries = np.zeros((count, length, len(BOUNDARY_NAMES)), dtype=bool)

    factors[:, 0, 0] = rng.integers(0, FACTOR_CARDINALITIES["fast"], size=count)
    factors[:, 0, 1] = rng.integers(0, FACTOR_CARDINALITIES["mid"], size=count)
    factors[:, 0, 2] = rng.integers(0, FACTOR_CARDINALITIES["slow"], size=count)
    factors[:, 0, 3] = rng.integers(0, FACTOR_CARDINALITIES["context"], size=count)
    factors[:, 0, 4] = rng.integers(0, FACTOR_CARDINALITIES["nuisance"], size=count)

    signed = ACTION_VALUES[actions]
    for t in range(length - 1):
        next_factors = factors[:, t].copy()
        next_factors[:, 0] = (next_factors[:, 0] + signed[:, t]) % FACTOR_CARDINALITIES["fast"]

        if (t + 1) % 4 == 0:
            block_sum = signed[:, max(0, t - 3):t + 1].sum(axis=1)
            next_factors[:, 1] = (next_factors[:, 1] + block_sum) % FACTOR_CARDINALITIES["mid"]

        if (t + 1) % 16 == 0:
            block_sum = signed[:, max(0, t - 15):t + 1].sum(axis=1)
            next_factors[:, 2] = (next_factors[:, 2] + np.sign(block_sum).astype(np.int16)) % FACTOR_CARDINALITIES["slow"]

        if (t + 1) % 64 == 0:
            next_factors[:, 3] = (next_factors[:, 3] + 1) % FACTOR_CARDINALITIES["context"]

        nuisance_step = rng.choice(np.array([-1, 1], dtype=np.int16), size=count)
        next_factors[:, 4] = (next_factors[:, 4] + nuisance_step) % FACTOR_CARDINALITIES["nuisance"]
        factors[:, t + 1] = next_factors

        changed = factors[:, t + 1] != factors[:, t]
        boundaries[:, t + 1, 0] = changed[:, 0]
        boundaries[:, t + 1, 1] = (t + 1) % 4 == 0
        boundaries[:, t + 1, 2] = (t + 1) % 16 == 0
        boundaries[:, t + 1, 3] = (t + 1) % 64 == 0
        boundaries[:, t + 1, 4] = boundaries[:, t + 1, 2] | boundaries[:, t + 1, 3]
    return factors, boundaries


def _make_state_observations(
    rng: np.random.Generator,
    factors: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    pieces = [
        _one_hot(factors[..., 0], FACTOR_CARDINALITIES["fast"]),
        _one_hot(factors[..., 1], FACTOR_CARDINALITIES["mid"]),
        _one_hot(factors[..., 2], FACTOR_CARDINALITIES["slow"]),
        _one_hot(factors[..., 3], FACTOR_CARDINALITIES["context"]),
        _one_hot(factors[..., 4], FACTOR_CARDINALITIES["nuisance"]),
    ]
    obs = np.concatenate(pieces, axis=-1).astype(np.float32)
    if noise_std > 0:
        obs += rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)
    return obs


def _make_pixel_observations(factors: np.ndarray) -> np.ndarray:
    count, length, _ = factors.shape
    obs = np.zeros((count, length, 64, 64, 3), dtype=np.uint8)
    context_colors = np.array(
        [[24, 24, 36], [28, 48, 32], [52, 34, 28], [34, 42, 58]], dtype=np.uint8
    )
    nuisance_colors = np.array(
        [[230, 80, 70], [70, 200, 90], [80, 130, 230], [220, 210, 80]], dtype=np.uint8
    )
    for i in range(count):
        for t in range(length):
            fast, mid, slow, context, nuisance = factors[i, t]
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            canvas[:] = context_colors[context]
            canvas[:3, :, :] = np.clip(context_colors[context] + 45, 0, 255)
            canvas[-3:, :, :] = np.clip(context_colors[context] + 45, 0, 255)
            canvas[:, :3, :] = np.clip(context_colors[context] + 45, 0, 255)
            canvas[:, -3:, :] = np.clip(context_colors[context] + 45, 0, 255)

            room_offset = int(slow) * 3
            canvas[10 + room_offset % 12:13 + room_offset % 12, 8:56, :] = [95, 95, 110]
            canvas[42 - room_offset % 12:45 - room_offset % 12, 8:56, :] = [95, 95, 110]

            obj_x = 8 + int(mid) * 6
            canvas[24:34, obj_x:obj_x + 6, :] = [180, 180, 190]

            agent_x = 6 + int(fast) * 7
            agent_y = 48 - (int(mid) % 3) * 7
            canvas[agent_y:agent_y + 5, agent_x:agent_x + 5, :] = [245, 245, 245]

            patch_x = 4 + (int(nuisance) % 8) * 7
            patch_y = 4 + (int(nuisance) // 8) * 8
            canvas[patch_y:patch_y + 7, patch_x:patch_x + 7, :] = nuisance_colors[int(nuisance) % 4]
            obs[i, t] = canvas
    return obs


def _ids(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fast, mid, slow, context = [factors[..., i].astype(np.int64) for i in range(4)]
    macro = slow * FACTOR_CARDINALITIES["context"] + context
    full = (((fast * FACTOR_CARDINALITIES["mid"] + mid) * FACTOR_CARDINALITIES["slow"] + slow)
            * FACTOR_CARDINALITIES["context"] + context)
    return macro.astype(np.int64), full.astype(np.int64)


def _write_split(
    output: Path,
    split: str,
    count: int,
    length: int,
    shard_size: int,
    noise_std: float,
    pixel: bool,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_id, start in enumerate(range(0, count, shard_size)):
        shard_count = min(shard_size, count - start)
        actions = _sample_actions(rng, shard_count, length)
        factors, boundaries = _simulate_factors(rng, actions)
        obs = _make_pixel_observations(factors) if pixel else _make_state_observations(rng, factors, noise_std)
        macro_state_id, full_state_id = _ids(factors)
        dones = np.zeros((shard_count, length), dtype=bool)
        dones[:, -1] = True
        episode_id = np.arange(start, start + shard_count, dtype=np.int64)[:, None].repeat(length, axis=1)
        timestep = np.arange(length, dtype=np.int64)[None, :].repeat(shard_count, axis=0)
        path = split_dir / f"{split}_{shard_id:03d}.npz"
        np.savez_compressed(
            path,
            obs=obs,
            actions=actions,
            rewards=np.zeros((shard_count, length), dtype=np.float32),
            dones=dones,
            factors=factors,
            boundaries=boundaries,
            boundary_fast=boundaries[..., 0],
            boundary_mid=boundaries[..., 1],
            boundary_slow=boundaries[..., 2],
            boundary_context=boundaries[..., 3],
            boundary_macro=boundaries[..., 4],
            macro_state_id=macro_state_id,
            full_state_id=full_state_id,
            revisit_group_id=macro_state_id,
            timestep=timestep,
            episode_id=episode_id,
        )
        shards.append({"file": str(path.relative_to(output)), "episodes": shard_count, "length": length})
    return shards


def main():
    parser = argparse.ArgumentParser("Generate the HTS-WM synthetic multi-timescale fixed-buffer dataset.")
    parser.add_argument("--output", default="data/synthetic_multiscale_state", help="Output dataset directory.")
    parser.add_argument("--train-trajectories", type=int, default=10000)
    parser.add_argument("--val-trajectories", type=int, default=2000)
    parser.add_argument("--test-trajectories", type=int, default=2000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel", action="store_true", help="Render 64x64x3 pixel observations instead of 44D states.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": (args.train_trajectories, args.seed + 101),
        "val": (args.val_trajectories, args.seed + 202),
        "test": (args.test_trajectories, args.seed + 303),
    }
    manifest = {
        "name": "synthetic_multiscale",
        "version": 1,
        "observation_type": "pixel_64x64_rgb" if args.pixel else "state_vector_44",
        "noise_std": args.noise_std,
        "episode_length": args.length,
        "action_values": ACTION_VALUES.tolist(),
        "factor_names": FACTOR_NAMES,
        "factor_cardinalities": FACTOR_CARDINALITIES,
        "boundary_names": BOUNDARY_NAMES,
        "macro_state_id": "slow * 4 + context",
        "full_state_id": "(((fast * 8 + mid) * 8 + slow) * 4 + context)",
        "reward_mode": "zero_fixed_buffer_representation",
        "behavior_policy": "70% iid random actions, 30% block-structured actions with block length in {4,8,16}",
        "splits": {},
    }
    for split, (count, split_seed) in splits.items():
        manifest["splits"][split] = _write_split(
            output,
            split,
            count,
            args.length,
            args.shard_size,
            args.noise_std,
            args.pixel,
            split_seed,
        )

    with (output / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Wrote synthetic dataset manifest to {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
