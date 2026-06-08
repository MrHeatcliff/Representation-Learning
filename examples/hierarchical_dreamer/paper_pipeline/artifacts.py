import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ATARI_HNS_REFERENCES = {
    "Alien": (227.8, 7127.7),
    "Amidar": (5.8, 1719.5),
    "Assault": (222.4, 742.0),
    "Asterix": (210.0, 8503.3),
    "BankHeist": (14.2, 753.1),
    "BattleZone": (2360.0, 37187.5),
    "Boxing": (0.1, 12.1),
    "Breakout": (1.7, 30.5),
    "ChopperCommand": (811.0, 7387.8),
    "CrazyClimber": (10780.5, 35829.4),
    "DemonAttack": (152.1, 1971.0),
    "Freeway": (0.0, 29.6),
    "Frostbite": (65.2, 4334.7),
    "Gopher": (257.6, 2412.5),
    "Hero": (1027.0, 30826.4),
    "Jamesbond": (29.0, 302.8),
    "Kangaroo": (52.0, 3035.0),
    "Krull": (1598.0, 2665.5),
    "KungFuMaster": (258.5, 22736.3),
    "MsPacman": (307.3, 6951.6),
    "Pong": (-20.7, 14.6),
    "PrivateEye": (24.9, 69571.3),
    "Qbert": (163.9, 13455.0),
    "RoadRunner": (11.5, 7845.0),
    "Seaquest": (68.4, 42054.7),
    "UpNDown": (533.4, 11693.2),
}


EVAL_CURVE_FIELDS = [
    "experiment_id",
    "suite",
    "task",
    "condition",
    "method",
    "seed",
    "checkpoint_path",
    "global_step",
    "env_steps",
    "agent_actions",
    "frames",
    "action_repeat",
    "optimizer_updates",
    "eval_episodes",
    "metric_name",
    "metric_value",
    "raw_or_normalized",
    "config_hash",
    "code_commit",
    "wall_clock_seconds",
]


FINAL_EVAL_FIELDS = EVAL_CURVE_FIELDS + ["episode_index"]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(key): to_builtin(val) for key, val in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(config: Any) -> str:
    payload = json.dumps(to_builtin(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atari_task_name(env_id: str) -> str:
    name = env_id.split("/")[-1]
    return name[:-3] if name.endswith("-v5") else name


def method_slug(method: str) -> str:
    return (
        method.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def artifact_dir(
    artifact_root: str | Path,
    experiment_id: str,
    suite: str,
    task: str,
    condition: str,
    method: str,
    seed: int,
) -> Path:
    return (
        Path(artifact_root)
        / experiment_id
        / suite
        / task
        / condition
        / method_slug(method)
        / f"seed_{seed}"
    )


def ensure_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_header(path, fields)
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def hns(task: str, raw_score: float) -> float | None:
    if task not in ATARI_HNS_REFERENCES:
        return None
    random_score, human_score = ATARI_HNS_REFERENCES[task]
    denom = human_score - random_score
    if abs(denom) < 1e-12:
        return None
    return 100.0 * (raw_score - random_score) / denom


def base_metric_context(
    config: Any,
    method: str,
    suite: str,
    condition: str,
    checkpoint_path: str,
    global_step: int,
    optimizer_updates: int,
    eval_episodes: int,
    start_time: float,
) -> dict[str, Any]:
    task = atari_task_name(config.env_id) if suite == "atari100k" else getattr(config, "env_id", "unknown")
    action_repeat = int(getattr(config, "frame_skip", getattr(config, "action_repeat", 1)))
    agent_actions = int(global_step)
    return {
        "experiment_id": getattr(config, "experiment_id", "paper_development"),
        "suite": suite,
        "task": task,
        "condition": condition,
        "method": method,
        "seed": int(config.seed),
        "checkpoint_path": checkpoint_path,
        "global_step": int(global_step),
        "env_steps": agent_actions,
        "agent_actions": agent_actions,
        "frames": agent_actions * action_repeat,
        "action_repeat": action_repeat,
        "optimizer_updates": int(optimizer_updates),
        "eval_episodes": int(eval_episodes),
        "config_hash": config_hash(config),
        "code_commit": code_commit(),
        "wall_clock_seconds": round(time.time() - start_time, 3),
    }


def write_run_files(output_dir: Path, config: Any, method: str, suite: str, condition: str, start_time: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = to_builtin(config)
    with (output_dir / "resolved_config.yaml").open("w") as file:
        yaml.safe_dump(resolved, file, sort_keys=False)
    meta = {
        "experiment_id": getattr(config, "experiment_id", "paper_development"),
        "suite": suite,
        "task": atari_task_name(config.env_id) if suite == "atari100k" else getattr(config, "env_id", "unknown"),
        "condition": condition,
        "method": method,
        "seed": int(config.seed),
        "run_name": getattr(config, "wandb_run_name", None),
        "config_hash": config_hash(config),
        "code_commit": code_commit(),
        "checkpoint_rule": getattr(config, "checkpoint_rule", "unknown"),
        "action_repeat": int(getattr(config, "frame_skip", getattr(config, "action_repeat", 1))),
        "start_wall_time": start_time,
        "hns_reference_version": "common_atari100k_random_human_reference_v1" if suite == "atari100k" else None,
    }
    with (output_dir / "run_meta.json").open("w") as file:
        json.dump(meta, file, indent=2)


def write_eval_artifacts(
    output_dir: Path,
    config: Any,
    method: str,
    suite: str,
    condition: str,
    scores: list[float],
    checkpoint_path: str,
    global_step: int,
    optimizer_updates: int,
    start_time: float,
    final: bool = False,
) -> None:
    scores_np = np.asarray(scores, dtype=np.float64)
    context = base_metric_context(
        config=config,
        method=method,
        suite=suite,
        condition=condition,
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        optimizer_updates=optimizer_updates,
        eval_episodes=len(scores),
        start_time=start_time,
    )
    rows = [
        {
            **context,
            "metric_name": "raw_return_mean",
            "metric_value": float(scores_np.mean()) if scores_np.size else float("nan"),
            "raw_or_normalized": "raw",
        },
        {
            **context,
            "metric_name": "raw_return_std",
            "metric_value": float(scores_np.std()) if scores_np.size else float("nan"),
            "raw_or_normalized": "raw",
        },
    ]
    if suite == "atari100k":
        normalized = hns(context["task"], float(scores_np.mean()) if scores_np.size else float("nan"))
        if normalized is not None:
            rows.append({
                **context,
                "metric_name": "hns_mean",
                "metric_value": normalized,
                "raw_or_normalized": "normalized",
            })
    append_rows(output_dir / "eval_curve.csv", EVAL_CURVE_FIELDS, rows)

    if final:
        episode_rows = []
        for index, score in enumerate(scores_np.tolist()):
            episode_rows.append({
                **context,
                "episode_index": index,
                "metric_name": "raw_return",
                "metric_value": score,
                "raw_or_normalized": "raw",
            })
        append_rows(output_dir / "final_eval.csv", FINAL_EVAL_FIELDS, episode_rows)
