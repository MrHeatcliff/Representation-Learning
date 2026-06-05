import argparse
import json
from pathlib import Path

import yaml


def linear_params(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def get_nested(config: dict, dotted: str, default=None):
    target = config
    for part in dotted.split("."):
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def set_nested(config: dict, dotted: str, value):
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def hts_addon_params(config: dict, recurrent_state_size: int, actions_dim: int) -> int:
    latent = config["hierarchical_latent"]
    levels = int(latent["num_heads"])
    head_dim = int(latent["head_dim"])
    trunk_hidden = int(latent["trunk_hidden_size"])
    trunk_out = int(latent["trunk_output_size"])
    decoder_hidden = int(latent["reconstruction"]["decoder_hidden_size"])
    predictor_hidden = int(latent["sparse_dynamics"]["predictor_hidden_size"])
    strides = list(latent["sparse_dynamics"]["strides"])
    projection_hidden = int(latent["temporal_consistency"]["projection_hidden_size"])
    projection_dim = int(latent["temporal_consistency"]["projection_dim"])

    params = 0
    params += linear_params(recurrent_state_size, trunk_hidden)
    params += linear_params(trunk_hidden, trunk_out)
    params += levels * linear_params(trunk_out, head_dim)
    params += linear_params(levels * head_dim, recurrent_state_size)

    for level in range(levels):
        prefix_dim = (level + 1) * head_dim
        params += linear_params(prefix_dim, decoder_hidden)
        params += linear_params(decoder_hidden, recurrent_state_size)

    for level, stride in enumerate(strides):
        prefix_dim = (level + 1) * head_dim
        action_dim = int(stride) * actions_dim
        params += linear_params(prefix_dim + action_dim, predictor_hidden)
        params += linear_params(predictor_hidden, predictor_hidden)
        params += linear_params(predictor_hidden, recurrent_state_size)

    params += linear_params(head_dim, projection_hidden)
    params += linear_params(projection_hidden, projection_dim)
    return params


def flat_mh_addon_params(config: dict, recurrent_state_size: int, actions_dim: int, width: int) -> int:
    latent = config["hierarchical_latent"]
    code_dim = int(latent["flat_code_dim"])
    horizons = list(latent["flat_mh"]["horizons"])

    params = 0
    params += linear_params(recurrent_state_size, width)
    params += linear_params(width, width)
    params += linear_params(width, code_dim)
    for horizon in horizons:
        action_dim = int(horizon) * actions_dim
        params += linear_params(code_dim + action_dim, width)
        params += linear_params(width, width)
        params += linear_params(width, recurrent_state_size)
    return params


def main():
    parser = argparse.ArgumentParser("Search a larger-flat width that matches HTS-WM add-on parameters.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--recurrent-state-size", type=int, default=None)
    parser.add_argument("--actions-dim", type=int, default=18)
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    with Path(args.base).open("r") as file:
        config = yaml.safe_load(file)

    recurrent_state_size = args.recurrent_state_size
    if recurrent_state_size is None:
        recurrent_state_size = int(get_nested(config, "world_model.recurrent_model.recurrent_state_size", 512))

    target = hts_addon_params(config, recurrent_state_size, args.actions_dim)
    candidates = []
    for width in range(args.min_width, args.max_width + 1, args.step):
        params = flat_mh_addon_params(config, recurrent_state_size, args.actions_dim, width)
        gap = abs(params - target) / max(target, 1)
        candidates.append({
            "candidate_width": width,
            "addon_params": params,
            "target_hts_addon_params": target,
            "relative_param_gap": gap,
        })

    selected = min(candidates, key=lambda row: row["relative_param_gap"])
    output_config = Path(args.output_config)
    artifact = Path(args.artifact)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    matched = json.loads(json.dumps(config))
    width = int(selected["candidate_width"])
    set_nested(matched, "hierarchical_latent.ablation_name", "larger_flat_param")
    set_nested(matched, "hierarchical_latent.control_mode", "flat_mh")
    set_nested(matched, "hierarchical_latent.flat_trunk_hidden_size", width)
    set_nested(matched, "hierarchical_latent.flat_trunk_output_size", width)
    set_nested(matched, "hierarchical_latent.sparse_dynamics.predictor_hidden_size", width)
    set_nested(matched, "hierarchical_latent.loss_weights.hierarchical", 0.0)
    set_nested(matched, "hierarchical_latent.loss_weights.sparse_dynamics", 1.0)
    set_nested(matched, "hierarchical_latent.loss_weights.temporal", 0.0)
    set_nested(matched, "hierarchical_latent.loss_weights.variance_covariance", 0.0)
    set_nested(matched, "hierarchical_latent.loss_weights.sparsity", 0.0)
    set_nested(matched, "hierarchical_latent.temporal_consistency.mode", "none")
    set_nested(matched, "hierarchical_latent.variance_covariance.mode", "none")
    set_nested(matched, "hierarchical_latent.sparsity.mode", "none")

    with output_config.open("w") as file:
        yaml.safe_dump(matched, file, sort_keys=False)
    payload = {
        "selected_width": width,
        "target_tolerance": args.tolerance,
        "matched_within_tolerance": selected["relative_param_gap"] <= args.tolerance,
        "selected": selected,
        "top_10_candidates": sorted(candidates, key=lambda row: row["relative_param_gap"])[:10],
    }
    with artifact.open("w") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload["selected"], indent=2))
    if not payload["matched_within_tolerance"]:
        print(
            f"WARNING: best relative gap {selected['relative_param_gap']:.4f} exceeds "
            f"tolerance {args.tolerance:.4f}."
        )


if __name__ == "__main__":
    main()
