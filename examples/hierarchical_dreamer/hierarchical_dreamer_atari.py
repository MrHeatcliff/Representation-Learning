import argparse
import json
import os
import time
import numpy as np
from copy import deepcopy
from pathlib import Path
from xuance.torch.utils.operations import set_seed
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import HierarchicalDreamerAgent
from examples.hierarchical_dreamer.paper_pipeline.artifacts import (
    artifact_dir,
    atari_task_name,
    write_eval_artifacts,
    write_run_files,
)


def configure_wandb_environment(mode: str | None):
    if mode is not None:
        os.environ["WANDB_MODE"] = mode
    wandb_home = Path.cwd() / ".wandb"
    for env_name, subdir in {
        "WANDB_CACHE_DIR": "cache",
        "WANDB_CONFIG_DIR": "config",
        "WANDB_DATA_DIR": "data",
    }.items():
        path = wandb_home / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(path))


def build_wandb_run_name(configs):
    env_name = configs.env_id.split("/")[-1]
    regime = getattr(getattr(configs, "training_regime", None), "name", "hierarchical")
    return f"{configs.agent}-{regime}-{env_name}-seed{configs.seed}-{configs.running_steps}steps"


def write_eval_artifact(configs, scores, checkpoint_rule, checkpoint_name, step):
    scores = np.asarray(scores, dtype=np.float64)
    result = {
        "run_name": getattr(configs, "wandb_run_name", None),
        "method": configs.agent,
        "ablation_name": getattr(getattr(configs, "hierarchical_latent", None), "ablation_name", None),
        "env_id": configs.env_id,
        "seed": int(configs.seed),
        "device": configs.device,
        "running_steps": int(configs.running_steps),
        "replay_ratio": float(configs.replay_ratio),
        "batch_size": int(configs.batch_size),
        "seq_len": int(configs.seq_len),
        "checkpoint_rule": checkpoint_rule,
        "checkpoint_name": checkpoint_name,
        "step": int(step),
        "eval_episodes": int(scores.size),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "episode_scores": scores.tolist(),
    }
    output_dir = Path(configs.log_dir) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_rule}_eval.json"
    with output_path.open("w") as file:
        json.dump(result, file, indent=2)
    print(f"Evaluation artifact: {output_path}")
    return result


def artifact_method(configs):
    latent = getattr(configs, "hierarchical_latent", None)
    ablation = getattr(latent, "ablation_name", None) if latent is not None else None
    if isinstance(latent, dict):
        ablation = latent.get("ablation_name", ablation)
    return ablation or configs.agent


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: Hierarchical Dreamer for Atari.")
    parser.add_argument("--config-file", type=str, default="config/atari.yaml")
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--harmony", type=bool, default=None)
    parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "wandb"])
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--wandb-user-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # atari100k, ratio=1, gradient_step=100k
    parser.add_argument("--running-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--replay-ratio", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--start-training", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--benchmark", type=int, default=None)
    parser.add_argument("--test-episode", type=int, default=None)
    parser.add_argument("--intermediate-test-episode", type=int, default=None)
    parser.add_argument("--checkpoint-rule", type=str, default=None, choices=["best", "final"])
    parser.add_argument("--artifact-root", type=str, default="artifacts")
    parser.add_argument("--experiment-id", type=str, default="paper_development")
    parser.add_argument("--suite", type=str, default="atari100k")
    parser.add_argument("--condition", type=str, default="default")
    return parser.parse_args()


if __name__ == '__main__':
    # print(sys.path)  # python path
    parser = parse_args()
    configure_wandb_environment(parser.wandb_mode)
    config_path = Path(parser.config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    configs_dict = load_yaml(file_dir=str(config_path))
    parser_overrides = {k: v for k, v in parser.__dict__.items() if k != "config_file" and v is not None}
    configs_dict = recursive_dict_update(configs_dict, parser_overrides)
    configs = argparse.Namespace(**configs_dict)
    configs.experiment_id = parser.experiment_id
    if getattr(configs, "wandb_run_name", None) is None:
        configs.wandb_run_name = build_wandb_run_name(configs)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = HierarchicalDreamerAgent(config=configs, envs=envs)
    run_start_time = time.time()
    method = artifact_method(configs)
    artifact_output_dir = artifact_dir(
        parser.artifact_root,
        parser.experiment_id,
        parser.suite,
        atari_task_name(configs.env_id),
        parser.condition,
        method,
        configs.seed,
    )
    write_run_files(artifact_output_dir, configs, method, parser.suite, parser.condition, run_start_time)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        configs_test = deepcopy(configs)
        configs_test.parallels = configs_test.test_episode
        test_envs = make_envs(configs_test)


        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        intermediate_test_episode = parser.intermediate_test_episode or test_episode
        num_epoch = int(train_steps / eval_interval)

        checkpoint_rule = getattr(configs, "checkpoint_rule", "best")
        if checkpoint_rule == "best":
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
            Agent.save_model(model_name="best_model.pth")
            write_eval_artifacts(
                artifact_output_dir, configs, method, parser.suite, parser.condition, test_scores,
                "best_model.pth", Agent.current_step, Agent.gradient_step, run_start_time, final=False,
            )
            best_scores_info = {"mean": np.mean(test_scores),
                                "std": np.std(test_scores),
                                "step": Agent.current_step,
                                "scores": test_scores}
            for i_epoch in range(num_epoch):
                print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                Agent.train(eval_interval)
                test_scores = Agent.test(test_episodes=intermediate_test_episode, test_envs=test_envs, close_envs=False)
                write_eval_artifacts(
                    artifact_output_dir, configs, method, parser.suite, parser.condition, test_scores,
                    "best_model.pth", Agent.current_step, Agent.gradient_step, run_start_time, final=False,
                )

                can_save = np.mean(test_scores) > best_scores_info["mean"]
                can_save |= (abs(np.mean(test_scores) - best_scores_info["mean"]) < 1e-6
                             and np.std(test_scores) < best_scores_info["std"])
                if can_save:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": Agent.current_step,
                                        "scores": test_scores}
                    # save best model
                    Agent.save_model(model_name="best_model.pth")
            print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
            write_eval_artifact(configs, best_scores_info["scores"], "best", "best_model.pth", best_scores_info["step"])
            write_eval_artifacts(
                artifact_output_dir, configs, method, parser.suite, parser.condition, best_scores_info["scores"],
                "best_model.pth", best_scores_info["step"], Agent.gradient_step, run_start_time, final=True,
            )
        else:
            for i_epoch in range(num_epoch):
                print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                Agent.train(eval_interval)
                test_scores = Agent.test(test_episodes=intermediate_test_episode, test_envs=test_envs, close_envs=False)
                write_eval_artifacts(
                    artifact_output_dir, configs, method, parser.suite, parser.condition, test_scores,
                    "periodic_checkpoint", Agent.current_step, Agent.gradient_step, run_start_time, final=False,
                )
            Agent.save_model(model_name="final_model.pth")
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
            final_scores_info = {"mean": np.mean(test_scores),
                                 "std": np.std(test_scores),
                                 "step": Agent.current_step,
                                 "scores": test_scores}
            print("Final Model Score: %.2f, std=%.2f" % (final_scores_info["mean"], final_scores_info["std"]))
            write_eval_artifact(configs, final_scores_info["scores"], "final", "final_model.pth", final_scores_info["step"])
            write_eval_artifacts(
                artifact_output_dir, configs, method, parser.suite, parser.condition, final_scores_info["scores"],
                "final_model.pth", final_scores_info["step"], Agent.gradient_step, run_start_time, final=True,
            )
        test_envs.close()
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            model = None
            # model = 'seed_1_2025_0324_100206'
            Agent.load_model(path=Agent.model_dir_load, model=model)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f'scores: {scores}')
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()
