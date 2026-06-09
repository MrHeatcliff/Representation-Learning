import argparse
import json
import os
import sys
import time
import numpy as np
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xuance.torch.utils.operations import set_seed
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import DreamerV3Agent
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
    method = "HarmonyDream" if getattr(configs, "harmony", False) else "DreamerV3"
    return f"{method}-baseline-{env_name}-seed{configs.seed}-{configs.running_steps}steps"


def write_eval_artifact(configs, scores, checkpoint_rule, checkpoint_name, step):
    scores = np.asarray(scores, dtype=np.float64)
    result = {
        "run_name": getattr(configs, "wandb_run_name", None),
        "method": "HarmonyDream" if getattr(configs, "harmony", False) else "DreamerV3",
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
    return "HarmonyDream" if getattr(configs, "harmony", False) else "DreamerV3"


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}.")


def apply_model_size(configs_dict, model_size):
    if model_size is None:
        return configs_dict
    sizes = {
        "size12m": {"deter": 2048, "units": 256, "cnn_depth": 16, "layers": 2, "classes": 16},
        "small": {"deter": 512, "units": 512, "cnn_depth": 32, "layers": 2},
        "medium": {"deter": 1024, "units": 640, "cnn_depth": 48, "layers": 3},
        "large": {"deter": 2048, "units": 768, "cnn_depth": 64, "layers": 4},
        "xlarge": {"deter": 4096, "units": 1024, "cnn_depth": 96, "layers": 5},
    }
    scale = sizes[model_size]
    world_model = configs_dict["world_model"]
    if "classes" in scale:
        world_model["discrete_size"] = scale["classes"]
    world_model["recurrent_model"]["recurrent_state_size"] = scale["deter"]
    world_model["recurrent_model"]["dense_units"] = scale["units"]
    world_model["transition_model"]["hidden_size"] = scale["units"]
    world_model["representation_model"]["hidden_size"] = scale["units"]
    world_model["encoder"]["cnn_channels_multiplier"] = scale["cnn_depth"]
    world_model["encoder"]["dense_units"] = scale["units"]
    world_model["encoder"]["mlp_layers"] = scale["layers"]
    world_model["observation_model"]["cnn_channels_multiplier"] = scale["cnn_depth"]
    world_model["observation_model"]["dense_units"] = scale["units"]
    world_model["observation_model"]["mlp_layers"] = scale["layers"]
    world_model["reward_model"]["dense_units"] = scale["units"]
    world_model["reward_model"]["mlp_layers"] = scale["layers"]
    world_model["discount_model"]["dense_units"] = scale["units"]
    world_model["discount_model"]["mlp_layers"] = scale["layers"]
    configs_dict["actor"]["dense_units"] = scale["units"]
    configs_dict["actor"]["mlp_layers"] = scale["layers"]
    configs_dict["critic"]["dense_units"] = scale["units"]
    configs_dict["critic"]["mlp_layers"] = scale["layers"]
    configs_dict["model_size"] = model_size
    return configs_dict


def agent_test(agent, test_episodes, test_envs, close_envs=False, render_video=False):
    previous_render = agent.render
    agent.render = bool(render_video)
    try:
        return agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=close_envs)
    finally:
        agent.render = previous_render


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for Atari.")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--obs-type", type=str, default=None, choices=["ram", "rgb", "grayscale"])
    parser.add_argument("--img-size", type=int, nargs=2, default=None)
    parser.add_argument("--num-stack", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--repeat-action-probability", type=float, default=None)
    parser.add_argument("--clip-reward", type=str2bool, default=None)
    parser.add_argument("--episodic-life", type=str2bool, default=None)
    parser.add_argument("--model-size", type=str, default=None, choices=["size12m", "small", "medium", "large", "xlarge"])
    parser.add_argument("--log-dir", type=str, default="./logs/Breakout-v5/")
    parser.add_argument("--model-dir", type=str, default="./models/Breakout-v5/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--harmony", type=str2bool, default=None)
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--project-name", type=str, default="Representation-Learning")
    parser.add_argument("--wandb-user-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # XuanCe replay_ratio is minibatch updates per agent step after start_training.
    # 1.0 matches DreamerV3 paper-v2 Atari100K train_ratio=1024 replayed timesteps
    # per agent step when batch_size x seq_len = 16 x 64 = 1024.
    parser.add_argument("--running-steps", type=int, default=100_000)  # 100k
    parser.add_argument("--eval-interval", type=int, default=2_000)  # 50 logs
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--start-training", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--test-episode", type=int, default=None)
    parser.add_argument("--intermediate-test-episode", type=int, default=None)
    parser.add_argument("--checkpoint-rule", type=str, default="best", choices=["best", "final"])
    parser.add_argument(
        "--eval-protocol",
        type=str,
        default="periodic",
        choices=["periodic", "final", "train_only"],
        help="periodic runs separate eval during training; final trains uninterrupted then evals once; "
             "train_only logs training episode returns without separate eval.",
    )
    parser.add_argument("--render-eval-video", type=str2bool, default=False)
    parser.add_argument("--render-intermediate-video", type=str2bool, default=False)
    parser.add_argument("--artifact-root", type=str, default="artifacts")
    parser.add_argument("--experiment-id", type=str, default="paper_development")
    parser.add_argument("--suite", type=str, default="atari100k")
    parser.add_argument("--condition", type=str, default="default")
    return parser.parse_args()


if __name__ == '__main__':
    # print(sys.path)  # python path
    parser = parse_args()
    configure_wandb_environment(parser.wandb_mode)
    config_path = Path(__file__).resolve().parent / "config" / "atari.yaml"
    configs_dict = load_yaml(file_dir=str(config_path))
    parser_overrides = {k: v for k, v in parser.__dict__.items() if v is not None}
    configs_dict = recursive_dict_update(configs_dict, parser_overrides)
    configs_dict = apply_model_size(configs_dict, parser.model_size)
    configs = argparse.Namespace(**configs_dict)
    configs.experiment_id = parser.experiment_id
    if getattr(configs, "wandb_run_name", None) is None:
        configs.wandb_run_name = build_wandb_run_name(configs)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV3Agent(config=configs, envs=envs)
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

    eval_protocol = parser.eval_protocol
    if eval_protocol == "periodic" and getattr(configs, "checkpoint_rule", "best") == "final":
        eval_protocol = "final"

    if configs.benchmark:
        if eval_protocol in ("final", "train_only"):
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model(model_name="final_model.pth")
            if eval_protocol == "train_only":
                print("Finish training without separate evaluation.")
            else:
                configs_test = deepcopy(configs)
                configs_test.parallels = configs_test.test_episode
                configs_test.render = bool(parser.render_eval_video)
                test_envs = make_envs(configs_test)
                test_scores = agent_test(
                    Agent, configs.test_episode, test_envs, close_envs=False,
                    render_video=parser.render_eval_video,
                )
                final_scores_info = {"mean": np.mean(test_scores),
                                     "std": np.std(test_scores),
                                     "step": Agent.current_step,
                                     "scores": test_scores}
                print("Final Model Score: %.2f, std=%.2f" % (final_scores_info["mean"], final_scores_info["std"]))
                write_eval_artifact(configs, final_scores_info["scores"], "final", "final_model.pth",
                                    final_scores_info["step"])
                write_eval_artifacts(
                    artifact_output_dir, configs, method, parser.suite, parser.condition, final_scores_info["scores"],
                    "final_model.pth", final_scores_info["step"], Agent.gradient_step, run_start_time, final=True,
                )
                test_envs.close()
        else:
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            configs_test.render = bool(parser.render_eval_video or parser.render_intermediate_video)
            test_envs = make_envs(configs_test)

            train_steps = configs.running_steps // configs.parallels
            eval_interval = configs.eval_interval // configs.parallels
            test_episode = configs.test_episode
            intermediate_test_episode = parser.intermediate_test_episode or test_episode
            num_epoch = int(train_steps / eval_interval)

            checkpoint_rule = getattr(configs, "checkpoint_rule", "best")
            if checkpoint_rule == "best":
                test_scores = agent_test(
                    Agent, test_episode, test_envs, close_envs=False,
                    render_video=parser.render_intermediate_video,
                )
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
                    test_scores = agent_test(
                        Agent, intermediate_test_episode, test_envs, close_envs=False,
                        render_video=parser.render_intermediate_video,
                    )
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
                        Agent.save_model(model_name="best_model.pth")
                print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
                write_eval_artifact(configs, best_scores_info["scores"], "best", "best_model.pth",
                                    best_scores_info["step"])
                write_eval_artifacts(
                    artifact_output_dir, configs, method, parser.suite, parser.condition, best_scores_info["scores"],
                    "best_model.pth", best_scores_info["step"], Agent.gradient_step, run_start_time, final=True,
                )
            else:
                for i_epoch in range(num_epoch):
                    print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                    Agent.train(eval_interval)
                    test_scores = agent_test(
                        Agent, intermediate_test_episode, test_envs, close_envs=False,
                        render_video=parser.render_intermediate_video,
                    )
                    write_eval_artifacts(
                        artifact_output_dir, configs, method, parser.suite, parser.condition, test_scores,
                        "periodic_checkpoint", Agent.current_step, Agent.gradient_step, run_start_time, final=False,
                    )
                Agent.save_model(model_name="final_model.pth")
                test_scores = agent_test(
                    Agent, test_episode, test_envs, close_envs=False,
                    render_video=parser.render_eval_video,
                )
                final_scores_info = {"mean": np.mean(test_scores),
                                     "std": np.std(test_scores),
                                     "step": Agent.current_step,
                                     "scores": test_scores}
                print("Final Model Score: %.2f, std=%.2f" % (final_scores_info["mean"], final_scores_info["std"]))
                write_eval_artifact(configs, final_scores_info["scores"], "final", "final_model.pth",
                                    final_scores_info["step"])
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
