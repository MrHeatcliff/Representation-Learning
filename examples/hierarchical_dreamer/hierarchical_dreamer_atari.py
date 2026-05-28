import argparse
import os
import numpy as np
from copy import deepcopy
from pathlib import Path
from xuance.torch.utils.operations import set_seed
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import HierarchicalDreamerAgent


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


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: Hierarchical Dreamer for Atari.")
    parser.add_argument("--config-file", type=str, default="config/atari.yaml")
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--harmony", type=bool, default=None)
    parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "wandb"])
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--wandb-user-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])

    # atari100k, ratio=1, gradient_step=100k
    parser.add_argument("--running-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--replay-ratio", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--start-training", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--benchmark", type=int, default=None)
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

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = HierarchicalDreamerAgent(config=configs, envs=envs)

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
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

            can_save = np.mean(test_scores) > best_scores_info["mean"]
            can_save |= (abs(np.mean(test_scores) - best_scores_info["mean"]) < 1e-6
                         and np.std(test_scores) < best_scores_info["std"])
            if can_save:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        test_envs.close()
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
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
