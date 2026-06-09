import torch
import csv
import os
from copy import deepcopy
from xuance.common import List, Union, SequentialReplayBuffer, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents import OffPolicyAgent
from xuance.torch import REGISTRY_Representation, REGISTRY_Policy
from xuance.torch.utils import ActivationFunctions

# '.': import from __init__
from xuance.torch.representations.world_model import DreamerV3WorldModel, PlayerDV3
from xuance.torch.policies import DreamerV3Policy

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from argparse import Namespace
from xuance.common import Optional


class DreamerV3Agent(OffPolicyAgent):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(DreamerV3Agent, self).__init__(config, envs, callback)
        # special judge for atari env
        self.atari = True if self.config.env_name == "Atari" else False

        # continuous or not
        self.is_continuous = (isinstance(self.train_envs.action_space, gym.spaces.Box))
        self.is_multidiscrete = isinstance(self.train_envs.action_space, gym.spaces.MultiDiscrete)
        self.config.is_continuous = self.is_continuous  # add to config

        # obs_shape & act_shape
        self.obs_shape = self.observation_space.shape
        """
        hwc 2 chw: 
        agent & memory both uses 'hwc'
        obs needed to be transformed to 'chw' and be normalized before sample & taking an action
        """
        if self.config.pixel:
            self.obs_shape = (self.obs_shape[2], ) + self.obs_shape[:2]
        self.act_shape = self.action_space.n if not self.is_continuous else self.action_space.shape
        self.config.act_shape = self.act_shape  # add to config

        # ratio
        self.replay_ratio = self.config.replay_ratio
        self.current_step, self.gradient_step = 0, 0
        self._debug_life_loss_count = 0
        self._debug_game_over_count = 0
        self._debug_truncation_count = 0

        # REGISTRY & create: representation, policy, learner
        ActivationFunctions['silu'] = torch.nn.SiLU
        REGISTRY_Representation['DreamerV3WorldModel'] = DreamerV3WorldModel
        self.model = self._build_representation(representation_key="DreamerV3WorldModel",
                                                config=None, input_space=None)

        REGISTRY_Policy["DreamerV3Policy"] = DreamerV3Policy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, self.act_shape, self.callback)
        self._log_model_parameter_counts()

        # train_player & train_states; make sure train & test to be independent
        self.train_player: PlayerDV3 = self.model.player
        self.train_player.init_states()
        self.train_states: List[np.ndarray] = [
            self.train_envs.buf_obs,  # obs: (envs, *obs_shape),
            np.zeros((self.train_envs.num_envs, )),  # rews
            np.zeros((self.train_envs.num_envs, )),  # terms
            np.zeros((self.train_envs.num_envs, )),  # truncs
            np.ones((self.train_envs.num_envs, ))  # is_first
        ]

    @staticmethod
    def _parameter_count(module, seen=None):
        if module is None:
            return 0
        seen = set() if seen is None else seen
        count = 0
        for parameter in module.parameters():
            ident = id(parameter)
            if ident in seen:
                continue
            seen.add(ident)
            count += parameter.numel()
        return count

    def _log_model_parameter_counts(self):
        core_seen = set()
        world_model_params = self._parameter_count(getattr(self.policy, "world_model", None), core_seen)
        actor_params = self._parameter_count(getattr(self.policy, "actor", None), core_seen)
        critic_params = self._parameter_count(getattr(self.policy, "critic", None), core_seen)
        target_critic_params = self._parameter_count(getattr(self.policy, "target_critic", None), core_seen)
        core_total = world_model_params + actor_params + critic_params + target_critic_params
        policy_total = self._parameter_count(self.policy)
        non_core_registered = max(policy_total - core_total, 0)
        info = {
            "params/world_model": world_model_params,
            "params/actor": actor_params,
            "params/critic": critic_params,
            "params/target_critic": target_critic_params,
            "params/core_agent_total": core_total,
            "params/registered_policy_total": policy_total,
            "params/non_core_registered_total": non_core_registered,
        }
        self.log_infos(info, 0)
        print(
            "Parameter count: "
            f"world_model={world_model_params:,}, actor={actor_params:,}, "
            f"critic={critic_params:,}, target_critic={target_critic_params:,}, "
            f"core_agent_total={core_total:,}, registered_policy_total={policy_total:,}, "
            f"non_core_registered_total={non_core_registered:,}"
        )

    def _env_frame_step(self, agent_step: Optional[int] = None) -> int:
        agent_step = self.current_step if agent_step is None else agent_step
        action_repeat = int(getattr(self.config, "frame_skip", getattr(self.config, "action_repeat", 1)))
        return int(agent_step * action_repeat)

    def _official_train_aliases(self, info: dict) -> dict:
        aliases = {}
        mapping = {
            "model_loss/model_loss": "train/loss/model",
            "model_loss/obs_loss": "train/loss/image",
            "model_loss/rew_loss": "train/loss/rew",
            "model_loss/continue_loss": "train/loss/con",
            "model_loss/kl_loss": "train/loss/kl",
            "actor_loss/actor_loss": "train/loss/policy",
            "critic_loss/critic_loss": "train/loss/value",
            "actor_loss/entropy_loss": "train/ent/action",
        }
        for src, dst in mapping.items():
            if src in info:
                aliases[dst] = info[src]
        if "debug/effective_replay_ratio" in info:
            update_ratio = float(info["debug/effective_replay_ratio"])
            aliases["replay/update_ratio"] = update_ratio
            aliases["replay/replay_ratio"] = update_ratio * int(self.batch_size) * int(self.config.seq_len)
        if "step/gradient_step" in info:
            aliases["train/updates"] = info["step/gradient_step"]
        return aliases

    def _log_train_episode_csv(
        self,
        env_id: int,
        episode_return: float,
        episode_length: int,
        agent_step: int,
    ) -> None:
        action_repeat = int(getattr(self.config, "frame_skip", getattr(self.config, "action_repeat", 1)))
        env_frame = int(agent_step * action_repeat)
        path = os.path.join(os.getcwd(), self.log_dir, "train_episode_returns.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "agent_step",
                    "env_frame",
                    "action_repeat",
                    "env_id",
                    "episode_index",
                    "episode_return",
                    "episode_length",
                    "episode_reward_mean",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "agent_step": agent_step,
                "env_frame": env_frame,
                "action_repeat": action_repeat,
                "env_id": env_id,
                "episode_index": int(self.current_episode[env_id]),
                "episode_return": episode_return,
                "episode_length": episode_length,
                "episode_reward_mean": episode_return / max(episode_length, 1),
            })

    def _build_representation(self, representation_key: str,
                              input_space: Optional[gym.spaces.Space],
                              config: Optional[Namespace]) -> DreamerV3WorldModel:
        # specify the type in order to use code completion
        actions_dim = tuple(
            self.train_envs.action_space.shape
            if self.is_continuous else (
                self.train_envs.action_space.nvec.tolist() if self.is_multidiscrete else [self.train_envs.action_space.n]
            )
        )
        input_representations = dict(
            actions_dim=actions_dim,
            is_continuous=self.is_continuous,
            config=self.config,
            obs_space=self.train_envs.observation_space
        )
        representation = REGISTRY_Representation[representation_key](**input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_memory(self, auxiliary_info_shape=None) -> SequentialReplayBuffer:
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return SequentialReplayBuffer(**input_buffer)

    def _build_policy(self) -> DreamerV3Policy:
        return REGISTRY_Policy["DreamerV3Policy"](self.model, self.config)

    def action(self,
               obs: np.ndarray,
               test_mode: Optional[bool] = False,
               player: Optional[PlayerDV3] = None) -> np.ndarray:
        """Returns actions and values.

        Parameters:
            obs (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.
            player (Optional[PlayerDV3]): The player whose action is taken, default is train_player.

        Returns:
            actions: The real_actions to be executed.
        """
        if self.config.pixel:
            obs = obs.transpose(0, 3, 1, 2) / 255.0 - 0.5
        player = player if player is not None else self.train_player
        # actions_output = self.policy(observations)
        # [envs, *obs_shape] -> [1: batch, envs, *obs_shape]
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actions = player.get_actions(obs, greedy=test_mode, mask=None)[0][0]
        # ont-hot -> real_actions
        if not self.is_continuous:
            actions = actions.argmax(dim=1).detach().cpu().numpy()
        else:  # [1, envs, *act_shape]
            actions = actions.reshape(obs.shape[1], *self.act_shape).detach().cpu().numpy()
            # action mapping in xuance.environment.utils.wrapper.XuanCeEnvWrapper.step
            # actions = (actions + 1.0) * 0.5 * (self.actions_high - self.actions_low) + self.actions_low  # action_scaling
        """
        for env_interaction: actions.shape, (envs, ) or (env, *act_shape)
        """
        return actions

    def train_epochs(self, n_epochs: int = 1):
        train_info = {}
        samples = self.memory.sample(self.config.seq_len)  # (envs, seq, batch, ~)
        if self.config.pixel:
            samples['obs'] = samples['obs'].transpose(0, 1, 2, 5, 3, 4) / 255.0 - 0.5
        # n_epoch(n_gradient step) scattered to each environment
        # st = np.random.choice(np.arange(self.train_envs.num_envs), 1).item()  # not necessary
        st = 0
        for _ in range(n_epochs):  # assert n_epochs == parallels
            cur_samples = {k: v[(st + _) % self.train_envs.num_envs] for k, v in samples.items()}
            train_info = self.learner.update(**cur_samples)
        return train_info

    def train(self, train_steps):  # each train still uses old rssm_states until episode end
        train_info = {}
        obs, rews, terms, truncs, is_first = self.train_states

        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            if self.current_step < self.start_training:  # ramdom_sample before training
                acts = np.array([self.train_envs.action_space.sample() for _ in range(self.train_envs.num_envs)])
            else:
                acts = self.action(obs)
            if self.atari:  # use truncs to train in xc_atari
                terms = deepcopy(truncs)
            """(o1, a1, r1, term1, trunc1, is_first1), acts: real_acts"""
            self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)
            next_obs, rews, terms, truncs, infos = self.train_envs.step(acts)
            self._debug_life_loss_count += sum(int(info.get("life_loss", False)) for info in infos)
            self._debug_game_over_count += sum(int(info.get("game_over", False)) for info in infos)
            self._debug_truncation_count += int(np.sum(truncs))
            protocol_info = {
                "debug/agent_step": int(self.current_step),
                "debug/env_frame": int(
                    self.current_step * int(getattr(self.config, "frame_skip", getattr(self.config, "action_repeat", 1)))
                ),
                "debug/gradient_step": int(self.gradient_step),
                "debug/effective_replay_ratio": float(
                    self.gradient_step / max(self.current_step - self.start_training, 1)
                    if self.current_step > self.start_training else 0.0
                ),
                "debug/sticky_action_probability": float(
                    infos[0].get(
                        "sticky_action_probability",
                        getattr(self.config, "repeat_action_probability", getattr(self.config, "sticky_action_probability", 0.0)),
                    )
                ) if len(infos) > 0 else float(getattr(self.config, "repeat_action_probability", 0.0)),
                "debug/clip_reward": float(bool(getattr(self.config, "clip_reward", False))),
                "debug/episodic_life": float(bool(getattr(self.config, "episodic_life", False))),
                "debug/life_loss_count": int(self._debug_life_loss_count),
                "debug/game_over_count": int(self._debug_game_over_count),
                "debug/env_reset_count": int(sum(int(info.get("env_reset_count", 0)) for info in infos)),
                "debug/truncation_count": int(self._debug_truncation_count),
                "train/raw_reward_mean": float(np.mean([info.get("raw_reward", rew) for info, rew in zip(infos, rews)])),
                "train/training_reward_mean": float(np.mean(rews)),
            }

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.policy,
                                        obs=obs, acts=acts, next_obs=next_obs, rewards=rews,
                                        terminals=terms, truncations=truncs, infos=infos,
                                        train_steps=train_steps)

            """
            set to zeros after the first step
            (o2, a1, r2, term2, trunc2, is_first2)
            """
            is_first = np.zeros_like(terms)
            obs = next_obs
            self.returns = self.gamma * self.returns + rews
            done_idxes = []
            for i in range(self.n_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):  # do not term until trunc
                        pass
                    else:
                        # carry the reset procedure to the outside
                        done_idxes.append(i)
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        episode_return = float(infos[i]["episode_score"])
                        episode_length = int(infos[i]["episode_step"])
                        agent_step = int(self.current_step)
                        action_repeat = int(getattr(self.config, "frame_skip", getattr(self.config, "action_repeat", 1)))
                        env_frame = agent_step * action_repeat
                        self._log_train_episode_csv(i, episode_return, episode_length, agent_step)
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": episode_length,
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": episode_return,
                                "train/episode_return": episode_return,
                                "train/episode_reward_mean": episode_return / max(episode_length, 1),
                                "train/episode_length": episode_length,
                                "train/agent_step": agent_step,
                                "train/env_frame": env_frame,
                                "train/action_repeat": action_repeat,
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        episode_info.update({
                            "episode/score": episode_return,
                            "episode/length": episode_length,
                        })
                        self.log_infos(episode_info, self._env_frame_step(self.current_step))
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.train_envs, policy=self.policy, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)
            self.current_step += self.n_envs
            if len(done_idxes) > 0:
                """
                store the last data and reset all
                (o_t, a_t = 0 for dones, r_t, term_t, trunc_t, is_first_t)
                """
                extra_shape = () if not self.is_continuous else self.act_shape
                acts[done_idxes] = np.zeros((len(done_idxes),) + extra_shape)
                if self.atari:  # use truncs to train in xc_atari
                    terms = deepcopy(truncs)
                self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)

                """reset DreamerV3 Player's states"""
                obs[done_idxes] = np.stack([infos[idx]["reset_obs"] for idx in done_idxes])  # reset obs
                self.train_envs.buf_obs[done_idxes] = obs[done_idxes]
                rews[done_idxes] = np.zeros((len(done_idxes), ))
                terms[done_idxes] = np.zeros((len(done_idxes), ))
                truncs[done_idxes] = np.zeros((len(done_idxes), ))
                is_first[done_idxes] = np.ones_like(terms[done_idxes])
                self.train_player.init_states(done_idxes)
            """
            start training 
            replay_ratio = self.gradient_step / self.current_step
            """
            if self.current_step > self.start_training:
                # count current_step when start_training
                n_epochs = max(int((self.current_step - self.start_training) * self.replay_ratio - self.gradient_step), 0)
                update_info = self.train_epochs(n_epochs=n_epochs)
                self.gradient_step += n_epochs
                if train_info is not None:
                    update_info.update(protocol_info)
                    update_info.update(self._official_train_aliases(update_info))
                    self.log_infos(update_info, self._env_frame_step(self.current_step))
                    train_info.update(update_info)
                    self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                      current_episode=self.current_episode, train_steps=train_steps,
                                                      update_info=update_info)

            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        # save the train_states for next train
        self.train_states = [obs, rews, terms, truncs, is_first]
        return train_info

    def test(self,
             test_episodes: int,
             test_envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
             close_envs: bool = True
             ) -> list:
        if test_envs is None:
            raise ValueError("`test_envs` must be provided for evaluation.")
        num_envs = test_envs.num_envs
        # copy the total network for test
        test_player = deepcopy(self.train_player)
        test_player.init_states(num_envs=num_envs)
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, lengths, best_score = 0, [], [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        is_done = np.zeros(num_envs)
        while is_done.sum() < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, test_mode=True, player=test_player)
            next_obs, rews, terms, truncs, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            done_idxes = []
            for i in range(num_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):
                        pass
                    else:
                        done_idxes.append(i)
                        obs[i] = infos[i]["reset_obs"]
                        if is_done[i] != 1:
                            is_done[i] = 1
                            scores.append(infos[i]["episode_score"])
                            lengths.append(infos[i]["episode_step"])
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()

            if len(done_idxes) > 0:
                test_player.init_states(reset_envs=done_idxes, num_envs=num_envs)

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)  # fps cannot work

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores),
            "eval/episode_return_mean": np.mean(scores),
            "eval/episode_return_std": np.std(scores),
            "eval/episode_return_min": np.min(scores),
            "eval/episode_return_max": np.max(scores),
            "eval/episode_reward_mean": np.mean(scores),
            "eval/score": np.mean(scores),
            "eval/length": np.mean(lengths) if lengths else 0.0,
            "eval/score_std": np.std(scores),
            "eval/length_std": np.std(lengths) if lengths else 0.0,
        }
        self.log_infos(test_info, self._env_frame_step(self.current_step))

        if close_envs:
            test_envs.close()


        return scores
