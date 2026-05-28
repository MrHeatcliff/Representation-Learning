import gymnasium as gym
import numpy as np
import torch
from argparse import Namespace

from xuance.common import BaseCallback, List, Optional, SequentialReplayBuffer, Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import REGISTRY_Policy, REGISTRY_Representation
from xuance.torch.agents.model_based_rl.dreamer_v3_agent import DreamerV3Agent
from xuance.torch.policies.hierarchical_dreamer import HierarchicalDreamerPolicy
from xuance.torch.representations.hierarchical_dreamer import HierarchicalDreamerWorldModel
from xuance.torch.representations.world_model import PlayerDV3
from xuance.torch.utils import ActivationFunctions


class HierarchicalDreamerAgent(DreamerV3Agent):
    """Isolated research fork of DreamerV3Agent.

    The train/test loops are inherited from DreamerV3Agent. Registration and
    component construction are separated so Hierarchical Dreamer can evolve
    without modifying DreamerV3's implementation.
    """

    representation_key = "HierarchicalDreamerWorldModel"
    policy_key = "HierarchicalDreamerPolicy"

    def __init__(
        self,
        config: Namespace,
        envs: Union[DummyVecEnv, SubprocVecEnv],
        callback: Optional[BaseCallback] = None,
    ):
        super(DreamerV3Agent, self).__init__(config, envs, callback)
        self.atari = True if self.config.env_name == "Atari" else False

        self.is_continuous = isinstance(self.train_envs.action_space, gym.spaces.Box)
        self.is_multidiscrete = isinstance(self.train_envs.action_space, gym.spaces.MultiDiscrete)
        self.config.is_continuous = self.is_continuous

        self.obs_shape = self.observation_space.shape
        if self.config.pixel:
            self.obs_shape = (self.obs_shape[2],) + self.obs_shape[:2]
        self.act_shape = self.action_space.n if not self.is_continuous else self.action_space.shape
        self.config.act_shape = self.act_shape

        self.replay_ratio = self.config.replay_ratio
        self.current_step, self.gradient_step = 0, 0

        ActivationFunctions["silu"] = torch.nn.SiLU
        REGISTRY_Representation[self.representation_key] = HierarchicalDreamerWorldModel
        self.model = self._build_representation(
            representation_key=self.representation_key,
            config=None,
            input_space=None,
        )

        REGISTRY_Policy[self.policy_key] = HierarchicalDreamerPolicy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, self.act_shape, self.callback)

        self.train_player: PlayerDV3 = self.model.player
        self.train_player.init_states()
        self.train_states: List[np.ndarray] = [
            self.train_envs.buf_obs,
            np.zeros((self.train_envs.num_envs,)),
            np.zeros((self.train_envs.num_envs,)),
            np.zeros((self.train_envs.num_envs,)),
            np.ones((self.train_envs.num_envs,)),
        ]

    def _build_representation(
        self,
        representation_key: str,
        input_space: Optional[gym.spaces.Space],
        config: Optional[Namespace],
    ) -> HierarchicalDreamerWorldModel:
        actions_dim = tuple(
            self.train_envs.action_space.shape
            if self.is_continuous
            else (
                self.train_envs.action_space.nvec.tolist()
                if self.is_multidiscrete
                else [self.train_envs.action_space.n]
            )
        )
        input_representations = dict(
            actions_dim=actions_dim,
            is_continuous=self.is_continuous,
            config=self.config,
            obs_space=self.train_envs.observation_space,
        )
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return REGISTRY_Representation[representation_key](**input_representations)

    def _build_memory(self, auxiliary_info_shape=None) -> SequentialReplayBuffer:
        input_buffer = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            auxiliary_shape=auxiliary_info_shape,
            n_envs=self.n_envs,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
        )
        return SequentialReplayBuffer(**input_buffer)

    def _build_policy(self) -> HierarchicalDreamerPolicy:
        return REGISTRY_Policy[self.policy_key](self.model, self.config)
