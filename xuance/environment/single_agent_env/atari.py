import gymnasium as gym
import numpy as np
from collections import deque
from xuance.environment.single_agent_env.gym import LazyFrames
from PIL import Image
try:
    import cv2
except ImportError:
    pass

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass


class Atari_Env(gym.Wrapper):
    """Modified Atari environment with training optimizations.

    Implements several enhancements from DeepMind's Nature paper:
    - Episode termination: Treats end-of-life as end-of-episode (resets only on true game over)
    - Frame skipping: Returns only every `skip`-th frame
    - Observation resizing: Warps frames from 210x160 to configurable size
    - Frame stacking: Efficiently stacks last k frames using lazy arrays

    Note:
        All configurations should be set in the provided config object.
    """
    def __init__(self, config, **kwargs):
        """Initializes the Atari environment wrapper.

        Args:
            config: Configuration object containing:
                env_id: Atari environment ID (e.g., "ALE/Breakout-v5")
                env_seed: Random seed for environment
                obs_type: Observation type ("ram"/"rgb"/"grayscale")
                frame_skip: Frame skip interval (int or tuple for stochastic skipping)
                num_stack: Number of frames to stack
                img_size: Target observation dimensions (default: [210, 160])
                noop_max: Maximum no-op actions during reset
                render_mode: Rendering mode (None/"human"/"rgb_array")
                full_action_space: Whether to use full action space
            **kwargs: Additional arguments passed to gym.make()
        """
        full_action_space = getattr(config, 'full_action_space', False)
        self.repeat_action_probability = float(
            getattr(config, "repeat_action_probability", getattr(config, "sticky_action_probability", 0.0))
        )
        self.frame_skip = int(getattr(config, "frame_skip", 4))
        self.manual_frame_skip = bool(getattr(config, "manual_frame_skip", True))
        gym_frame_skip = 1 if self.manual_frame_skip else self.frame_skip
        self.env = gym.make(config.env_id,
                            render_mode=config.render_mode,
                            obs_type=config.obs_type,
                            frameskip=gym_frame_skip,
                            repeat_action_probability=self.repeat_action_probability,
                            full_action_space=full_action_space)
        self.env.metadata['render_fps'] = config.fps
        self.env.action_space.seed(seed=config.env_seed)
        self.env.reset(seed=config.env_seed)
        self.max_episode_steps = getattr(self.env, '_max_episode_steps', 1e5)
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.num_stack = config.num_stack
        self.obs_type = config.obs_type
        self.frames = deque([], maxlen=self.num_stack)
        self.raw_frames = deque([], maxlen=int(getattr(config, "frame_pooling", 2)))
        self.frame_pooling = int(getattr(config, "frame_pooling", 2))
        self.frame_aggregate = getattr(config, "frame_aggregate", "max")
        self.resize_backend = getattr(config, "resize_backend", "pillow")
        self.autostart = bool(getattr(config, "autostart", False))
        assert self.frame_aggregate in ("max", "mean")
        assert self.resize_backend in ("opencv", "pillow")
        self.image_size = getattr(config, 'img_size', [210, 160])
        assert config.img_size is not None
        self.noop_max = config.noop_max
        self.clip_reward = bool(getattr(config, "clip_reward", False))
        self.episodic_life = bool(getattr(config, "episodic_life", False))
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.life_loss_count = 0
        self.game_over_count = 0
        self.env_reset_count = 0
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(config.img_size[0], config.img_size[1], 3 * self.num_stack),
                                                    dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(config.img_size[0], config.img_size[1], self.num_stack),
                                                    dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self._render_mode = config.render_mode
        self._episode_step = 0
        self._episode_raw_step = 0
        self._episode_score = 0.0

    def close(self):
        """Closes the underlying environment and releases resources."""
        self.env.close()

    def render(self, *args, **kwargs):
        """Renders the environment.

        Returns:
            Rendered frame according to specified mode
        """
        return self.env.render()

    def reset(self, *args):
        """Resets the environment with random no-op actions.

        Performs:
        1. Environment reset
        2. Random number of no-op actions
        3. Initial fire action (if available)
        4. Frame stacking initialization

        Returns:
            tuple: (stacked_observations, info_dict)
        """
        info = {}
        self.raw_frames.clear()
        if self.was_real_done:
            obs, _ = self.env.reset()
            self.raw_frames.append(np.array(obs))
            self.env_reset_count += 1
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max + 1)
            for _ in range(num_noops):
                obs, _, done, _, _ = self.env.step(0)
                self.raw_frames.append(np.array(obs))
                if done:
                    obs, _ = self.env.reset()
                    self.raw_frames.clear()
                    self.raw_frames.append(np.array(obs))
            if self.autostart:
                action_meanings = self.env.unwrapped.get_action_meanings()
                if "FIRE" in action_meanings:
                    obs, _, done, _, _ = self.env.step(action_meanings.index("FIRE"))
                    self.raw_frames.append(np.array(obs))
                    if done:
                        obs, _ = self.env.reset()
                        self.raw_frames.clear()
                        self.raw_frames.append(np.array(obs))
                if "UP" in action_meanings:
                    obs, _, done, _, _ = self.env.step(action_meanings.index("UP"))
                    self.raw_frames.append(np.array(obs))
                    if done:
                        obs, _ = self.env.reset()
                        self.raw_frames.clear()
                        self.raw_frames.append(np.array(obs))
            # stack reset observations
            processed_obs = self.observation()
            for _ in range(self.num_stack):
                self.frames.append(processed_obs)

            self._episode_step = 0
            self._episode_raw_step = 0
            self._episode_score = 0.0
            info["episode_step"] = 0
        else:
            obs, _, done, _, _ = self.env.step(0)
            self.raw_frames.append(np.array(obs))
            processed_obs = self.observation()
            for _ in range(self.num_stack):
                self.frames.append(processed_obs)

        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = False
        return self._get_obs(), info

    def step(self, actions):
        """Executes one environment step.

        Args:
            actions: Action to execute

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        reward, terminated, truncated, info = 0.0, False, False, {}
        observation = None
        repeats = self.frame_skip if self.manual_frame_skip else 1
        for repeat in range(repeats):
            observation, step_reward, terminated, truncated, info = self.env.step(actions)
            reward += step_reward
            if repeat >= repeats - self.frame_pooling:
                self.raw_frames.append(np.array(observation))
            self._episode_raw_step += 1
            if terminated or truncated:
                break
        self.frames.append(self.observation())
        lives = self.env.unwrapped.ale.lives()
        real_done = bool(terminated)
        life_loss = bool((lives < self.lifes) and (lives > 0))
        # avoid environment bug
        if self._episode_raw_step >= self.max_episode_steps:
            terminated = True
            real_done = True
        self.was_real_done = bool(terminated)
        if life_loss:
            self.life_loss_count += 1
        if self.episodic_life and life_loss:
            terminated = True
        if real_done:
            self.game_over_count += 1
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
        self._episode_score += reward
        info["episode_score"] = self._episode_score
        info["episode_step"] = self._episode_step
        info["raw_reward"] = float(reward)
        info["life_loss"] = life_loss
        info["game_over"] = real_done
        info["env_reset_count"] = self.env_reset_count
        info["life_loss_count"] = self.life_loss_count
        info["game_over_count"] = self.game_over_count
        info["sticky_action_probability"] = self.repeat_action_probability
        info["clip_reward"] = self.clip_reward
        info["episodic_life"] = self.episodic_life
        return self._get_obs(), self.reward(reward), terminated, truncated, info

    def _get_obs(self):
        """Returns stacked observations as LazyFrames.

        Returns:
            LazyFrames: Stacked frame observations

        Raises:
            AssertionError: If frame stack is incomplete
        """
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))

    def observation(self):
        """Processes raw frame into desired observation format.

        Returns:
            Processed observation (resized grayscale/RGB or raw RAM)
        """
        frames = list(self.raw_frames)
        if not frames:
            raise RuntimeError("Atari raw frame buffer is empty.")
        if len(frames) == 1:
            frame = frames[0]
        elif self.frame_aggregate == "max":
            frame = np.max(np.stack(frames, axis=0), axis=0)
        else:
            frame = np.mean(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        if self.grayscale:
            if self.resize_backend == "pillow":
                frame = np.array(Image.fromarray(frame).resize(tuple(self.image_size), Image.BILINEAR))
            else:
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
            return np.expand_dims(frame, -1)
        elif self.rgb:
            if self.resize_backend == "pillow":
                return np.array(Image.fromarray(frame).resize(tuple(self.image_size), Image.BILINEAR))
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame

    def reward(self, reward):
        """Applies reward shaping for training.

        Args:
            reward: Original environment reward

        Returns:
            Shaped reward (sign function in this implementation)
        """
        if self.clip_reward:
            return np.sign(reward)
        return float(reward)
