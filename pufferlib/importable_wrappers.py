from typing import Tuple, Dict, Any
import numpy as np
from gymnasium import spaces, ObservationWrapper
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from minigrid.wrappers import ImgObsWrapper, Wrapper
from stable_baselines3 import PPO


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Appends a 1-channel mask with a constant value indicating the *next* repeat factor.
    0 -> next action repeats once; 255 -> next action repeats three times.
    Assumes channel-first obs (C,H,W) from ImgObsWrapper; adjust if channel-last.
    """
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)
        # super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c + 1, h, w), dtype=np.uint8
        )

    def observation(self, obs):
        val = 0 if self.env.get_wrapper_attr("step_mode") == 1 else 1 # 0 for no repeat, 1 for repeat
        flag = np.full((1, obs.shape[1], obs.shape[2]), val, dtype=np.uint8)
        return np.concatenate([obs, flag], axis=0)


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that, with a given probability, performs a second
    'bonus' step using the same action.
    """

    def __init__(self, env: gym.Env) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        # super().__init__(env)
        self.step_mode = 0

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs1, reward1, terminated1, truncated1, info1 = self.env.step(action)

        if terminated1 or truncated1:
            self.step_mode = 0

        if self.step_mode == 0:
            info1['bonus_step_taken'] = False
            self.step_mode = 1
            return obs1, int(reward1), terminated1, truncated1, info1
        else:
            info1['bonus_step_taken'] = True
            self.step_mode = 0
            obs2, reward2, terminated2, truncated2, info2 = self.env.step(action)
            info1.update(info2)
            full_reward = reward1 + reward2
            if terminated2 or truncated2:
                return obs2, int(full_reward), terminated2, truncated2, info1

            # If we are still not done, we return the second observation
            obs3, reward3, terminated3, truncated3, info3 = self.env.step(action)
            info1.update(info3)
            full_reward += reward3
            return obs3, int(full_reward), terminated3, truncated3, info1

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.step_mode = 0
        obs, info = self.env.reset(*args, **kwargs)
        info['bonus_step_taken'] = False
        return obs, info

class RecordableImgObsWrapper(RecordConstructorArgs, ImgObsWrapper):
    """
    A version of ImgObsWrapper that records constructor args.
    """
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        ImgObsWrapper.__init__(self, env)
        # super().__init__(env)


class CallablePPO(PPO):
    """
    A version of PPO that can be called like a function to get actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, obs):
        action, _ = self.predict(obs, deterministic=True)
        return action


def make_lavastep_env(**kwargs):
    env = gym.make("MiniGrid-LavaGapS7-v0", **kwargs)
    env = AlternateStepWrapper(env)
    env = RecordableImgObsWrapper(env)         # (C,H,W) uint8 image
    env = RepeatFlagChannel(env)     # +1 channel flag
    return env

