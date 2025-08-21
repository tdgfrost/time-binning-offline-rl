# pip install minigrid gymnasium
from typing import Tuple, Optional, Dict, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == "__main__":
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env_name = 'MiniGrid-LavaGapS7-v0'

    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(2e5)

    # Managing your own trainer
    env = gym.make(env_name, render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in tqdm(range(1000)):
        action = model.predict(observation)[0]  # User-defined policy function
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()