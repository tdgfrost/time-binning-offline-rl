from typing import Tuple, Dict, Any, Union
import numpy as np
from gymnasium import spaces, ObservationWrapper
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.utils import RecordConstructorArgs
from minigrid.wrappers import ImgObsWrapper, Wrapper, FullyObsWrapper
from stable_baselines3 import PPO
import torch.nn as nn
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Sequence
from pathlib import Path
from d3rlpy.models.encoders import register_encoder_factory
import dataclasses
from stable_baselines3.common.callbacks import BaseCallback
import d3rlpy
from torch.distributions import Categorical
from d3rlpy.dataset.trajectory_slicers import TrajectorySlicerProtocol
from d3rlpy.dataset.components import EpisodeBase, PartialTrajectory
from d3rlpy.types import Float32NDArray, Int32NDArray, ObservationSequence, NDArray
from d3rlpy.dataset.utils import (batch_pad_array, batch_pad_observations, slice_observations, check_dtype,
                                  cast_recursively, stack_observations)
from d3rlpy.dataset.mini_batch import TrajectoryMiniBatch
from tqdm import tqdm
from collections import deque


class ReplayBufferEnv:
    def __init__(self, env, buffer_size: int = 100000):
        self.observations = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.env = env

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed)
        self.observations.append(obs)
        return obs, info

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None):
        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0

            if not self.observations:
                self.reset(seed=123)

            while frame_count < n_frames:
                done = False
                while not done:
                    action, _ = model.predict(self.observations[-1])
                    obs, reward, term, trunc, info = self.env.step(action)
                    done = term or trunc
                    if done:
                        obs, info = self.env.reset(seed=seed)

                    self.update_buffer(obs, action, reward, done)

                    pbar.update(1)
                    frame_count += 1

    def update_buffer(self, obs, action: None, reward: None, done: None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample_transition_batch(self, batch_size: int = 32):
        idxs = np.random.randint(0, len(self.observations) - 1, size=batch_size)
        obs_batch = np.array([self.observations[idx] for idx in idxs])
        next_obs_batch = np.array([self.observations[idx + 1] for idx in idxs])
        action_batch = np.array([[self.actions[idx]] for idx in idxs])
        reward_batch = np.array([[self.rewards[idx]] for idx in idxs])
        done_batch = np.array([[self.dones[idx]] for idx in idxs])

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (5, 5, 3). Append a 1-channel flag to make (5, 5, 4).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)
        # super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8
        )

    def observation(self, obs):
        # val = 1 if self.env.get_wrapper_attr("step_mode") == 1 else 0 # 0 for no repeat, 1 for repeat
        val = 0
        flag = np.full((obs.shape[0], obs.shape[1], 1), val, dtype=np.uint8)
        return np.concatenate([flag, obs], axis=-1)


class FloatRewardChannel(RecordConstructorArgs, Wrapper):
    """
    Original obs shape (5, 5, 3). Append a 1-channel flag to make (5, 5, 4).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, float(rew), term, trunc, info


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that, with a given probability, performs a second
    'bonus' step using the same action.
    """

    def __init__(self, env: gym.Env, max_steps: int = 100) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        # super().__init__(env)
        self.step_mode = 0
        self.step_count = 0
        self.max_steps = max_steps

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs1, reward1, term1, _, info1 = self.env.step(action)
        self.step_count += 1
        trunc = self.get_trunc()
        reward1 = self._update_step_reward(reward1)

        if term1:
            self.step_mode = 0

        if self.step_mode == 0:
            info1['bonus_step_taken'] = False
            self.step_mode = 1
            return obs1, float(reward1), (term1 or trunc), False, info1
        else:
            info1['bonus_step_taken'] = True
            self.step_mode = 0

            # Second step:
            self.unwrapped.step_count -= 1
            obs2, reward2, term2, _, info2 = self.env.step(action)
            info1.update(info2)
            reward2 = self._update_step_reward(reward2)
            if term2:
                return obs2, float(reward2), term2, False, info1

            # Third step:
            self.unwrapped.step_count -= 1
            obs3, reward3, term3, _, info3 = self.env.step(action)
            info1.update(info3)
            reward3 = self._update_step_reward(reward3)
            return obs3, float(reward3), (term3 or trunc), False, info1

    def get_trunc(self):
        # This is used to set term, NOT trunc
        if self.step_count >= self.max_steps:
            return True
        return False

    @staticmethod
    def _update_step_reward(reward: float) -> float:
        # Simplify the reward to per-step basis
        if reward > 0:
            return reward # 1.0
        return 0.0

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.step_mode = 0
        self.step_count = 0
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
        assert isinstance(env.observation_space['image'], spaces.Box)
        h, w, c = env.observation_space["image"].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8
        )

    def observation(self, obs):
        target_shape = obs['image'].shape
        target_dtype = obs['image'].dtype
        return np.concatenate((np.full(target_shape[:-1] + (1,), fill_value=obs['direction'], dtype=target_dtype),
                               obs["image"]), -1)


class CallablePPO(PPO):
    """
    A version of PPO that can be called like a function to get actions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, obs):
        action, _ = self.predict(obs, deterministic=True)
        return action


class ResidualConvBlock(nn.Module):
    """
    Pre-activation residual block.
    If stride>1 or in/out channels differ, uses a 1x1 projection on the skip.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.act1  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(1, out_ch)
        self.act2  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.act1(self.norm1(x))
        out = self.conv1(out)
        out = self.act2(self.norm2(out))
        out = self.conv2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        return out + identity


class ResidualMLPBlock(nn.Module):
    """
    Two-layer residual MLP block with LayerNorm and ReLU pre-activation.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, hidden)
        self.act1  = nn.ReLU(inplace=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc2   = nn.Linear(hidden, dim)

    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h = self.act1(h)
        h = self.fc2(self.norm2(h))
        return x + h


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention. Produces an HxW mask and reweights all channels."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        nn.init.zeros_(self.conv.weight)  # start near identity

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B,1,H,W]
        return x * a


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        # Init so the block starts as a no-op
        nn.init.kaiming_uniform_(self.fc1.weight, a=1.0)
        nn.init.zeros_(self.fc2.weight)
        nn.init.ones_(self.fc2.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        s = torch.mean(x, dim=(2, 3))            # GAP -> [B, C]
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))           # [B, C]
        s = s.view(s.size(0), s.size(1), 1, 1)
        return x * s

# (Optional) Pixel self-attention; cheap at 7x7. Uncomment to use.
class NonLocalBlock(nn.Module):
    def __init__(self, in_ch: int, inter_ch: int | None = None):
        super().__init__()
        inter_ch = inter_ch or max(1, in_ch // 2)
        self.theta = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.phi   = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.g     = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.out   = nn.Conv2d(inter_ch, in_ch, 1, bias=False)
        # Zero-init output conv for stability (residual starts as identity)
        nn.init.zeros_(self.out.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H*W).transpose(1, 2)   # [B, HW, I]
        phi   = self.phi(x).view(B, -1, H*W)                     # [B, I, HW]
        attn  = torch.softmax(theta @ phi, dim=-1)               # [B, HW, HW]
        g     = self.g(x).view(B, -1, H*W).transpose(1, 2)       # [B, HW, I]
        y     = (attn @ g).transpose(1, 2).view(B, -1, H, W)     # [B, I, H, W]
        return x + self.out(y)


class PPOMiniGridCNN(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], feature_size: int = 128,
                 *args, **kwargs) -> None:
        super().__init__()
        if observation_shape[-1] in (1, 5):
            H, W, C = observation_shape
            self.permute_obs_maybe = lambda x: x.permute(0, 3, 1, 2)
        else:
            C, H, W = observation_shape
            self.permute_obs_maybe = lambda x: x

        C -= 1  # we will ignore the final channel (the always-0 channel in LavaGap)
        self.shrink_obs = lambda x: x[:, :C, :, :]

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=2),
            nn.GroupNorm(1, 16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),

            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        permuted_x = self.permute_obs_maybe(x)
        permuted_x = self.shrink_obs(permuted_x)
        return self.fc(self.cnn(permuted_x))


class OfflineMiniGridCNN(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], feature_size: int = 128,
                 input_scaling: bool = False, *args, **kwargs) -> None:
        super().__init__()
        if observation_shape[-1] in (1, 5):
            H, W, C = observation_shape
            self.permute_obs_maybe = lambda x: x.permute(0, 3, 1, 2)
        else:
            C, H, W = observation_shape
            self.permute_obs_maybe = lambda x: x

        C -= 2  # ignore flag and last (always-zero) channel
        self.shrink_obs = lambda x: x[:, 1:C+1, :, :]

        if input_scaling:
            self.scale_inputs_maybe = self.scale_inputs
        else:
            self.scale_inputs_maybe = lambda x: x

        # --- CNN trunk with channel + spatial attention ---
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),   # 7x7 -> 7x7
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            SEBlock(32, reduction=8),                     # channel attention

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),

            SpatialAttention(kernel_size=7),              # pixel (spatial) attention
            NonLocalBlock(64),                          # optional: enable if you want full pixel self-attention

            nn.MaxPool2d(kernel_size=2, stride=2),        # 7x7 -> 3x3
            nn.Flatten()
        )
        """

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=2),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(16, 32, kernel_size=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]  # likely 64*3*3=576

        self.fc = nn.Sequential(
            nn.Linear(n_flat, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    @staticmethod
    def scale_inputs(x):
        scaled_x = x.clone()
        scaled_x[:, 1, :, :] /= 3.0
        scaled_x[:, 2, :, :] /= 10.0
        scaled_x[:, 3, :, :] /= 5.0
        return scaled_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute_obs_maybe(x)
        x = self.scale_inputs_maybe(x)
        x = self.shrink_obs(x)
        return self.fc(self.cnn(x))


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None: #, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.net = PPOMiniGridCNN(observation_space.shape, feature_size=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class SaveEachBestCallback(BaseCallback):
    """Called by EvalCallback when a new best model is found."""
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0

    def _on_step(self) -> bool:
        # This is triggered by EvalCallback when there's a new best
        self.idx += 1
        best_mean = getattr(self.parent, "best_mean_reward", None)  # provided by EvalCallback
        if best_mean is None:
            fname = f"best_{self.idx:03d}_steps={self.num_timesteps}.zip"
        else:
            fname = f"best_{self.idx:03d}_steps={self.num_timesteps}_mean={best_mean:.2f}.zip"
        path = self.save_dir / fname
        self.model.save(str(path))
        if self.verbose:
            print(f"[SaveEachBest] Saved: {path}")
        return True


class CustomNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int = 1, feature_size: int = 128,
                 is_dummy=False, device: str = 'cpu', feature_extractor=None, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        if feature_extractor is None:
            self.encoder = OfflineMiniGridCNN(observation_shape=observation_shape,
                                              feature_size=feature_size).to(device)
        else:
            self.encoder = feature_extractor

        self.decoder = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LayerNorm(feature_size // 2),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(feature_size // 2, feature_size // 2),
            nn.LayerNorm(feature_size // 2),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(feature_size // 2, output_size)
        ).to(device)

    def forward(self, x: torch.Tensor, action: torch.Tensor = None, flag: torch.Tensor = None) -> torch.Tensor:
        x, action, flag = self._ndarray_to_tensor(x, action, flag)

        # (N, C, H, W) -> (N, feature_size)
        hidden = self.encoder(x)

        # (N, feature_size) -> (N, output_size)
        output = self.decoder(hidden)

        if flag is not None:
            output = output.view(x.size(0), 2, -1)
            output = torch.take_along_dim(output, flag.unsqueeze(-1), 1).squeeze(1)

        if action is None:
            return output

        # (N, output_size) -> (N, 1)
        return output.gather(1, action.long())

    def _ndarray_to_tensor(self, *arrays):
        new_tensors = []
        for array in arrays:
            if array is None:
                new_tensors.append(None)
                continue
            if not isinstance(array, torch.Tensor):
                array = torch.tensor(array, dtype=torch.float32)
            new_tensors.append(array.to(self._device))

        return new_tensors


class CustomIQL(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], action_size: int,  input_length: int = 2,
                 is_dummy: bool = False, feature_size: int = 128, batch_size: int = 128, expectile: float = 0.7,
                 gamma: float = 0.99, critic_lr: float = 3e-4, value_lr: float = 3e-4, actor_lr: float = 3e-4,
                 device: str = 'cpu'):
        super().__init__()
        self._is_dummy = is_dummy
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._expectile = expectile
        self._gamma = gamma
        self._batch_diff = None
        self._input_length = input_length
        self._device = device
        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, device=device)

        self.feature_extractor = OfflineMiniGridCNN(**net_kwargs).to(device)

        _expand_output = (not is_dummy) + 1

        self.critic_net1 = CustomNet(output_size=action_size * _expand_output, feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=action_size * _expand_output, feature_extractor=self.feature_extractor, **net_kwargs)
        self.value_net = CustomNet(output_size=1 * _expand_output, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = CustomNet(output_size=1 * _expand_output, feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = CustomNet(output_size=action_size * _expand_output, feature_extractor=self.feature_extractor, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.Adam(list(self.critic_net1.encoder.parameters()) +
                                             list(self.critic_net1.decoder.parameters()) +
                                             list(self.critic_net2.decoder.parameters()), lr=critic_lr)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr)

        # clone the value net to a target network
        for target_param, param in zip(self.target_value_net.decoder.parameters(), self.value_net.decoder.parameters()):
            target_param.data.copy_(param.data)

    def fit(self, dataset, n_steps: int = 100_000, n_steps_per_epoch: int = 1_000, evaluators=None,
            show_progress: bool = True, experiment_name: str = None):

        total_epochs = n_steps // n_steps_per_epoch + 1

        loss_dict = {
            'critic_loss': [],
            'value_loss': [],
            'policy_loss': []
        }

        for epoch in range(1, total_epochs + 1):
            desc_str = f"{epoch}/{total_epochs}"
            for update_step in tqdm(
                    range(n_steps_per_epoch),
                    disable=not show_progress,
                    mininterval=2.0,
                    desc=desc_str,
                    leave=False
            ):
                minibatch = dataset.sample_transition_batch(self._batch_size)
                obs, acts, rews, next_obs, dones, flag = self._unpack_batch(minibatch)

                # Update the networks
                loss_dict['critic_loss'].append(self._update_critic(obs, acts, rews, next_obs, dones, flag))
                loss_dict['value_loss'].append(self._update_value(obs, acts, flag))
                loss_dict['policy_loss'].append(self._update_actor(obs, acts, flag))

                # Soft update of target value network
                for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            # Logging
            loss_dict = self._log_progress(
                epoch=epoch,
                loss_dict=loss_dict,
                experiment_name=experiment_name,
                evaluators=evaluators
            )

    def forward(self, obs, acts, flag=None):
        if flag is None:
            flag = self._extract_flag(obs)
        q1, q2 = self.critic_net1(obs, acts, flag=flag), self.critic_net2(obs, acts, flag=flag)
        v = self.value_net(obs, flag=flag)
        logits = self.policy_net(obs, flag=flag)
        return (q1, q2), v, logits

    def predict(self, obs, flag=None, deterministic: bool = False):
        if flag is None:
            flag = self._extract_flag(obs)
        logits = self.policy_net(obs, flag=flag)
        if deterministic:
            return logits.argmax(dim=-1).cpu().numpy()
        return Categorical(logits=logits).sample().cpu().numpy()

    def _update_critic(self, obs, acts, rews, next_obs, dones, flag=None):
        q1, q2 = self.critic_net1(obs, acts, flag=flag), self.critic_net2(obs, acts, flag=flag)
        with torch.no_grad():
            next_flag = self._extract_flag(next_obs)
            v_next = self.target_value_net(next_obs, flag=next_flag)
            flag = torch.where(flag == 1, 3, 1)
            q_target = rews + self._gamma ** flag * (1 - dones) * v_next

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def _update_value(self, obs, acts, flag=None):
        v = self.value_net(obs, flag=flag)
        with torch.no_grad():
            q1, q2 = self.critic_net1(obs, acts, flag=flag), self.critic_net2(obs, acts, flag=flag)
            q = torch.min(q1, q2)

        diff = q - v
        self._batch_diff = diff.detach().squeeze()
        weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
        value_loss = (weights * (diff.squeeze() ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.item()

    def _update_actor(self, obs, acts, flag=None):
        logits = self.policy_net(obs, flag=flag)
        weights = self._batch_diff
        # weights = (weights - weights.mean()) / (weights.std() + 1e-6)
        policy_loss = F.cross_entropy(logits, acts.squeeze().long(), reduction='none')
        policy_loss = (policy_loss * torch.clip(torch.exp(2.0 * weights), -torch.inf, 100)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    def _unpack_batch(self, batch):
        obs, acts, rews, next_obs, dones = (batch.observations, batch.actions, batch.rewards,
                                            batch.next_observations, batch.terminals)

        # Convert to torch tensors
        obs, acts, rews, next_obs, dones = self._to_tensors(obs, acts, rews, next_obs, dones)
        flag = self._extract_flag(obs)

        return obs, acts, rews, next_obs, dones, flag

    def _extract_flag(self, obs):
        obs = self._to_tensors(obs)[0]
        if self._is_dummy:
            return None
        if obs.shape[-1] in (1, 5):
            return obs[:, -1, -1, 0].unsqueeze(-1).long()
        return obs[:, 0, -1, -1].unsqueeze(-1).long()

    def _to_tensors(self, *arrays):
        new_tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32)
            new_tensors.append(arr.to(self._device))
        return new_tensors

    def _log_progress(self, epoch: int, loss_dict: dict, experiment_name: str,
                      evaluators: dict = None):
        if evaluators and 'environment' in evaluators:
            mean_reward, std_reward = evaluators['environment'](self)
            print('\n', '=' * 40)
            print(f"Epoch {epoch}: \n     mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"     critic_loss = {np.mean(loss_dict['critic_loss']):.7f}")
            print(f"     value_loss = {np.mean(loss_dict['value_loss']):.7f}")
            print(f"     policy_loss = {np.mean(loss_dict['policy_loss']):.7f}\n")
            print('=' * 40, '\n')

        # Not yet implemented saving logs
        pass

        return dict(critic_loss=[], value_loss=[], policy_loss=[])


@dataclasses.dataclass(frozen=True)
class CustomPartialTrajectory(PartialTrajectory):
    r"""Partial trajectory.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        returns_to_go: Sequence of remaining returns.
        terminals: Sequence of terminal flags.
        timesteps: Sequence of timesteps.
        masks: Sequence of masks that represent padding.
        length: Sequence length.
    """

    observations: ObservationSequence  # (L, ...)
    actions: NDArray  # (L, ...)
    rewards: Float32NDArray  # (L, 1)
    returns_to_go: Float32NDArray  # (L, 1)
    next_observations: ObservationSequence  # (L, ...)
    terminals: Float32NDArray  # (L, 1)
    timesteps: Int32NDArray  # (L,)
    masks: Float32NDArray  # (L,)
    next_masks: Float32NDArray  # (L,)
    length: int


def next_mask_pad_array(array: NDArray, pad_size: int) -> NDArray:
    return np.concatenate((batch_pad_array(array, pad_size), np.ones((1,), dtype=np.float32)), axis=0)


class CustomTrajectorySlicer(TrajectorySlicerProtocol):
    r"""Standard trajectory slicer.

    This class implements a basic trajectory slicing.
    """

    def __call__(
        self, episode: EpisodeBase, end_index: int, size: int
    ) -> PartialTrajectory:
        end = end_index + 1
        next_end = end_index + 2
        start = max(end - size, 0)
        next_start = max(next_end - size, 0)
        actual_size = end - start
        actual_next_size = next_end - next_start

        # slice data
        observations = slice_observations(episode.observations, start, end)
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        ret = np.sum(episode.rewards[start:])
        # cumsum includes the current timestep
        all_returns_to_go = (
            ret
            - np.cumsum(episode.rewards[start:], axis=0)
            + episode.rewards[start:]
        )
        returns_to_go = all_returns_to_go[:actual_size].reshape((-1, 1))

        # prepare terminal flags
        terminals: Float32NDArray = np.zeros((actual_size, 1), dtype=np.float32)
        if episode.terminated and end_index == episode.size() - 1:
            terminals[-1][0] = 1.0
            next_observations = np.zeros((actual_next_size, *observations.shape[1:]))
        else:
            next_observations = slice_observations(episode.observations, next_start, next_end)

        # prepare metadata
        timesteps: Int32NDArray = np.arange(start, end) + 1
        masks: Float32NDArray = np.ones(end - start, dtype=np.float32)
        next_masks: Float32NDArray = np.concatenate((masks[1:], np.ones((1,), dtype=np.float32)), axis=0)

        # compute backward padding size
        pad_size = size - actual_size

        if pad_size == 0:
            return CustomPartialTrajectory(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                next_observations=next_observations,
                terminals=terminals,
                timesteps=timesteps,
                masks=masks,
                next_masks=next_masks,
                length=size,
            )

        return CustomPartialTrajectory(
            observations=batch_pad_observations(observations, pad_size),
            actions=batch_pad_array(actions, pad_size),
            rewards=batch_pad_array(rewards, pad_size),
            returns_to_go=batch_pad_array(returns_to_go, pad_size),
            next_observations=batch_pad_observations(next_observations, pad_size - 1),
            terminals=batch_pad_array(terminals, pad_size),
            timesteps=batch_pad_array(timesteps, pad_size),
            masks=batch_pad_array(masks, pad_size),
            next_masks=next_mask_pad_array(next_masks, pad_size - 1),
            length=size,
        )


@dataclasses.dataclass(frozen=True)
class CustomTrajectoryMiniBatch(TrajectoryMiniBatch):
    r"""Mini-batch of trajectories.

    Args:
        observations: Batched sequence of observations.
        actions: Batched sequence of actions.
        rewards: Batched sequence of rewards.
        returns_to_go: Batched sequence of returns-to-go.
        terminals: Batched sequence of environment terminal flags.
        timesteps: Batched sequence of environment timesteps.
        masks: Batched masks that represent padding.
        length: Length of trajectories.
    """

    observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, L, ...)
    actions: Float32NDArray  # (B, L, ...)
    rewards: Float32NDArray  # (B, L, 1)
    returns_to_go: Float32NDArray  # (B, L, 1)
    next_observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, L, ...)
    terminals: Float32NDArray  # (B, L, 1)
    timesteps: Float32NDArray  # (B, L)
    masks: Float32NDArray  # (B, L)
    next_masks: Float32NDArray  # (B, L)
    length: int

    def __post_init__(self) -> None:
        assert check_dtype(self.observations, np.float32)
        assert check_dtype(self.actions, np.float32)
        assert check_dtype(self.rewards, np.float32)
        assert check_dtype(self.returns_to_go, np.float32)
        assert check_dtype(self.next_observations, np.float32)
        assert check_dtype(self.terminals, np.float32)
        assert check_dtype(self.timesteps, np.float32)
        assert check_dtype(self.masks, np.float32)
        assert check_dtype(self.next_masks, np.float32)

    @classmethod
    def from_partial_trajectories(
        cls, trajectories: Sequence[CustomPartialTrajectory]
    ) -> "TrajectoryMiniBatch":
        r"""Constructs mini-batch from list of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            Mini-batch of trajectories.
        """
        observations = stack_observations(
            [traj.observations for traj in trajectories]
        )
        actions = np.stack([traj.actions for traj in trajectories], axis=0)
        rewards = np.stack([traj.rewards for traj in trajectories], axis=0)
        returns_to_go = np.stack(
            [traj.returns_to_go for traj in trajectories], axis=0
        )
        next_observations = stack_observations(
            [traj.next_observations for traj in trajectories]
        )
        terminals = np.stack([traj.terminals for traj in trajectories], axis=0)
        timesteps = np.stack([traj.timesteps for traj in trajectories], axis=0)
        masks = np.stack([traj.masks for traj in trajectories], axis=0)
        next_masks = np.stack([traj.next_masks for traj in trajectories], axis=0)
        return CustomTrajectoryMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            returns_to_go=cast_recursively(returns_to_go, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            timesteps=cast_recursively(timesteps, np.float32),
            masks=cast_recursively(masks, np.float32),
            next_masks=cast_recursively(next_masks, np.float32),
            length=trajectories[0].length,
        )


class CustomEnvironmentEvaluator:
    def __init__(self, env: RepeatFlagChannel, n_trials: int):
        self.env = env
        self.n_trials = n_trials

    def __call__(self, algo) -> float:
        mean_returns = []
        for _ in range(self.n_trials):
            obs = self.env.reset()[0]
            done = False
            total_reward = 0.0

            while not done:
                with torch.no_grad():
                    action = algo.predict(np.expand_dims(obs, 0))
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            mean_returns.append(total_reward)

        return float(np.mean(mean_returns)), float(np.std(mean_returns) / np.sqrt(self.n_trials))


def make_lavastep_env(*, max_steps=100, **kwargs):
    env_name = "MiniGrid-LavaGapS5-v0"
    # env_name = "MiniGrid-Empty-5x5-v0"
    env = gym.make(env_name, max_episode_steps=None, **kwargs)
    env = FullyObsWrapper(env)
    # env = AlternateStepWrapper(env, max_steps=max_steps)
    env = RecordableImgObsWrapper(env)         # (H,W,C) uint8 image
    env = RepeatFlagChannel(env)     # +1 channel flag
    env = FloatRewardChannel(env)
    return env

def sample_trajectory_batch(
        self, batch_size: int, length: int
) -> CustomTrajectoryMiniBatch:
    return CustomTrajectoryMiniBatch.from_partial_trajectories(
        [self.sample_trajectory(length) for _ in range(batch_size)]
    )

