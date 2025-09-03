from typing import Tuple, Dict, Any, Union, Sequence, Optional
import numpy as np
from gymnasium import spaces, ObservationWrapper
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from minigrid.wrappers import ImgObsWrapper, Wrapper, FullyObsWrapper
from stable_baselines3 import PPO
import torch.nn as nn
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque


class ReplayBufferEnv:
    def __init__(self, env, buffer_size: int = 100000):
        self.observations = {
            0: deque(maxlen=buffer_size),
            1: deque(maxlen=buffer_size),
            2: deque(maxlen=buffer_size)
        }
        self.actions = {
            0: deque(maxlen=buffer_size),
            1: deque(maxlen=buffer_size),
            2: deque(maxlen=buffer_size)
        }
        self.rewards = {
            0: deque(maxlen=buffer_size),
            1: deque(maxlen=buffer_size),
            2: deque(maxlen=buffer_size)
        }
        self.dones = {
            0: deque(maxlen=buffer_size),
            1: deque(maxlen=buffer_size),
            2: deque(maxlen=buffer_size)
        }

        self.buffer_size = buffer_size
        self.env = env
        self._tensors_set = False
        self._device = None

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed)
        ep_buffer = self._reset_ep_buffer(obs, info)
        return obs, info, ep_buffer

    @staticmethod
    def _reset_ep_buffer(obs, info):
        return {
            'obs': [obs], 'decoy_obs': [info['obs'][0]],
            'action': [], 'decoy_action': [],
            'reward': [], 'decoy_reward': [],
            'done': [], 'decoy_done': [],
        }

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None):
        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0
            if seed is None:
                seed = 123

            obs, info, ep_buffer = self.reset(seed=seed)
            model.set_random_seed(seed)

            while frame_count < n_frames:
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, term, trunc, info = self.env.step(action)
                    done = term or trunc

                    self.update_episode_buffer(obs, action, reward, done, info, ep_buffer)

                    if done:
                        ep_buffer['obs'] = ep_buffer['obs'][:-1]
                        ep_buffer['decoy_obs'] = ep_buffer['decoy_obs'][:-1]
                        obs, info = self.env.reset(seed=seed + frame_count)

                    pbar.update(1)
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)

                # Add ep_buffer to permanent buffer
                self.update_permanent_buffer(ep_buffer)
                # Reset ep_buffer and add 'obs' to it
                ep_buffer = self._reset_ep_buffer(obs, info)

            # Add a garbage all-zeros "final obs"
            for i in [0, 1, 2]:
                self.observations[i] += [np.zeros_like(self.observations[0][0])]

    def set_to_tensors(self, device: str = 'cpu'):
        if self._tensors_set:
            if self._device == device:
                return
            for i in [0, 1, 2]:
                self.observations[i] = self.observations[i].to(device)
                self.actions[i] = self.actions[i].to(device)
                self.rewards[i] = self.rewards[i].to(device)
                self.dones[i] = self.dones[i].to(device)
        else:
            for i in [0, 1, 2]:
                self.observations[i] = torch.from_numpy(np.array(self.observations[i])).to(device)
                self.actions[i] = torch.from_numpy(np.array(self.actions[i])).to(device)
                self.rewards[i] = torch.from_numpy(np.array(self.rewards[i])).to(device)
                self.dones[i] = torch.from_numpy(np.array(self.dones[i])).to(device)

        self._tensors_set = True
        self._device = device

    @staticmethod
    def update_episode_buffer(obs, action: Union[int, np.ndarray], reward: float, done: bool, info: dict, ep_buffer: dict):
        for key in ['obs', 'action', 'reward', 'done']:
            ep_buffer[f'decoy_{key}'] += info[key]

        ep_buffer['obs'] += [obs]
        ep_buffer['action'] += [action]
        ep_buffer['reward'] += [reward]
        ep_buffer['done'] += [done]

    def update_permanent_buffer(self, ep_buffer: dict):
        for i, decoy_maybe in [(0, ''), (1, 'decoy_')]:
            self.observations[i] += ep_buffer[f'{decoy_maybe}obs']
            self.actions[i] += ep_buffer[f'{decoy_maybe}action']
            self.rewards[i] += ep_buffer[f'{decoy_maybe}reward']
            self.dones[i] += ep_buffer[f'{decoy_maybe}done']

        obs, actions, rewards, dones = [
            [ep_buffer[f'decoy_{key}'][idx] for idx in range(0, len(ep_buffer[f'decoy_{key}']), 2)]
            for key in ['obs', 'action', 'reward', 'done']
        ]
        if not dones[-1]:
            rewards[-1] = ep_buffer['reward'][-1]
            dones[-1] = ep_buffer['done'][-1]
        assert dones[-1], "Last done flag should be True"
        self.observations[2] += obs
        self.actions[2] += actions
        self.rewards[2] += rewards
        self.dones[2] += dones

    def sample_transition_batch(self, batch_size: int = 32, decoy_interval: int = 0):
        assert self._tensors_set, "Replay buffer must be set to tensors first using .set_to_tensors(model)"
        idxs = torch.randint(0, len(self.observations[decoy_interval]) - 1, (batch_size,), device=self._device)

        obs_batch = self.observations[decoy_interval][idxs]
        next_obs_batch = self.observations[decoy_interval][idxs + 1]
        action_batch = self.actions[decoy_interval][idxs].unsqueeze(-1)
        reward_batch = self.rewards[decoy_interval][idxs].unsqueeze(-1)
        done_batch = self.dones[decoy_interval][idxs].unsqueeze(-1)

        flags = self._extract_flags(obs_batch)
        next_flags = self._extract_flags(next_obs_batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, flags, next_flags

    @staticmethod
    def _extract_flags(obs):
        if obs.shape[-1] in (1, 5):
            return obs[:, -1, -1, 0].unsqueeze(-1).long()
        return obs[:, 0, -1, -1].unsqueeze(-1).long()


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that, with a given probability, performs a second
    'bonus' step using the same action.
    """

    def __init__(self, env: gym.Env, max_steps: int = 100, forced_interval: int = 0) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        # super().__init__(env)
        self.last_step_mode = 0
        self.current_step_mode = 0
        self.step_count = 0
        self.max_steps = max_steps
        assert 0 <= forced_interval <= 1, "Forced interval must be 0 or 1"
        self.forced_interval = forced_interval

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.last_step_mode = 0
        self.current_step_mode = 0
        self.step_count = 0
        obs, info = self.env.reset(*args, **kwargs)
        info['obs'] = [obs]
        info['action'] = []
        info['reward'] = []
        info['done'] = []
        info = self._take_no_additional_steps(info)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Do our first genuine environment step
        obs, reward, term, trunc, info = self._take_first_step(action)

        # Decide if we are forcing 1-step intervals
        if self.forced_interval:
            info = self._take_no_additional_steps(info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        # Take 1-step only (or if environment has already terminated)
        if term or self.current_step_mode == 0:
            # Update step_mode for next observation
            self._flip_step_modes()

            info = self._take_no_additional_steps(info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        # Take 3-steps
        elif self.current_step_mode == 1:
            # Update step_mode for next observation
            self._flip_step_modes()

            # Take another step (second)
            obs, reward, term, _, info = self._take_another_step(action, info)
            if term:
                info['done'][-1] = term or trunc
                return obs, reward, (term or trunc), False, info

            # Take another step (third)
            obs, reward, term, _, info = self._take_another_step(action, info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

    def _flip_step_modes(self):
        self.last_step_mode = self.current_step_mode
        self.current_step_mode = 1 - self.current_step_mode

    def _take_first_step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, _, info = self.env.step(action)
        self.step_count += 1
        trunc = self._get_trunc()
        reward = self._update_step_reward(reward)
        info['obs'] = [obs]
        info['action'] = [action]
        info['reward'] = [reward]
        info['done'] = [False]
        return obs, reward, term, trunc, info

    @staticmethod
    def _take_no_additional_steps(info):
        info['bonus_step_taken'] = False
        return info

    def _take_another_step(self, action: Any, base_info: Dict[str, Any], record_obs: bool = False) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Take additional step
        base_info['bonus_step_taken'] = True
        self.unwrapped.step_count -= 1
        obs, reward, term, _, new_info = self.env.step(action)

        # Update return variables
        base_info, reward = self._update_info_and_reward(base_info, new_info, reward)
        base_info['obs'].append(obs)
        base_info['action'].append(action)
        base_info['reward'].append(reward)
        base_info['done'].append(False)
        return obs, reward, term, _, base_info

    def _update_info_and_reward(self, base_info, new_info, reward):
        base_info.update(new_info)
        reward = self._update_step_reward(reward)
        return base_info, reward

    def _get_trunc(self):
        # This is used to set term, NOT trunc
        if self.step_count >= self.max_steps:
            return True
        return False

    @staticmethod
    def _update_step_reward(reward: float) -> float:
        # Simplify the reward to per-step basis
        if reward > 0:
            return 1.0  # float(reward)
        return 0.0


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (5, 5, 3). Append a 1-channel flag to make (5, 5, 4).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """
    def __init__(self, env, use_flag: bool = True):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)
        # super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8
        )
        self.use_flag = use_flag

    def observation(self, obs):
        # Concat flag (0/1) to the start of the channels
        # - always set to 0 if use_flag = False
        val = 1 if self.env.get_wrapper_attr("last_step_mode") == 1 and self.use_flag else 0 # 0 for no repeat, 1 for repeat
        flag = np.full((obs.shape[0], obs.shape[1], 1), val, dtype=np.uint8)
        return np.concatenate([flag, obs], axis=-1)


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


class DecoyObsWrapper(RecordConstructorArgs, Wrapper):
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = self._fill_obs(info)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.step(action)
        info = self._fill_obs(info)
        return obs, rew, term, trunc, info

    @staticmethod
    def _fill_obs(info):
        for idx, vanilla_obs in enumerate(info['obs']):
            obs, direction = vanilla_obs['image'], vanilla_obs['direction']
            target_shape = obs.shape[:-1] + (1,)
            # Concat flag to the start of the channels
            obs = np.concatenate((np.full(target_shape, fill_value=direction, dtype=obs.dtype),
                                  obs), -1)

            # Always set decoy obs flag to zero and concat to the start of the channels
            flag = np.full(target_shape, 0, dtype=obs.dtype)
            obs = np.concatenate([flag, obs], axis=-1)
            info['obs'][idx] = obs
        return info


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

            nn.AdaptiveAvgPool2d(output_size=1),
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
            # nn.Dropout(p=0.2),
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
                 device: str = 'cpu', feature_extractor=None, *args, **kwargs) -> None:
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

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for Conv2d/Linear with Kaiming normal,
        biases with zeros, Norm layers with weight=1, bias=0.
        The final Linear layer in the decoder is zero-initialized.
        """
        def _init_fn(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to all submodules
        self.apply(_init_fn)

        # special-case: zero-init the very last decoder layer
        if isinstance(self.decoder[-1], nn.Linear):
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None, flags: torch.Tensor = None) -> torch.Tensor:
        x, actions, flags = self._ndarray_to_tensor(x, actions, flags)
        x = x.to(dtype=torch.float32)

        # (N, C, H, W) -> (N, feature_size)
        hidden = self.encoder(x)

        # (N, feature_size) -> (N, output_size)
        output = self.decoder(hidden)

        if flags is not None:
            output = output.view(x.size(0), 2, -1)
            output = torch.take_along_dim(output, flags.long().unsqueeze(-1), 1).squeeze(1)

        if actions is None:
            return output

        # (N, output_size) -> (N, 1)
        return output.gather(1, actions.long())

    def _ndarray_to_tensor(self, *arrays):
        new_tensors = []
        for array in arrays:
            if array is None:
                new_tensors.append(None)
                continue
            if not isinstance(array, torch.Tensor):
                array = torch.tensor(array)
            new_tensors.append(array.to(self._device))

        return new_tensors


class CustomIQL(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], action_size: int,  input_length: int = 2,
                 feature_size: int = 128, batch_size: int = 128, expectile: float = 0.7,
                 gamma: float = 0.99, critic_lr: float = 3e-4, value_lr: float = 3e-4, actor_lr: float = 3e-4,
                 device: str = 'cpu'):
        super().__init__()
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._expectile = expectile
        self._gamma = gamma
        self._batch_diff = None
        self._input_length = input_length
        self._device = device
        self._cloning_only = expectile == 0.5
        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, device=device)

        self.feature_extractor = OfflineMiniGridCNN(**net_kwargs).to(device)

        self.critic_net1 = CustomNet(output_size=action_size * 2, feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=action_size * 2, feature_extractor=self.feature_extractor, **net_kwargs)
        self.value_net = CustomNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = CustomNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = CustomNet(output_size=action_size * 2, feature_extractor=self.feature_extractor, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.AdamW(list(self.critic_net1.encoder.parameters()) +
                                             list(self.critic_net1.decoder.parameters()) +
                                             list(self.critic_net2.decoder.parameters()), lr=critic_lr)
        self.value_optim = torch.optim.AdamW(self.value_net.parameters(), lr=value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=actor_lr)

        # clone the value net to a target network
        for target_param, param in zip(self.target_value_net.decoder.parameters(), self.value_net.decoder.parameters()):
            target_param.data.copy_(param.data)

    def fit(self, dataset, epochs: int = 1, n_steps_per_epoch: int = 1_000, evaluators=None,
            show_progress: bool = True, experiment_name: str = None, dataset_kwargs: Optional[Dict] = None):
        # Initialise our dataset and loss dictionary
        dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        dataset.set_to_tensors(self._device)

        loss_dict = self._reset_loss_dict()

        # Start training
        with tqdm(total=epochs * n_steps_per_epoch, desc="Progress", mininterval=2.0, disable=not show_progress) as pbar:
            for epoch in range(1, epochs + 1):
                epoch_str = f"{epoch}/{epochs}"

                for update_step in range(n_steps_per_epoch):
                    obs, acts, rews, next_obs, dones, flags, next_flags = dataset.sample_transition_batch(
                        self._batch_size, **dataset_kwargs
                    )

                    # Update the networks
                    if not self._cloning_only:
                        loss_dict['critic_loss'].append(self._update_critic(obs, acts, rews, next_obs, dones, flags, next_flags))
                        loss_dict['value_loss'].append(self._update_value(obs, acts, flags))
                    loss_dict['policy_loss'].append(self._update_actor(obs, acts, flags))

                    # Soft update of target value network
                    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

                    pbar.update(1)
                    pbar.set_postfix(epoch=epoch_str,
                                     policy_loss=f"{np.mean(loss_dict['policy_loss']):.5f}",
                                     critic_loss=f"{np.mean(loss_dict['critic_loss']):.5f}",
                                     value_loss=f"{np.mean(loss_dict['value_loss']):.5f}",
                                     refresh=False)

                # Logging
                loss_dict, log_dict = self._log_progress(
                    epoch=epoch,
                    loss_dict=loss_dict,
                    experiment_name=experiment_name,
                    evaluators=evaluators
                )

        return log_dict

    def forward(self, obs, acts, flags=None):
        if flags is None:
            flags = self._extract_flags(obs)
        q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
        v = self.value_net(obs, flags=flags)
        logits = self.policy_net(obs, flags=flags)
        return (q1, q2), v, logits

    def predict(self, obs, flags=None, deterministic: bool = False):
        obs = self._to_tensors(obs)[0]
        if flags is None:
            flags = self._extract_flag(obs)
        logits = self.policy_net(obs, flags=flags)
        if deterministic:
            return logits.argmax(dim=-1).cpu().numpy()
        return Categorical(logits=logits).sample().cpu().numpy()

    def _update_critic(self, obs, acts, rews, next_obs, dones, flags=None, next_flags=None):
        q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
        with torch.no_grad():
            v_next = self.target_value_net(next_obs, flags=next_flags)
            multistep_discount = torch.where(next_flags == 1, 3, 1)
            q_target = rews.float() + self._gamma ** multistep_discount * (1 - dones.float()) * v_next

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def _update_value(self, obs, acts, flags=None):
        v = self.value_net(obs, flags=flags)
        with torch.no_grad():
            q1, q2 = self.critic_net1(obs, acts, flags=flags), self.critic_net2(obs, acts, flags=flags)
            q = torch.min(q1, q2)

        diff = q - v
        self._batch_diff = diff.detach().squeeze()
        weights = torch.absolute(self._expectile - (self._batch_diff < 0).float()).squeeze()
        value_loss = (weights * (diff.squeeze() ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.item()

    def _update_actor(self, obs, acts, flags=None):
        logits = self.policy_net(obs, flags=flags)
        weights = 1.0
        if not self._cloning_only:
            weights = self._batch_diff
            weights = torch.clip(torch.exp(2.0 * weights), -torch.inf, 100)

        policy_loss = F.cross_entropy(logits, acts.squeeze().long(), reduction='none')
        policy_loss = (policy_loss * weights).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    def _extract_flag(self, obs):
        obs = self._to_tensors(obs)[0]
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

        rewards = {}
        for key in evaluators.keys():
            mean_rew, std_rew = evaluators[key](self)
            rewards[key] = (mean_rew, std_rew)

        eval_str = '\n' + '=' * 40 + f"\nEpoch {epoch}:"
        for key in rewards.keys():
            eval_str += f"\n     {key} = {rewards[key][0]:.2f} +/- {rewards[key][1]:.2f}"

        eval_str += f"\n\n     policy_loss = {np.mean(loss_dict['policy_loss']):.7f}"
        eval_str += f"\n     critic_loss = {np.mean(loss_dict['critic_loss']):.7f}"
        eval_str += f"\n     value_loss = {np.mean(loss_dict['value_loss']):.7f}\n"
        eval_str += '=' * 40 + '\n'
        print(eval_str)

        return self._reset_loss_dict(), rewards

    @staticmethod
    def _reset_loss_dict():
        return {
            'critic_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100)
        }


class CustomEnvironmentEvaluator:
    def __init__(self, env: RepeatFlagChannel, n_trials: int):
        self.env = env
        self.n_trials = n_trials

    def __call__(self, algo) -> float:
        mean_returns = []
        for _ in range(self.n_trials):
            obs, info = self.env.reset()
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


def make_lavastep_env(*, max_steps=100, forced_interval: int = 0, use_flag: bool = True, **kwargs):
    env_name = "MiniGrid-LavaGapS6-v0"
    # env_name = "MiniGrid-Empty-6x6-v0"
    env = gym.make(env_name, max_episode_steps=None, **kwargs)
    # env = FullyObsWrapper(env)
    env = AlternateStepWrapper(env, max_steps=max_steps, forced_interval=forced_interval)
    env = RecordableImgObsWrapper(env)         # (H,W,C) uint8 image
    env = RepeatFlagChannel(env, use_flag=use_flag)     # +1 channel flag
    env = DecoyObsWrapper(env)
    return env

