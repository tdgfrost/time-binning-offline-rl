from typing import Tuple, Dict, Any, Union
import numpy as np
from gymnasium import spaces, ObservationWrapper
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from minigrid.wrappers import ImgObsWrapper, Wrapper
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import deque

class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (7, 7, 3). Append a 1-channel flag to make (7, 7, 4).
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
        val = 0 if self.env.get_wrapper_attr("step_mode") == 1 else 1 # 0 for no repeat, 1 for repeat
        flag = np.full((obs.shape[0], obs.shape[1], 1), val, dtype=np.uint8)
        return np.concatenate([obs, flag], axis=-1)


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
        obs1, reward1, term1, trunc1, info1 = self.env.step(action)
        term1, trunc1 = self.override_trunc(term1, trunc1)

        if term1 or trunc1:
            self.step_mode = 0

        if self.step_mode == 0:
            info1['bonus_step_taken'] = False
            self.step_mode = 1
            return obs1, float(reward1), term1, trunc1, info1
        else:
            info1['bonus_step_taken'] = True
            self.step_mode = 0
            self.unwrapped.step_count -= 1
            obs2, reward2, term2, trunc2, info2 = self.env.step(action)
            term2, trunc2 = self.override_trunc(term2, trunc2)
            info1.update(info2)
            full_reward = reward1 + reward2
            if term2:
                return obs2, float(full_reward), term2, trunc2, info1

            # If we are still not done, we return the second observation
            self.unwrapped.step_count -= 1
            obs3, reward3, term3, trunc3, info3 = self.env.step(action)
            term3, trunc3 = self.override_trunc(term3, trunc3)
            info1.update(info3)
            full_reward += reward3
            return obs3, float(full_reward), term3, trunc3, info1

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.step_mode = 0
        obs, info = self.env.reset(*args, **kwargs)
        info['bonus_step_taken'] = False
        return obs, info

    @staticmethod
    def override_trunc(term: bool, trunc: bool) -> Tuple[bool, bool]:
        # Apparently required for compatibility with d3rlpy - must be mutually exclusive, and only registers
        # episodes as terminated if the `term` flag is positive.
        if trunc:
            return True, False
        return term, trunc

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


class MiniGridCNN(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], feature_size: int = 128, is_dummy=False,
                 has_mlp=False) -> None:
        super().__init__()
        if observation_shape[-1] in (1, 4):
            H, W, C = observation_shape
            self.permute_obs_maybe = lambda x: x.permute(0, 3, 1, 2)
        else:
            C, H, W = observation_shape
            self.permute_obs_maybe = lambda x: x

        C -= 1  # we will ignore the last channel (the repeat flag)
        self.split_obs = lambda x: (x[:, :C, :, :], x[:, -1, 0, 0].unsqueeze(-1))
        if is_dummy:
            self.process_flag_maybe = lambda x: torch.zeros_like(x)
        else:
            self.process_flag_maybe = nn.Sequential(
                nn.Linear(1, feature_size // 2),
                nn.ReLU(),
                nn.Linear(feature_size // 2, feature_size)
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),

            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, feature_size),
            nn.ReLU(),
        )

        if has_mlp:
            self.mlp_maybe = nn.Sequential(
                nn.Linear(feature_size, feature_size // 2),
                nn.ReLU(),

                nn.Linear(feature_size // 2, feature_size // 2),
                nn.ReLU(),
            )
        else:
            self.mlp_maybe = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute_obs_maybe(x)
        x, x_flag = self.split_obs(x)
        x_flag = self.process_flag_maybe(x_flag)
        return self.mlp_maybe(self.fc(self.cnn(x)) + x_flag)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None: #, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.net = MiniGridCNN(observation_space.shape, feature_size=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


@dataclasses.dataclass()
class MiniGridCNNFactory(d3rlpy.models.encoders.EncoderFactory):
    is_dummy: bool = False
    has_mlp: bool = True
    feature_size: int = 128

    def create(self, observation_shape: Sequence[int]) -> nn.Module:
        return MiniGridCNN(tuple(observation_shape), feature_size=self.feature_size, is_dummy=self.is_dummy,
                           has_mlp=self.has_mlp)

    @staticmethod
    def get_type() -> str:
        return "minigrid_cnn"


# Register in d3rlpy (so the model can be saved/loaded)
register_encoder_factory(MiniGridCNNFactory)

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
                 is_dummy=False, device: str = 'cpu', *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self.encoder = MiniGridCNN(observation_shape=observation_shape,
                                   feature_size=feature_size,
                                   is_dummy=is_dummy,
                                   has_mlp=False).to(device)

        self.lstm = nn.LSTM(feature_size, feature_size, batch_first=True).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),

            nn.Linear(feature_size // 2, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, output_size)
        ).to(device)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None, action: torch.Tensor = None) -> torch.Tensor:
        x, masks, action = self._ndarray_to_tensor(x, masks, action)

        # x: (N, L, C, H, W) -> (N*L, C, H, W)
        N = x.shape[0]
        x = x.view(-1, *x.shape[2:])

        # (N*L, C, H, W) -> (N, L, feature_size)
        x = self.encoder(x)
        x = x.view(N, -1, x.shape[-1])

        # Pack the padded sequences
        x = self._pack_sequence(x, masks)
        x, _ = self.lstm(x)

        # Unpack the padded sequences
        # (N, L, feature_size) -> (N, feature_size)
        x = self._unpack_sequence(x)

        # (N, feature_size) -> (N, output_size)
        output = self.decoder(x)
        if action is None:
            return output

        # (N, output_size) -> (N, 1)
        return output.gather(1, action.long())

    @staticmethod
    def _pack_sequence(x: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        We have a sequence (N, L, C) which goes from old to new. However, this sequence is also LEFT zero-padded.
        We must first reverse the sequence to make it RIGHT padded, then pack it, and then feed to LSTM.
        After LSTM, we must unpack it and reverse it back to the original order.
        """

        N, L, C = x.shape
        lengths = masks.sum(dim=1).long()
        shifts = (L - lengths).clamp(min=0)  # how much left-padding each row has

        # Build per-row gather indices that rotate the sequence so real steps move to the front
        base = torch.arange(L, device=x.device).unsqueeze(0).expand(N, -1)  # (N, L)
        idx = (base + shifts.unsqueeze(1)) % L  # (N, L)

        # Right-pad without changing temporal order (old→new is preserved)
        x_right_padded = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))  # (N, L, C)

        # Pack and run the LSTM
        packed = pack_padded_sequence(x_right_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        return packed

    def _unpack_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unpacked_x, lengths = pad_packed_sequence(x, batch_first=True)
        unpacked_x = torch.take_along_dim(unpacked_x, (lengths.view(-1, 1, 1) - 1).to(self._device), dim=1).squeeze()
        return unpacked_x

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
                 is_dummy: bool = False, feature_size: int = 128, batch_size: int = 128, expectile: float = 0.8,
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
        net_kwargs = dict(observation_shape=observation_shape, feature_size=feature_size, is_dummy=is_dummy,
                          device=device)

        self.critic_net1 = CustomNet(output_size=action_size, **net_kwargs)
        self.critic_net2 = CustomNet(output_size=action_size, **net_kwargs)
        self.value_net = CustomNet(output_size=1, **net_kwargs)
        self.target_value_net = CustomNet(output_size=1, **net_kwargs)
        self.policy_net = CustomNet(output_size=action_size, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.Adam(list(self.critic_net1.parameters()) +
                                             list(self.critic_net2.parameters()), lr=critic_lr)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr)

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
                minibatch = dataset.sample_trajectory_batch(self._batch_size, length=self._input_length)
                obs, acts, rews, next_obs, dones, masks, next_masks = self._unpack_batch(minibatch)

                # Update the networks
                loss_dict['critic_loss'].append(self._update_critic(obs, acts, rews, next_obs, dones, masks, next_masks))
                loss_dict['value_loss'].append(self._update_value(obs, acts, masks))
                loss_dict['policy_loss'].append(self._update_actor(obs, acts, masks))

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

    def forward(self, obs, acts, masks):
        q1, q2 = self.critic_net1(obs, masks, acts), self.critic_net2(obs, masks, acts)
        v = self.value_net(obs, masks)
        logits = self.policy_net(obs, masks)
        return (q1, q2), v, logits

    def predict(self, obs, masks, deterministic: bool = False):
        logits = self.policy_net(obs, masks)
        if deterministic:
            return logits.argmax(dim=-1).cpu().numpy()
        return Categorical(logits=logits).sample().cpu().numpy()

    def _update_critic(self, obs, acts, rews, next_obs, dones, masks, next_masks):
        q1, q2 = self.critic_net1(obs, masks, acts), self.critic_net2(obs, masks, acts)
        with torch.no_grad():
            v_next = self.target_value_net(next_obs, next_masks)
            flag = obs[:, -1, -1, -1, -1].unsqueeze(-1)
            flag = torch.where(flag == 1, 3, 1)
            q_target = rews + self._gamma ** flag * (1 - dones) * v_next

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def _update_value(self, obs, acts, masks):
        v = self.value_net(obs, masks)
        with torch.no_grad():
            q1, q2 = self.critic_net1(obs, masks, acts), self.critic_net2(obs, masks, acts)
            q = torch.min(q1, q2)

        diff = q - v
        self._batch_diff = diff.detach().squeeze()
        weights = torch.absolute(self._expectile - (diff < 0).float()).squeeze()
        value_loss = (weights * (diff.squeeze() ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.item()

    def _update_actor(self, obs, acts, masks):
        logits = self.policy_net(obs, masks)
        policy_loss = F.cross_entropy(logits, acts.squeeze().long(), reduction='none')
        policy_loss = (policy_loss * torch.exp(2.0 * self._batch_diff)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    def _unpack_batch(self, batch):
        (obs, acts, rews, next_obs,
         dones, masks, next_masks) = (batch.observations, batch.actions, batch.rewards,
                                      batch.next_observations, batch.terminals, batch.masks, batch.next_masks)

        # Only obs/next_obs/masks should be sequential - the rest can be indexed to the final position
        acts, rews, dones = acts[:, -1], rews[:, -1], dones[:, -1]

        # Convert to torch tensors
        (obs, acts, rews, next_obs,
         dones, masks, next_masks) = self._to_tensors(obs, acts, rews, next_obs, dones, masks, next_masks)

        return obs, acts, rews, next_obs, dones, masks, next_masks

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
    def __init__(self, env: RepeatFlagChannel, n_trials: int, input_length: int = 2):
        self.env = env
        self.n_trials = n_trials
        self.input_length = input_length

    def __call__(self, algo) -> float:
        mean_returns = []
        for _ in range(self.n_trials):
            obs = self.env.reset()[0]
            done = False
            total_reward = 0.0
            obs_queue = deque(maxlen=self.input_length)
            mask_queue = deque(maxlen=self.input_length)
            for _ in range(self.input_length - 1):
                obs_queue.append(np.zeros_like(obs))
                mask_queue.append(0.0)
            obs_queue.append(obs)
            mask_queue.append(1.0)

            while not done:
                stacked_obs = np.stack(obs_queue, axis=0)[None, ...]
                masks = np.array(mask_queue, dtype=np.float32)[None, ...]
                with torch.no_grad():
                    action = algo.predict(stacked_obs, masks=masks)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                obs_queue.append(obs)
                mask_queue.append(1.0)

            mean_returns.append(total_reward)

        return float(np.mean(mean_returns)), float(np.std(mean_returns) / np.sqrt(self.n_trials))


def make_lavastep_env(**kwargs):
    env = gym.make("MiniGrid-LavaGapS7-v0", **kwargs)
    env = AlternateStepWrapper(env)
    env = RecordableImgObsWrapper(env)         # (H,W,C) uint8 image
    env = RepeatFlagChannel(env)     # +1 channel flag
    return env

def sample_trajectory_batch(
        self, batch_size: int, length: int
) -> CustomTrajectoryMiniBatch:
    return CustomTrajectoryMiniBatch.from_partial_trajectories(
        [self.sample_trajectory(length) for _ in range(batch_size)]
    )

