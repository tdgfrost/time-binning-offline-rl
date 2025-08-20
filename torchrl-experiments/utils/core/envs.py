from typing import Optional, Any

from torchrl.envs import GymEnv, ParallelEnv, TransformedEnv, Compose, NoopResetEnv, ToTensorImage, GrayScale, Resize, \
    RenameTransform, CatFrames, RewardSum, StepCounter, DoubleToFloat, SignTransform, TimeMaxPool, Transform
from torchrl.data.tensor_specs import Unbounded
import torch
from tensordict import TensorDictBase


class AlternatingStep(Transform):
    """
    A TorchRL Transform that alternates between taking one and two steps
    in the parent environment for each call to `step()`.

    This transform is stateful and correctly modifies the environment spec.
    """

    def __init__(self):
        super().__init__()

    def transform_output_spec(self, spec: TensorDictBase) -> TensorDictBase:
        # Define the spec for our integer 'step_skip' key.
        step_skip_spec = Unbounded(
            shape=(1,),
            device=spec.device,
            dtype=torch.int64,
        )

        # Add the key to the root spec (under 'observation_spec').
        spec.set(("full_observation_spec", "step_skip"), step_skip_spec.clone())

        # Add the "intermediate" key, which holds a full 'next' state.
        reference_spec = spec.clone()
        for spec_key, reference_keys in [
            ("full_observation_spec", ["observations", "episode_reward", "step_count", "step_skip"]),
            ("full_done_spec", ["done", "terminated", "truncated"]),
            ("full_reward_spec", ["reward"])
        ]:
            for key in reference_keys:
                spec.set(("full_observation_spec", "intermediate", key),
                         reference_spec.get(spec_key).get(key).clone())

        return spec

    def _reset(
            self, td: TensorDictBase, td_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets the transform's internal state (`step_mode`)."""
        # We need to know the device, which we can get from the input.
        device = td_reset.device

        # Initialize the step_mode for the new episode.
        self.step_mode = torch.randint(1, 3, (1,), device=device)

        # Add the initial 'step_skip' value to the reset tensordict.
        td_reset.set("step_skip", self.step_mode.clone())
        # Set the "intermediate" key to the reset tensordict.
        td_reset.set("intermediate", td_reset.clone())
        return td_reset

    def _alternate_step(self):
        self.step_mode = self.step_mode % 2 + 1

    def _step(self, initial_td: TensorDictBase, next_td, **kwargs) -> TensorDictBase:
        """
        Overrides the step behavior to perform one or two steps.
        This is the core logic of the transform.
        """
        # --- Mode 1: Take a single environment step ---
        if "step_skip" not in initial_td.keys():
            # Initialize the step mode if not present
            initial_td.set("step_skip", self.step_mode.clone())

        elif next_td['terminated'] or next_td['truncated']:
            # If the environment is done, we should not skip steps
            self.step_mode = self.step_mode * 0 + 1  # Reset to mode 1
            initial_td.set("step_skip", self.step_mode.clone())

        elif next_td['reward'] != 0:
            # If a point has been scored, we should not skip steps
            self.step_mode = self.step_mode * 0 + 1  # Reset to mode 1
            initial_td.set("step_skip", self.step_mode.clone())

        if initial_td.get('step_skip') == 1:
            # Set the "intermediate" key to the next_td
            next_td.set("intermediate", next_td.clone())
            next_td.set(("intermediate", "step_skip"), initial_td.get("step_skip").clone())
            # Alternate the mode for the *next* step
            self._alternate_step()
            next_td.set("step_skip", self.step_mode.clone())
            return next_td

        # --- Mode 2: Take two consecutive environment steps ---
        else:
            # Create an intermediate tensordict to hold the state after the first step
            one_extra_td = next_td.clone()
            one_extra_td.set('action', initial_td.get('action').clone())

            # Perform our second step
            self.parent.step(one_extra_td)

            # Update our next_td with the final state
            next_td.set("intermediate", next_td.clone())
            next_td.set(("intermediate", "step_skip"), initial_td.get("step_skip").clone())
            next_td.update(one_extra_td.get("next").clone())
            self._alternate_step()
            next_td.set("step_skip", self.step_mode.clone())
            return next_td


def make_atari_env(env_name=None, frame_skip=None, device=None, is_test=False):
    env = GymEnv(f"{env_name}NoFrameskip-v4", from_pixels=True, frame_skip=frame_skip, device=device,
                 categorical_action_encoding=False)
    transform = Compose(
        NoopResetEnv(noops=30, random=True),
        ToTensorImage(from_int=True),  # Additionally scales to 0-1
        GrayScale(),
        Resize((84, 84)),
        RenameTransform(in_keys=["pixels"], out_keys=["observations"]),  # Rename observation key
        TimeMaxPool(in_keys=["observations"], T=2),
        CatFrames(N=frame_skip, dim=-3, in_keys=["observations"], padding="constant", padding_value=0),  # Stack frames
        RewardSum(),
        StepCounter(),
    )
    if not is_test:
        env.append_transform(SignTransform(in_keys=["reward"]))
    env.append_transform(DoubleToFloat())
    env = TransformedEnv(env, transform)
    env.append_transform(AlternatingStep())
    return env


def make_parallel_atari_env(n_envs=None, cfg=None, device=None, is_test=False):
    env_generator_kwargs = {
        'env_name': cfg.env.env_name,
        'frame_skip': cfg.env.frame_skip,
        'device': device,
        'is_test': is_test
    }
    return make_parallel_env(n_envs=n_envs, env_generator_fn=make_atari_env, env_generator_kwargs=env_generator_kwargs)


def make_parallel_env(n_envs=None, env_generator_fn=None, env_generator_kwargs=None):
    return ParallelEnv(
        n_envs,
        env_generator_fn,
        create_env_kwargs=env_generator_kwargs
    )
