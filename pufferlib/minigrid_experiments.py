from gymnasium.envs.registration import register, WrapperSpec
from tqdm import tqdm
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from pathlib import Path
from minari import DataCollector
import d3rlpy
import minari
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.preprocessing import StandardObservationScaler

from importable_wrappers import *

train_ppo = False
generate_dataset = True
train_iql = True
render_performance = False

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


register(
    id="LavaGapS7AltStep-v0",
    entry_point="importable_wrappers:make_lavastep_env",
    # This metadata is what Minari will use to reconstruct later
    additional_wrappers=(
        WrapperSpec(
            name="AlternateStepWrapper",
            entry_point="importable_wrappers:AlternateStepWrapper",
            kwargs={},
        ),
        WrapperSpec(
            name="RecordableImgObsWrapper",
            entry_point="importable_wrappers:RecordableImgObsWrapper",
            kwargs={},
        ),
        WrapperSpec(
            name="RepeatFlagChannel",
            entry_point="importable_wrappers:RepeatFlagChannel",
            kwargs={},
        ),
    ),
)


"""
register(
    id="LavaGapS7AltStep-v0",
    entry_point=lambda **kwargs: make_wrapped_env("MiniGrid-LavaGapS7-v0", **kwargs),
)
"""


if __name__ == "__main__":
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env_name = "LavaGapS7AltStep-v0"
    dataset_id = "minigrid_dataset/LavaGapS7AltStepMedium-v0"
    if train_ppo:
        # Create eval callback
        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)
        save_each_best = SaveEachBestCallback(save_dir="./ppo_minigrid_logs/historic_bests", verbose=1)
        eval_callback = EvalCallback(gym.make(env_name),
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     best_model_save_path="./ppo_minigrid_logs")

        model = PPO("CnnPolicy", gym.make(env_name), policy_kwargs=policy_kwargs, verbose=1)
        model.learn(5e5, callback=eval_callback)  # Train for 500,000 step with early stopping

    if generate_dataset:
        base_env = gym.make(env_name)
        recorded_env = DataCollector(base_env, record_infos=True, data_format="arrow")

        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_003_steps=120000_mean=0.57.zip',
                                 env=recorded_env, device="auto")

        # Collect episodes
        target_frames = 100#_000
        seed = 123
        n_frames = 0
        ep_count = 0
        stop_loop = False
        print("\n===== Generating dataset =====\n")
        with tqdm(total=target_frames, desc="Progress", mininterval=2.0) as pbar:
            while not stop_loop:
                ep_count += 1
                obs, info = recorded_env.reset(seed=seed + ep_count)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = recorded_env.step(action)
                    done = terminated or truncated
                    n_frames += 1
                    pbar.update(1)
                    if n_frames >= target_frames:
                        print(f"Reached {n_frames} frames, stopping data collection.")
                        stop_loop = True
                        break

        # Save dataset
        dataset = recorded_env.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="PPO-0.57",
            eval_env=base_env,
            expert_policy=model,
        )

        print("Total episodes collected: ", dataset.total_episodes)
        print("Total steps collected: ", dataset.total_steps)

    if train_iql:
        # Load the dataset via d3rlpy's minari integration
        dataset, _ = d3rlpy.datasets.get_minari(dataset_id)
        # Get the environment for later evaluation
        eval_env = minari.load_dataset(dataset_id).recover_environment()
        # Set up IQL
        algo = DiscreteCQLConfig(observation_scaler=StandardObservationScaler()).create()
        # Train IQL
        algo.fit(dataset, n_steps=1000)
        algo.fit(
            dataset,
            n_steps=200_000,
            n_steps_per_epoch=10_000,
            save_interval=50_000,
            eval_env=eval_env,
            eval_episodes=10,
            eval_interval=10,
            experiment_name="iql_minigrid_lavagap_altstep",
            logdir="./iql_minigrid_logs",
        )

    if render_performance:
        # Managing your own trainer
        eval_env = make_wrapped_env(env_name)
        observation, info = eval_env.reset(seed=42)
        for _ in tqdm(range(1000)):
            action = model.predict(observation)[0]  # User-defined policy function
            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = eval_env.step(action)

            if terminated or truncated:
                observation, info = eval_env.reset()
        eval_env.close()