from typing import Sequence

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
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.models.encoders import register_encoder_factory
import dataclasses

from importable_wrappers import *

train_ppo = False
generate_dataset = False
train_cql = True
render_performance = False

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

        if is_dummy:
            C -= 1  # we will ignore the last channel (the repeat flag)
            self.filter_obs_maybe = lambda x: x[:, :C, :, :]
        else:
            self.filter_obs_maybe = lambda x: x

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
        self.fc = nn.Sequential(nn.Linear(n_flat, feature_size), nn.ReLU())

        if has_mlp:
            self.mlp_maybe = nn.Sequential(nn.Linear(feature_size, feature_size), nn.ReLU())
        else:
            self.mlp_maybe = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute_obs_maybe(x)
        x = self.filter_obs_maybe(x)
        return self.mlp_maybe(self.fc(self.cnn(x)))


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

        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_002_steps=100000_mean=0.38.zip',
                                 env=recorded_env, device="auto")

        # Collect episodes
        target_frames = 100_000
        seed = 123
        model.set_random_seed(123)
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
                    action, _ = model.predict(obs)
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
            algorithm_name="PPO-0.38",
            eval_env=base_env,
            expert_policy=model,
        )

        print("Total episodes collected: ", dataset.total_episodes)
        print("Total steps collected: ", dataset.total_steps)

    if train_cql:
        # Load the dataset via d3rlpy's minari integration
        dataset, _ = d3rlpy.datasets.get_minari(dataset_id)
        # Get the environment for later evaluation
        eval_env = minari.load_dataset(dataset_id).recover_environment()
        env_evaluator = EnvironmentEvaluator(eval_env, n_trials=5)
        # Set up CQL
        for algo_type in ["smart", "dumb"]:
            """
            algo = DiscreteCQLConfig(encoder_factory=MiniGridCNNFactory(feature_size=128,
                                                                        is_dummy=algo_type == "dumb")).create()
            """
            algo = DiscreteBCConfig(batch_size=128,
                                    encoder_factory=MiniGridCNNFactory(feature_size=128,
                                                                       is_dummy=algo_type == "dumb")).create()
            # Train CQL
            algo.fit(
                dataset,
                n_steps=1_000_000,
                n_steps_per_epoch=50_000,
                evaluators={
                    'environment': env_evaluator,
                },
                callback=None,
                experiment_name=f"cql_{algo_type}_minigrid_lavagap_altstep",
                show_progress=False,
            )

            algo.save(f"cql_{algo_type}_minigrid_lavagap_altstep_final.d3")

        # algo = d3rlpy.load_learnable(f"./d3rlpy_logs/cql_smart_minigrid_lavagap_altstep_20250822181226/model_20000.d3")

    if render_performance:
        # Managing your own trainer
        eval_env = gym.make(env_name)
        observation, info = eval_env.reset(seed=42)
        for _ in tqdm(range(1000)):
            action = model.predict(observation)[0]  # User-defined policy function
            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = eval_env.step(action)

            if terminated or truncated:
                observation, info = eval_env.reset()
        eval_env.close()