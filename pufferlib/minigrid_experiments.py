import glob
import os
import types

from gymnasium.envs.registration import register, WrapperSpec
from tqdm import tqdm
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from minari import DataCollector
import d3rlpy
import minari
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig, DiscreteIQLConfig
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.preprocessing import StandardObservationScaler

from importable_wrappers import *

train_ppo = False
generate_dataset = False
train_iql = True
render_performance = False


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
        target_frames = 500_000
        seed = 123
        ep_count = 0
        frame_count = 0
        stop_loop = False
        print("\n===== Generating dataset =====\n")
        with tqdm(total=target_frames, desc="Progress", mininterval=2.0) as pbar:
            while not stop_loop:
                ep_count += 1
                obs, info = recorded_env.reset(seed=seed + ep_count)
                done = False
                while not done:
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = recorded_env.step(action)
                    done = terminated or truncated
                    pbar.update(1)
                    if frame_count >= target_frames:
                        print(f"Reached {frame_count} frames, stopping data collection.")
                        stop_loop = True
                        break

        # Save dataset
        dataset = recorded_env.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="PPO-0.50",
            eval_env=base_env,
            expert_policy=model,
        )

        print("Total episodes collected: ", dataset.total_episodes)
        print("Total steps collected: ", dataset.total_steps)

    if False: # train_iql:
        # Load the dataset via d3rlpy's minari integration
        dataset, _ = d3rlpy.datasets.get_minari(dataset_id)
        # Get the environment for later evaluation
        eval_env = minari.load_dataset(dataset_id).recover_environment()
        env_evaluator = EnvironmentEvaluator(eval_env, n_trials=50)
        # Set up IQL
        for algo_type in ["smart", "dumb"]:
            algo = DiscreteIQLConfig(batch_size=128,
                                     # observation_scaler=StandardObservationScaler(),
                                     actor_encoder_factory=MiniGridCNNFactory(feature_size=128,
                                                                              is_dummy=algo_type == "dumb"),
                                     value_encoder_factory=MiniGridCNNFactory(feature_size=128,
                                                                              is_dummy=algo_type == "dumb"),
                                     critic_encoder_factory=MiniGridCNNFactory(feature_size=128,
                                                                              is_dummy=algo_type == "dumb"),
                                     expectile=0.8,
                                     ).create()

            # Train iql
            experiment_name = f"iql_{algo_type}_minigrid_lavagap_altstep"
            algo.fit(
                dataset,
                n_steps=200_000,
                n_steps_per_epoch=1_000,
                evaluators={
                    'environment': env_evaluator,
                },
                callback=None,
                experiment_name=experiment_name,
                show_progress=True,
            )

            full_eval_result = EnvironmentEvaluator(eval_env, n_trials=500)(algo, dataset=None)

            print(f"Overall result of {algo_type}: ", full_eval_result)

            # Get full directory name
            experiment_dir = glob.glob(f"./d3rlpy_logs/{experiment_name}_*")[0]
            with open(os.path.join(experiment_dir, "final_eval.txt"), "w") as f:
                f.write(str(full_eval_result))

        # algo = d3rlpy.load_learnable(f"./d3rlpy_logs/cql_smart_minigrid_lavagap_altstep_20250822181226/model_20000.d3")

    if train_iql:
        # Load the dataset via d3rlpy's minari integration
        dataset, _ = d3rlpy.datasets.get_minari(dataset_id, trajectory_slicer=CustomTrajectorySlicer())
        dataset.sample_trajectory_batch = types.MethodType(sample_trajectory_batch, dataset)
        input_length = 8  # For LSTM model
        # Get the environment for later evaluation
        eval_env = minari.load_dataset(dataset_id).recover_environment()
        env_evaluator = CustomEnvironmentEvaluator(eval_env, n_trials=50, input_length=input_length)
        # Set up IQL
        for algo_type in ["smart", "dumb"]:
            algo = CustomIQL(observation_shape=eval_env.observation_space.shape,
                             action_size=eval_env.action_space.n,
                             is_dummy=algo_type == "dumb",
                             feature_size=128,
                             batch_size=128,
                             input_length=input_length,
                             device='cuda' if torch.cuda.is_available() else 'cpu')

            algo.compile()

            # Train iql
            experiment_name = f"iql_{algo_type}_minigrid_lavagap_altstep"
            algo.fit(
                dataset,
                n_steps=200_000,
                n_steps_per_epoch=1_000,
                evaluators={
                    'environment': env_evaluator,
                },
                experiment_name=experiment_name,
                show_progress=True,
            )

            full_eval_result = CustomEnvironmentEvaluator(eval_env, n_trials=500, input_length=input_length)(algo)

            print(f"Overall result of {algo_type}: ", full_eval_result)

            # Get full directory name
            """
            experiment_dir = glob.glob(f"./d3rlpy_logs/{experiment_name}_*")[0]
            with open(os.path.join(experiment_dir, "final_eval.txt"), "w") as f:
                f.write(str(full_eval_result))
            """
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