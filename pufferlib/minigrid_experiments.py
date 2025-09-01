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
import pickle

from importable_wrappers import *

train_ppo = True
generate_dataset = False
train_iql = False
render_performance = False

GAMMA = 0.99


register(
    id="LavaGapS5AltStep-v0",
    # id="EmptyS5NoAltStep-v0",
    entry_point="importable_wrappers:make_lavastep_env",
    # This metadata is what Minari will use to reconstruct later
    additional_wrappers=(
        WrapperSpec(
            name="FullyObsWrapper",
            entry_point="minigrid.wrappers:FullyObsWrapper",
            kwargs=None,
        ),
        # WrapperSpec(
            # name="AlternateStepWrapper",
            # entry_point="importable_wrappers:AlternateStepWrapper",
            # kwargs={},
        # ),
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
        WrapperSpec(
            name="FloatRewardChannel",
            entry_point="importable_wrappers:FloatRewardChannel",
            kwargs={},
        )
    ),
)


if __name__ == "__main__":
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env_name = "LavaGapS5AltStep-v0"
    # env_name = "EmptyS5NoAltStep-v0"
    dataset_id = "minigrid_dataset/LavaGapS5AltStepMedium-v0"
    # dataset_id = "minigrid_dataset/EmptyS5NoAltStepExpert-v0"
    if train_ppo:
        # Create eval callback
        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)
        save_each_best = SaveEachBestCallback(save_dir="./ppo_minigrid_logs/historic_bests", verbose=1)
        eval_callback = EvalCallback(gym.make(env_name, max_steps=100),
                                     n_eval_episodes=50,
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     eval_freq=1000,
                                     deterministic=False,
                                     best_model_save_path="./ppo_minigrid_logs")

        model = PPO("CnnPolicy", gym.make(env_name, max_steps=100), ent_coef=0.1,
                    policy_kwargs=policy_kwargs, gamma=GAMMA, verbose=1)
        model.learn(5e5, callback=eval_callback)  # Train for 500,000 step with early stopping

    if False: # generate_dataset:
        base_env = gym.make(env_name, max_steps=50)
        recorded_env = DataCollector(base_env, record_infos=True, data_format="arrow")

        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_002_steps=20000_mean=0.90.zip',
                                 env=recorded_env, device="auto")

        # Collect episodes
        target_frames = 100_000
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
                    done = terminated or truncated  # should never be truncated
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

    if generate_dataset:  # Temp
        base_env = gym.make(env_name, max_steps=50)
        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_011_steps=18000_mean=0.74.zip',
                                 env=base_env, device="auto")
        env_evaluator = CustomEnvironmentEvaluator(gym.make(env_name, max_steps=50), n_trials=50)

        replay_buffer_env = ReplayBufferEnv(base_env, buffer_size=1000000)

        # Alternately collect and training
        algo = CustomIQL(observation_shape=base_env.observation_space.shape,
                         action_size=base_env.action_space.n,
                         feature_size=128,
                         batch_size=128,
                         expectile=0.7,
                         gamma=GAMMA,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        algo.compile()

        # Start training
        if not os.path.exists('./dataset.pkl'):
            replay_buffer_env.fill_buffer(model=model, n_frames=10_000, seed=123)
            with open('./dataset.pkl', 'wb') as f:
                pickle.dump(replay_buffer_env, f)
                f.close()
        else:
            with open('./dataset.pkl', 'rb') as f:
                replay_buffer_env = pickle.load(f)
                f.close()
        epoch = 0
        eval_interval = 1_000
        training_steps = 10_000
        n_steps = 0
        policy_losses = deque(maxlen=100)
        critic_losses = deque(maxlen=100)
        value_losses = deque(maxlen=100)
        with tqdm(total=training_steps, desc="Progress", mininterval=2.0) as pbar:
            while n_steps < training_steps:
                # Sample buffer
                obs, acts, rews, next_obs, dones = replay_buffer_env.sample_transition_batch(batch_size=32)

                # Update algo
                flags = algo._extract_flag(obs)
                obs, acts, rews, next_obs, dones, flags = algo._to_tensors(obs, acts, rews, next_obs, dones, flags)
                critic_losses.append(algo._update_critic(obs, acts, rews, next_obs, dones, flag=flags))
                value_losses.append(algo._update_value(obs, acts, flag=flags))
                policy_losses.append(algo._update_actor(obs, acts, flag=flags))

                pbar.update(1)
                pbar.set_postfix(policy_loss=f"{np.mean(policy_losses):.5f}",
                                 critic_loss=f"{np.mean(critic_losses):.5f}",
                                 value_loss=f"{np.mean(value_losses):.5f}")
                n_steps += 1

                # Evaluate
                if n_steps % eval_interval == 0:
                    epoch += 1
                    mean_reward, std_reward = env_evaluator(algo)
                    print('\n', '=' * 40)
                    print(f"Epoch {epoch}: \n     mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"     policy_loss = {np.mean(policy_losses):.7f}\n")
                    print('=' * 40, '\n')

    if train_iql:
        # Load the dataset via d3rlpy's minari integration
        dataset, _ = d3rlpy.datasets.get_minari(dataset_id, trajectory_slicer=CustomTrajectorySlicer())
        dataset.sample_trajectory_batch = types.MethodType(sample_trajectory_batch, dataset)
        input_length = 8  # For LSTM model
        # Get the environment for later evaluation
        eval_env = minari.load_dataset(dataset_id).recover_environment()
        env_evaluator = CustomEnvironmentEvaluator(eval_env, n_trials=50)
        # Set up IQL
        for algo_type in ["smart", "dumb"]:
            algo = CustomIQL(observation_shape=eval_env.observation_space.shape,
                             action_size=eval_env.action_space.n,
                             is_dummy=algo_type == "dumb",
                             feature_size=128,
                             batch_size=128,
                             gamma=GAMMA,
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

            full_eval_result = CustomEnvironmentEvaluator(eval_env, n_trials=500)(algo)

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
        eval_env = gym.make(env_name, max_steps=100, render_mode="human")
        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_002_steps=20000_mean=0.82.zip',
                                 env=eval_env, device="auto")
        observation, info = eval_env.reset(seed=42)
        for _ in tqdm(range(1000)):
            action = model.predict(observation)[0]  # User-defined policy function
            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = eval_env.step(action)

            if terminated or truncated:
                observation, info = eval_env.reset()
        eval_env.close()
