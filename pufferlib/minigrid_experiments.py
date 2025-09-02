import os

from gymnasium.envs.registration import register, WrapperSpec
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import pickle
import argparse
from collections import defaultdict
import polars as pl

from importable_wrappers import *

train_ppo = False
train_iql = True
render_performance = False

parser = argparse.ArgumentParser()
parser.add_argument('--expectile', default=0.5, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99


"""
Trained on natural dataset:
- evaluated on natural environment
- evaluated on forced 1-step environment - flag forced to 0

Trained on artificial 2-step decoy dataset - flag forced to 0
- evaluated on natural environment - flag forced to 0
- evaluated on forced 1-step environment - flag forced to 0

Trained on artificial 1-step decoy dataset - flag forced to 0
- evaluated on natural environment - flag forced to 0
- evaluated on forced 1-step environment - flag forced to 0
"""


register(
    id="LavaGapS5AltStep-v0",
    # id="EmptyS5NoAltStep-v0",
    entry_point="importable_wrappers:make_lavastep_env",
    # This metadata is what Minari will use to reconstruct later
    additional_wrappers=(
        # WrapperSpec(
            # name="FullyObsWrapper",
            # entry_point="minigrid.wrappers:FullyObsWrapper",
            # kwargs=None,
        # ),
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
        WrapperSpec(
            name="DecoyObsWrapper",
            entry_point="importable_wrappers:DecoyObsWrapper",
            kwargs={},
        ),
    ),
)


if __name__ == "__main__":
    args = parser.parse_args()
    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval
    print(f"EXPECTILE: {EXPECTILE}, DECOY_INTERVAL: {DECOY_INTERVAL}")

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

    if train_iql:
        logs = defaultdict(list)

        # Get our PPO model
        base_env = gym.make(env_name, max_steps=50)
        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_013_steps=22000_mean=0.86.zip',
                                 env=base_env, device="auto")

        # Fill our replay buffer (or load pre-filled)
        replay_buffer_env = ReplayBufferEnv(base_env, buffer_size=1000000)
        if not os.path.exists('./dataset.pkl'):
            replay_buffer_env.fill_buffer(model=model, n_frames=50_000)
            with open('./dataset.pkl', 'wb') as f:
                pickle.dump(replay_buffer_env, f)
                f.close()
        else:
            print('='*50, '\nRe-using existing dataset.pkl...\n', '='*50)
            with open('./dataset.pkl', 'rb') as f:
                replay_buffer_env = pickle.load(f)
                f.close()

        dataset_rewards = np.array(replay_buffer_env.rewards[0])[np.array(replay_buffer_env.dones[0]) == 1]
        dataset_n_episodes = len(dataset_rewards)

        print(f"Baseline reward of the dataset: {dataset_rewards.mean():.2f} "
              f"+/- {dataset_rewards.std() / np.sqrt(dataset_n_episodes):.2f}")

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["natural_alt_step", (0, not DECOY_INTERVAL)],
            ["forced_one_step", (1, False)],  # Will always be set to 0, which is correct
        ]:
            evaluators[key] = CustomEnvironmentEvaluator(gym.make(env_name,
                                                                  max_steps=50,
                                                                  use_flag=flag,
                                                                  forced_interval=interval),
                                                         n_trials=500)

        for n_trial in range(10):
            logs['expectile'].append(EXPECTILE)
            logs['decoy_interval'].append(DECOY_INTERVAL)
            logs['dataset_reward'].append(dataset_rewards.mean())

            # Alternately collect and training
            algo = CustomIQL(observation_shape=base_env.observation_space.shape,
                             action_size=base_env.action_space.n,
                             feature_size=32,
                             batch_size=32,
                             expectile=EXPECTILE,
                             gamma=GAMMA,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
            algo.compile()

            log_dict = algo.fit(
                dataset=replay_buffer_env,
                epochs=1,
                n_steps_per_epoch=10_000,
                evaluators=evaluators,
                dataset_kwargs={'decoy_interval': DECOY_INTERVAL},
            )
            for key in evaluators.keys():
                logs[f'{key}_eval'].append(log_dict[key][0])

        # Save logs
        os.makedirs('./iql_minigrid_logs', exist_ok=True)
        pl.DataFrame(logs).write_csv(f'./iql_minigrid_logs/log_expectile={EXPECTILE}_decoy={DECOY_INTERVAL}.csv')

    if render_performance:
        # Managing your own trainer
        eval_env = gym.make(env_name, max_steps=100, render_mode="human")
        model = CallablePPO.load('./ppo_minigrid_logs/historic_bests/best_013_steps=22000_mean=0.86.zip',
                                 env=eval_env, device="auto")
        observation, info = eval_env.reset(seed=42)
        for _ in tqdm(range(1000)):
            action = model.predict(observation)[0]  # User-defined policy function
            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = eval_env.step(action)

            if terminated or truncated:
                observation, info = eval_env.reset()
        eval_env.close()
