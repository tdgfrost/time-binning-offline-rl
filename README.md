# Temporal Binning Effects on Live Performance in Offline Reinforcement Learning

As presented at the Machine Learning for Health (ML4H) 2025 Findings Track (submission pending). 

## Environment Overview
This open-source code makes use of the LavaGap MiniGrid environment, in which the agent must navigate a partially observable MDP maze in order to reach a goal, whilst avoiding pits of lava. In the original environment, the agent receives a single reward at the end of the trajectory, which is equal to 1 - 0.9 * (steps_taken / max_steps) if the agent reaches its goal, and 0 if the agent falls into lava or reaches the maximum step limit.

The paper uses two modified versions of this environment. In both cases, the reward is now a fixed value of 1.0 if the goal square is reached and 0.0 otherwise.
- Forced One-Step: identical to the original LavaGap environment, aside from changes to the reward.
- Natural Alt-Step: the environment alternates between taking the user action once, and taking the action three times in a row.

For clarity, when we refer to the "environmental time-step", we refer to the true time-step of the underlying environment (e.g., moving from one square to another). When we refer to the "agent time-step", we refer to the time-step as experienced by the agent (e.g., the agent chooses an action once, and the next observation is from a position three squares away).

In both environments, the observation includes the standard 3 channels from LavaGap, but also includes a channel containing the direction the agent is facing, and a channel reflecting the *environmental time-step* for the preceding *agent time-step* (flag = 0 means the last action was taken once, flag = 1 means the last action was taken three times).

## Training PPO (behaviour policy)
The paper starts by training a PPO agent on a version of Natural Alt-Step which includes the conventional LavaGap reward structure, to incentivise quickly solving the maze. This can be reproduced with `python main.py --train_ppo True`, with the model checkpoints saved in the ../logs/ppo_minigrid_logs folder.

## Generating the Dataset and training the IQL agent
The datasets used in the paper consist of the following:

- Natural: the agent sees the agent time-step observation tuples only, as if they had interacted with the Natural Alt-Step environment directly. The "flag" channel is preserved here.
- Artificial One-Step: the agent sees ALL environmental steps. This creates the impression that the behaviour policy was taking a decision every environmental time-step, rather than every agent time-step. The "flag" channel here is set to 0.
- Artificial Two-Step: the agent sees every second environmental step. Similar to Artificial One-Step, this implies two environmental time-steps passed for every one agent time-step. This is analogous to aggregating multiple decisions in a healthcare dataset into a single 4-hour time block, for example. The "flag" channel here is set to 0.

The chosen PPO agent performance can be selected and added to the code by editing line 14 at the top of main.py.

If `python main.py --train_iql True` is run, the script will generate a dataset of 100k frames (if not already generated) using the pre-trained PPO agent, and then train an IQL agent for 10k update steps, evaluating it on the two environments. The dataset generated contains all three possible datasets.

The `main.py` script accepts the following inputs:

`--expectile`: if set to 0.5, the agent is trained using standard behavioural cloning, otherwise this is the expectile value for conventional Implicit Q-Learning.
`--decoy_interval`: if set to 0, the agent is trained on the Natural dataset; if set to 1, the agent is trained on the Artificial One-Step dataset; if set to 2, the agent is trained on the Artificial Two-Step dataset.

## Additional operations
`python train.py --render_performance True --record_video True` will take the provided PPO agent and generate some videos demonstrating navigation of the modified LavaGap environment.

Plotting functions can be found in `plots.py`, and should generate directly by running the script.
