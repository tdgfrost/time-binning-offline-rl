import torch
import os
from .core.logging import PPOLogger


def no_grad_update_hook(tensordict_data, some_module=None):
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        tensordict_data = some_module(tensordict_data).clone()
    return tensordict_data


def default_hook(input_data):
    return input_data


def ppo_loss_hook(loss_vals):
    return loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]


def get_ppo_logging_hook(checkpoint_dir=None, writer=None, device=None):
    return PPOLogger(
        checkpoint_dir=checkpoint_dir, writer=writer, device=device
    )


def get_evaluator_hook(env=None, actor=None, eval_steps=1_000_000, eval_freq=100_000, device=None):
    return Evaluator(
        env=env,
        actor=actor,
        eval_steps=eval_steps,
        eval_freq=eval_freq,
        device=device
    )

def get_actor_save_hook(actor=None, optimizer=None, save_freq=100_000, checkpoint_dir=None):
    return DefaultActorSave(
        actor=actor,
        optimizer=optimizer,
        save_freq=save_freq,
        checkpoint_dir=checkpoint_dir
    )


class DummyProgressBar:
    def update(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass


class DefaultActorSave:
    def __init__(self, actor=None, optimizer=None, save_freq=None, checkpoint_dir=None):
        self.actor = actor
        self.optim = optimizer
        self.save_freq = save_freq
        self.save_count = 1
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    def save(self, frames_seen=None, update_steps=None, mean_reward=None, force=False, *args, **kwargs):
        if not force and (frames_seen < self.save_freq * self.save_count):
            return

        if force:
            current_save_dir = os.path.join(self.checkpoint_dir, f'actor_checkpoint_rew_{int(mean_reward)}_final.pt')
        else:
            current_save_dir = os.path.join(self.checkpoint_dir, f'actor_checkpoint_rew_{int(mean_reward)}_frames_{frames_seen // 1_000}k.pt')

        torch.save({
            'frames_seen': frames_seen,
            'update_steps': update_steps,
            'mean_reward': mean_reward,
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, current_save_dir)
        self.save_count += 1
        return


class DummyHook:
    def __call__(self, *args, **kwargs):
        return None


class Evaluator:
    def __init__(self, env=None, actor=None, eval_episodes=10, eval_steps=1000, eval_freq=100_000, device=None):
        self.env = env
        self.actor = actor
        self.eval_episodes = eval_episodes
        self.eval_steps = eval_steps
        self.eval_count = 1
        self.eval_freq = eval_freq
        self.device = device
        self.mean_reward = torch.tensor(0.0, device=device)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, frames_seen=None, force=False, *args, **kwargs):
        if not force and (frames_seen < self.eval_freq * self.eval_count):
            return None

        # We evaluate the policy to get our current mean policy reward.
        with torch.no_grad():
            # --- Rollout with our policy --- #
            eval_rollout = self.env.rollout(self.eval_steps, self.actor, break_when_all_done=True)

        # --- Calculate our mean reward --- #
        n_eval_envs = eval_rollout["observations"].shape[0]
        for env_n in range(n_eval_envs):
            idx = eval_rollout['next', 'terminated'][env_n].nonzero(as_tuple=True)[0][0] + 1
            self.mean_reward += eval_rollout['next', 'reward'][env_n][:idx].sum()
        self.mean_reward /= n_eval_envs
        mean_reward = self.mean_reward.cpu().item()

        self.eval_count += 1
        self.mean_reward *= 0.0
        return mean_reward
