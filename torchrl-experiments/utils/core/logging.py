import subprocess
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch


def start_tensorboard(logdir='./logs'):
    # Check if tensorboard is already running
    try:
        result = subprocess.run(['pgrep', '-f', 'tensorboa'], stdout=subprocess.PIPE)
        if result.returncode == 0:
            return

    except Exception as e:
        print(f'Error checking Tensorboard processes: {e}')

    logdir = os.path.abspath(logdir)

    # The following ensures the tensorboard instance persists even when the script stops
    subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--load_fast=false'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

def start_logging(env_name=None, model_save=False, logging=False):
    now = datetime.now()
    current_time = now.strftime('%H%M%S')
    current_date = now.strftime('%Y-%m-%d')
    model_name = f"{env_name}_{current_date}-{current_time}"

    if model_save:
        checkpoint_folder = f"./models/PPO/{model_name}/checkpoints"
        os.makedirs(checkpoint_folder, exist_ok=True)
    else:
        checkpoint_folder = None

    # Create Tensorboard for logging
    if logging:
        log_dir = f"./logs/PPO/{model_name}"
        tb_writer = SummaryWriter(log_dir=log_dir)
        start_tensorboard(logdir=log_dir)
    else:
        tb_writer = DummyWriter()

    return checkpoint_folder, tb_writer

class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def flush(self):
        pass


class DummyLogger:
    def __call__(self, *args, **kwargs):
        pass


class PPOLogger:
    def __init__(self, checkpoint_dir=None, writer=None, device=None):
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer
        self.device = device
        self.total_loss = torch.tensor(0.0, device=self.device)
        self.loss_objective = torch.tensor(0.0, device=self.device)
        self.loss_critic = torch.tensor(0.0, device=self.device)

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def log(self, loss_record=None, frames_seen=None, update_steps=None, epoch=None, mean_reward=None, *args, **kwargs):
        # Log losses
        total_loss, loss_objective, loss_critic = self.unnest_and_sum_loss(loss_record=loss_record, epoch=epoch)

        for step_key, step in [['frames', frames_seen], ['updates', update_steps]]:
            for log_key, value in [
                ['loss', total_loss], ['loss_objective', loss_objective], ['loss_critic', loss_critic],
            ]:
                self.writer.add_scalar(f'train_{log_key}_per_{step_key}', value, step)

        # Log mean reward
        if mean_reward is None:
            return

        self.writer.add_scalar('eval_reward_mean_per_frames', mean_reward, frames_seen)
        self.writer.add_scalar('eval_reward_mean_per_updates', mean_reward, update_steps)
        self.writer.flush()

        return

    def unnest_and_sum_loss(self, loss_record=None, epoch=None):
        loss_vals = loss_record[f"flat_epoch_{epoch}"]
        for loss_element in loss_vals:
            self.total_loss += loss_element["total_loss"]
            self.loss_objective += loss_element["loss_objective"]
            self.loss_critic += loss_element["loss_critic"]

        total_loss, loss_objective, loss_critic = self.move_losses_to_cpu_and_reset()

        return total_loss, loss_objective, loss_critic

    def move_losses_to_cpu_and_reset(self):
        new_items = (self.total_loss.cpu().item(), self.loss_objective.cpu().item(), self.loss_critic.cpu().item())
        for old_item in [self.total_loss, self.loss_objective, self.loss_critic]:
            old_item *= 0
        return new_items