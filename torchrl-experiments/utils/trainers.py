from omegaconf import DictConfig
from tqdm import tqdm
from .hooks import default_hook, DummyProgressBar, DummyHook
from .core.logging import DummyLogger
import torch
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict


class Trainer:
    def __init__(self, collector, replay_buffer, loss_module, optimizer, optim_steps_per_minibatch,
                 loss_hook=default_hook, pre_minibatch_hook=default_hook, evaluate_hook=DummyHook(),
                 total_frames=None, logger=DummyLogger(), save_hook=DummyHook(), clip_grad_norm=True, clip_norm=0.5,
                 progress_bar=False, seed=None):
        # Collectors/loss_modules/optimizers
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.loss_module = loss_module
        self.optim = optimizer
        self.optim_steps_per_minibatch = optim_steps_per_minibatch
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        # Hooks
        self.pre_minibatch_hook = pre_minibatch_hook
        self.loss_hook = loss_hook
        self.evaluate_hook = evaluate_hook
        self.save_hook = save_hook
        self.progress_hook = DummyProgressBar()
        if progress_bar:
            if total_frames is None:
                raise ValueError("If progress_bar is True, total_frames must be specified.")
            self.progress_hook = tqdm(total=total_frames)
        # Logging
        self.logger = logger
        self.progress_bar = progress_bar
        self.total_frames = total_frames
        self.seed = seed
        # Empty attributes
        self.obs_shape = None
        self.frames_seen = 0
        self.update_steps = 0
        self.loss_record = defaultdict(list)

    def train(self):
        for epoch, tensordict_data in enumerate(self.collector):
            if self.obs_shape is None:
                self.obs_shape = tensordict_data["observations"].shape[-3:]
            self.frames_seen += tensordict_data["observations"].view(-1, *self.obs_shape).shape[0]

            self.loss_record[f'epoch_{epoch}'] = defaultdict(list)
            for j in range(self.optim_steps_per_minibatch):
                tensordict_data = self.pre_minibatch_hook(tensordict_data)

                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view)

                for k, batch in enumerate(self.replay_buffer):
                    torch.compiler.cudagraph_mark_step_begin()

                    loss_vals = self.loss_module(batch)
                    loss = self.loss_hook(loss_vals)
                    loss_vals['total_loss'] = loss

                    self.optim.zero_grad()
                    loss.backward()
                    if self.clip_grad_norm:
                        clip_grad_norm_(self.loss_module.parameters(), self.clip_norm)
                    self.optim.step()

                    self.update_steps += 1
                    self.loss_record[f'epoch_{epoch}'][f'minibatch_{j}'].append(loss_vals.detach())
                    self.loss_record[f'flat_epoch_{epoch}'].append(loss_vals.detach())

            # Update progress bar
            self.progress_hook.update(tensordict_data.numel())

            # Make sure collector policy weights are updated
            self.collector.update_policy_weights_()

            # Evaluate
            mean_reward = self.evaluate_hook(frames_seen=self.frames_seen, update_steps=self.update_steps)
            if mean_reward is not None:
                # --- Print out evaluation result --- #
                self._printout_reward(mean_reward=mean_reward, update_progress_bar=self.progress_bar)

            # Logging
            self.logger(loss_record=self.loss_record, frames_seen=self.frames_seen, update_steps=self.update_steps,
                        mean_reward=mean_reward, epoch=epoch)

            # Saving
            self.save_hook(frames_seen=self.frames_seen, update_steps=self.update_steps, mean_reward=mean_reward)

        self.save_hook(force=True)
        return

    def evaluate(self, verbose=True):
        if self.evaluate_hook is None:
            raise ValueError("No evaluate_hook provided. Cannot evaluate.")

        mean_reward = self.evaluate_hook(frames_seen=self.frames_seen, update_steps=self.update_steps, force=True)
        if verbose:
            self._printout_reward(mean_reward=mean_reward)

        return mean_reward

    def _printout_reward(self, mean_reward=None, update_progress_bar=False):
        eval_str = (f"eval mean reward: {mean_reward: 4.4f} "
                    f"at {self.frames_seen} frames")

        print('\n\n===== ', eval_str, ' =====\n')

        if self.progress_bar and update_progress_bar:
            self.progress_hook.set_description(eval_str)


def get_trainer(cfg: DictConfig = None, collector=None, replay_buffer=None, loss_module=None, loss_hook=None,
                optimizer=None, logger=None, evaluate_hook=None, save_hook=None, optim_steps_per_minibatch=None,
                pre_minibatch_hook=None):
    return Trainer(
        collector=collector,
        replay_buffer=replay_buffer,
        loss_module=loss_module,
        loss_hook=loss_hook,
        optimizer=optimizer,
        optim_steps_per_minibatch=optim_steps_per_minibatch,
        pre_minibatch_hook=pre_minibatch_hook,
        total_frames=cfg.collector.total_frames,
        logger=logger,
        evaluate_hook=evaluate_hook,
        save_hook=save_hook,
        clip_grad_norm=True,
        clip_norm=0.5,
        progress_bar=True,
        seed=None,
    )
