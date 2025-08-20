from multiprocessing.spawn import freeze_support
import torch
import hydra
from omegaconf import DictConfig

from utils.core.envs import make_atari_env, make_parallel_atari_env
from utils.core.models import get_ppo_models, get_gae_module
from utils.core.losses import get_ppo_loss, get_optimizer
from utils.core.collectors import get_collector
from utils.core.logging import start_logging
from utils.hooks import (no_grad_update_hook, ppo_loss_hook, get_evaluator_hook, get_actor_save_hook,
                         get_ppo_logging_hook)
from utils.trainers import get_trainer
from functools import partial

# --- Additional hyperparameters
# Run with `python train_ppo.py`, or `python train_ppo.py env=Breakout` to specify the environment.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compilable = True
logging = True
model_save = True


@hydra.main(version_base=None, config_path="conf/ppo_conf", config_name="config")
def main(cfg: DictConfig):
    cfg = cfg.env
    # --- Get our "setup" environment --- #
    proof_env = make_atari_env(
        env_name=cfg.env.env_name,
        frame_skip=cfg.env.frame_skip,
        device=device,
        is_test=False
    )

    # --- Get our PPO models --- #
    actor, value, value_head = get_ppo_models(env=proof_env, device=device, common_module_kwargs={'impala': True})

    # --- Compile our models if needed --- #
    advantage_module = get_gae_module(cfg=cfg, value_network=value, device=device)

    # --- Get our loss function --- #
    loss_module = get_ppo_loss(actor=actor, critic=value_head, cfg=cfg)

    # --- Get our optimizer --- #
    optimizer = get_optimizer(cfg=cfg, parameters=loss_module.parameters())
    proof_env.close()

    # --- Get our data collector --- #
    train_env = make_parallel_atari_env(n_envs=cfg.env.n_train_envs, cfg=cfg, device=device, is_test=False)
    eval_env = make_parallel_atari_env(n_envs=cfg.env.n_eval_envs, cfg=cfg, device=device, is_test=True)

    collector, replay_buffer = get_collector(env=train_env, actor=actor, cfg=cfg, compilable=compilable, device=device)

    # --- Set up evaluator --- #
    evaluator = get_evaluator_hook(env=eval_env, actor=actor, eval_freq=cfg.logging.eval_freq, device=device)

    # --- Set up logging --- #
    checkpoint_dir, writer = start_logging(env_name=cfg.env.env_name, model_save=model_save, logging=logging)
    logging_hook = get_ppo_logging_hook(checkpoint_dir=checkpoint_dir, writer=writer, device=device)

    # -- Set up saving --- #
    save_hook = get_actor_save_hook(actor=actor, optimizer=optimizer, save_freq=cfg.logging.save_freq,
                                    checkpoint_dir=checkpoint_dir)

    # --- Set up trainer --- #
    trainer = get_trainer(
        cfg=cfg, collector=collector, replay_buffer=replay_buffer, loss_module=loss_module, loss_hook=ppo_loss_hook,
        optimizer=optimizer, logger=logging_hook, evaluate_hook=evaluator, save_hook=save_hook,
        optim_steps_per_minibatch=cfg.loss.ppo_epochs,
        pre_minibatch_hook=partial(no_grad_update_hook, some_module=advantage_module)
    )

    # --- Train! --- #
    trainer.train()

    # --- Perform final evaluation --- #
    mean_reward = trainer.evaluate(verbose=False)

    print("✅ Done: PPO trained for {env_name}.")
    print("Total frames seen:", trainer.frames_seen)
    print("Total update steps:", trainer.update_steps)
    print("Mean reward:", mean_reward)

    writer.close()
    collector.shutdown()
    eval_env.close()

if __name__ == '__main__':
    # Ensure that the script can be run in a multiprocessing context
    freeze_support()
    # Run the main function
    main()
