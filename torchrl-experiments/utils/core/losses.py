from torchrl.objectives import ClipPPOLoss
from omegaconf import DictConfig
from torch import optim


def get_ppo_loss(actor=None, critic=None, cfg: DictConfig = None):
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_bonus=cfg.loss.entropy_bonus,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        loss_critic_type=cfg.loss.loss_critic_type,
        normalize_advantage=cfg.loss.normalize_advantage,
    )

    return loss_module


def get_optimizer(cfg: DictConfig = None, parameters=None):
    if cfg.optimizer.type == "Adam":
        optimizer = optim.Adam(
            parameters,
            lr=cfg.optimizer.learning_rate,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            parameters,
            lr=cfg.optimizer.learning_rate,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer.type}")

    return optimizer