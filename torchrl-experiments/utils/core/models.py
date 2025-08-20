from torchrl.modules import ConvNet, MLP, ProbabilisticActor, OneHotCategorical, ValueOperator, ActorValueOperator
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import ExplorationType
from torch import nn
import torch.nn.functional as F
import torch
from tensordict.nn import TensorDictModule


class ConvLayer(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        if in_channels is None:
            self.conv = nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels)
        self.conv2 = ConvLayer(channels, channels)

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return out + x


class ConvSequence(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaBlock(nn.Module):
    def __init__(self, input_shape=None, output_shape=256, num_cells=(16, 32, 32)):
        super().__init__()
        self.sequences = nn.Sequential()
        in_channels = input_shape

        for cells in num_cells:
            self.sequences.append(ConvSequence(in_channels, cells))
            in_channels = cells

        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(output_shape)

    def forward(self, x):
        batch_dim = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:])
        x = self.sequences(x)
        x = self.flatten(x)
        x = F.relu(x)
        x = F.relu(self.fc(x))
        return x.view(*batch_dim, -1)


def warmup_models(*models, actor=None, env=None, td_data=None, rollout_steps=None):
    if td_data is None and env is None:
        raise ValueError("Either 'data' or 'env' must be provided for warmup.")

    with torch.no_grad():
        if td_data is None:
            td_data = env.rollout(rollout_steps, actor)
        for model in models:
            _ = model(td_data).clone()


def get_ppo_models(env=None, device=None, return_actor_critic=False, common_module_kwargs: dict = None):
    """
    Currently implemented for discrete action spaces only.
    """
    # ======= Common CNN + MLP ====== #
    common_module, cnn_output_shape = get_pixel_module(
        input_shape=env.observation_spec["observations"].shape,
        with_mlp=True,
        device=device,
        **common_module_kwargs
    )

    # ======= Actor/Policy ====== #
    actor_module = get_actor_module(
        input_shape=cnn_output_shape,
        in_keys="hidden_state",
        output_shape=env.action_spec.n,
        action_spec=env.action_spec,
        device=device
    )

    # ======= Critic ====== #
    value_module = get_critic_module(
        input_shape=cnn_output_shape,
        in_keys="hidden_state",
        output_shape=1,
        device=device
    )

    # ======= Combined Model ====== #
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=actor_module,
        value_operator=value_module,
    )

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()
    critic_head = actor_critic.get_value_head()

    if return_actor_critic:
        return actor, critic, critic_head, actor_critic
    return actor, critic, critic_head


def get_pixel_module(input_shape=None, output_shape=None, in_keys="observations", out_keys="hidden_state",
                     with_mlp=True, num_mlp_cells=512, impala=False, device=None):
    if output_shape is not None and not with_mlp:
        raise ValueError("If output_shape is provided, with_mlp must be True.")

    if impala:
        # Impala CNN
        output_shape = 256 if output_shape is None else output_shape
        cnn_module = ImpalaBlock(
            input_shape=input_shape[-3],
            output_shape=output_shape if output_shape is not None else 256,
            num_cells=[16, 32, 32]
        ).to(device)
        cnn_module = TensorDictModule(
                module=cnn_module,
                in_keys=[in_keys],
                out_keys=[out_keys],
        )

        return cnn_module, output_shape
    else:
        # Nature CNN
        # You can also get do ConvNet.default_atari_dqn(num_actions) to get a prebuilt architecture.
        cnn_module = ConvNet(
                activation_class=nn.ReLU,
                num_cells=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                device=device
        )
    with torch.no_grad():
        cnn_output = cnn_module(torch.ones(input_shape, device=device))
        cnn_output_shape = cnn_output.shape[-1]

    if with_mlp:
        output_shape = 512 if output_shape is None else output_shape
        mlp_module = MLP(
                in_features=cnn_output_shape,
                activation_class=torch.nn.ReLU,
                activate_last_layer=True,
                out_features=output_shape,
                num_cells=[num_mlp_cells],
                device=device,
        )
        with torch.no_grad():
            mlp_output = mlp_module(cnn_output)
            output_shape = mlp_output.shape[-1]

        main_module = TensorDictModule(
                module=torch.nn.Sequential(cnn_module, mlp_module),
                in_keys=[in_keys],
                out_keys=[out_keys],
        )

    else:
        output_shape = cnn_output_shape
        main_module = TensorDictModule(
                module=cnn_module,
                in_keys=[in_keys],
                out_keys=[out_keys],
        )

    return main_module, output_shape


def get_actor_module(input_shape=None, output_shape=None, action_spec=None, in_keys="observations",
                     num_cells=512, device=None):
    """
    Currently implemented for discrete action spaces only.
    """
    policy_net = MLP(
        in_features=input_shape,
        out_features=output_shape,
        activation_class=torch.nn.ReLU,
        num_cells=[num_cells],
        device=device,
    )

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=[in_keys],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
        safe=True,
    )

    return policy_module


def get_critic_module(input_shape=None, output_shape=None, in_keys="observations", num_cells=512, device=None):
    """
    Currently implemented for discrete action spaces only.
    """
    critic_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=input_shape,
        out_features=output_shape,
        num_cells=[num_cells],
        device=device,
    )
    critic_module = ValueOperator(
        critic_net,
        in_keys=[in_keys],
    )

    return critic_module


def get_gae_module(cfg, value_network=None, device=None):
    advantage_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=value_network,
        average_gae=cfg.loss.average_gae,
        device=device,
    )

    return advantage_module
