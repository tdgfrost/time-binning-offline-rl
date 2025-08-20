from torchrl.collectors import SyncDataCollector
from torchrl.data import SamplerWithoutReplacement, TensorDictReplayBuffer, LazyTensorStorage
from omegaconf import DictConfig


def get_collector(env=None, actor=None, cfg: DictConfig = None, compilable=False, device=None):
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        split_trajs=False,
        max_frames_per_traj=-1,
        compile_policy=compilable,
        cudagraph_policy={'warmup': 10} if compilable else None,
        device=device,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.collector.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.collector.minibatch_size,
        compilable=compilable
    )

    return collector, replay_buffer