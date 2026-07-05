import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import os

import openpi.models.model as _model
import openpi.models.pi0_align as _pi0_align
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.align_utils as align_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

os.environ["WANDB_MODE"] = "offline"

# import debugpy
# try:
#     debugpy.listen(("localhost", 9602))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Failed to set up debugpy: {e}")
#     pass


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            id="-".join([config.name, config.exp_name]),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        if not getattr(config, "align_enabled", False):
            raise ValueError("scripts/train_align.py requires align_enabled=True.")
        model = _pi0_align.Pi0Align(config, rngs=nnx.Rngs(model_rng))

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    lr_schedule_fn: optax.Schedule,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    align_targets: at.Float[at.Array, "b s d"],
    align_mask: at.Bool[at.Array, "b s"],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        loss, action_loss, align_loss = model.compute_align_losses(
            rng,
            observation,
            actions,
            align_targets,
            align_mask,
            train=True,
        )
        return loss, (action_loss, align_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (action_loss, align_loss)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    learning_rate = lr_schedule_fn(state.step)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "action_loss": action_loss,
        "align_loss": align_loss,
        "learning_rate": learning_rate,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _shardalign_utils(features: align_utils.AlignFeatures, sharding_spec: jax.sharding.Sharding):
    return (
        jax.make_array_from_process_local_data(sharding_spec, features.targets),
        jax.make_array_from_process_local_data(sharding_spec, features.mask),
    )


def _unpack_align_batch(batch):
    if len(batch) == 3:
        observation, actions, sf_identity = batch
        return observation, actions, sf_identity
    observation, actions = batch
    return observation, actions, None


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    device_layout = align_utils.resolve_align_devices(config)
    pi_devices = [jax.devices()[idx] for idx in device_layout.pi_device_ids]
    logging.info("Using PI GPUs %s and VGGT GPUs %s", device_layout.pi_device_ids, device_layout.vggt_device_ids)
    if config.batch_size % len(pi_devices) != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by JAX PI device count {len(pi_devices)}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices, devices=pi_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    observation, actions, sf_identity = _unpack_align_batch(next(data_iter))
    batch = (observation, actions)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    lr_schedule_fn = config.lr_schedule.create()
    ptrain_step = jax.jit(
        functools.partial(train_step, config, lr_schedule_fn),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    reference_tokens = sum((image.shape[1] // 14) * (image.shape[2] // 14) for image in batch[0].images.values())
    extractor = align_utils.create_align_feature_extractor(config, devices=device_layout.vggt_device_ids)
    align_worker = align_utils.AsyncAlignFeatureWorker(extractor)
    align_worker.submit(
        align_utils.prepare_align_observation(observation),
        reference_tokens=reference_tokens,
        sf_identity=None if sf_identity is None else jax.device_get(sf_identity),
    )
    align_features = align_worker.result()

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    timing_infos = []
    try:
        for step in pbar:
            next_observation, next_actions, next_sf_identity = _unpack_align_batch(next(data_iter))
            align_worker.submit(
                align_utils.prepare_align_observation(next_observation),
                reference_tokens=reference_tokens,
                sf_identity=None if next_sf_identity is None else jax.device_get(next_sf_identity),
            )

            feature_wait_start = time.perf_counter()
            align_targets, align_mask = _shardalign_utils(align_features, data_sharding)
            feature_wait_s = time.perf_counter() - feature_wait_start

            train_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch, align_targets, align_mask)
            jax.block_until_ready(info)
            train_step_s = time.perf_counter() - train_start
            infos.append(info)

            timing_info = {
                "feature_wait_s": feature_wait_s,
                "train_step_s": train_step_s,
                "samples_per_s": config.batch_size / train_step_s,
                "steps_per_h": 3600.0 / train_step_s,
            }
            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                if timing_infos:
                    reduced_info.update({key: float(np.mean([info[key] for info in timing_infos])) for key in timing_infos[0]})
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                infos = []
                timing_infos = []

            data_wait_start = time.perf_counter()
            align_features = align_worker.result()
            observation, actions, sf_identity = next_observation, next_actions, next_sf_identity
            batch = (observation, actions)
            timing_info["data_wait_s"] = time.perf_counter() - data_wait_start
            timing_infos.append(timing_info)

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    finally:
        align_worker.close()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
