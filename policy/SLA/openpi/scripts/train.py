import dataclasses
import functools
import logging
import os
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

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.distributed as _distributed
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

os.environ["WANDB_MODE"]="offline"


def init_logging():
    """Custom logging format for better readability."""
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
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
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
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

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
def _apply_gradients(
    config: _config.TrainConfig,
    lr_schedule_fn: optax.Schedule,
    state: training_utils.TrainState,
    model: _model.BaseModel,
    grads: at.Params,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
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
        "learning_rate": learning_rate,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _compute_loss_and_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    model: _model.BaseModel,
    batch: tuple[_model.Observation, _model.Actions],
    micro_step: at.Int[at.ArrayLike, ""],
) -> tuple[at.Array, at.Params]:
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

    step_rng = training_utils.train_rng(rng, state.step, micro_step, config.gradient_accumulation_steps)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    return nnx.value_and_grad(loss_fn, argnums=diff_state)(model, step_rng, observation, actions)


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    lr_schedule_fn: optax.Schedule,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    loss, grads = _compute_loss_and_grads(config, rng, state, model, batch, jnp.array(0))
    new_state, update_info = _apply_gradients(config, lr_schedule_fn, state, model, grads)
    return new_state, {"loss": loss, **update_info}


@at.typecheck
def accumulate_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    gradient_accumulator: at.Params,
    batch: tuple[_model.Observation, _model.Actions],
    micro_step: at.Int[at.ArrayLike, ""],
) -> tuple[at.Params, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    loss, grads = _compute_loss_and_grads(config, rng, state, model, batch, micro_step)
    gradient_accumulator = training_utils.accumulate_gradients(
        gradient_accumulator, grads, config.gradient_accumulation_steps
    )
    return gradient_accumulator, {"loss": loss}


@at.typecheck
def apply_accumulated_gradients(
    config: _config.TrainConfig,
    lr_schedule_fn: optax.Schedule,
    state: training_utils.TrainState,
    gradient_accumulator: at.Params,
) -> tuple[training_utils.TrainState, at.Params, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    new_state, info = _apply_gradients(config, lr_schedule_fn, state, model, gradient_accumulator)
    return new_state, training_utils.init_gradient_accumulator(gradient_accumulator), info


def main(config: _config.TrainConfig):
    init_logging()
    dist_info = _distributed.initialize()
    logging.info(
        "Running on: %s | process=%s/%s",
        platform.node(),
        dist_info.process_index,
        dist_info.process_count,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    accumulation_steps = config.gradient_accumulation_steps
    logging.info(
        "Gradient accumulation: micro_batch_size=%s, accumulation_steps=%s, effective_batch_size=%s",
        config.batch_size,
        accumulation_steps,
        config.effective_batch_size,
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices, num_nodes=dist_info.node_count)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    if dist_info.is_primary:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    _distributed.sync_global_devices("data loader initialized")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if dist_info.is_primary and jax.device_count() == 1:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    lr_schedule_fn = config.lr_schedule.create()
    if accumulation_steps == 1:
        ptrain_step = jax.jit(
            functools.partial(train_step, config, lr_schedule_fn),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        gradient_sharding = train_state_sharding.params.filter(config.trainable_filter)
        gradient_accumulator = jax.jit(
            training_utils.init_gradient_accumulator,
            in_shardings=(gradient_sharding,),
            out_shardings=gradient_sharding,
        )(train_state.params.filter(config.trainable_filter))
        paccumulate_train_step = jax.jit(
            functools.partial(accumulate_train_step, config),
            in_shardings=(
                replicated_sharding,
                train_state_sharding,
                gradient_sharding,
                data_sharding,
                replicated_sharding,
            ),
            out_shardings=(gradient_sharding, replicated_sharding),
            donate_argnums=(2,),
        )
        papply_accumulated_gradients = jax.jit(
            functools.partial(apply_accumulated_gradients, config, lr_schedule_fn),
            in_shardings=(train_state_sharding, gradient_sharding),
            out_shardings=(train_state_sharding, gradient_sharding, replicated_sharding),
            donate_argnums=(0, 1),
        )

    start_step = int(train_state.step)
    show_tqdm = _distributed.show_tqdm(is_primary=dist_info.is_primary)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not show_tqdm,
    )

    infos = []
    timing_infos = []
    loop_start = time.perf_counter()
    for step in pbar:
        if dist_info.is_primary and step == start_step:
            logging.info("Starting first train step; JAX compilation may take several minutes.")
        train_step_s = 0.0
        data_wait_s = 0.0
        micro_infos = []
        for micro_step in range(accumulation_steps):
            train_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                if accumulation_steps == 1:
                    train_state, info = ptrain_step(train_rng, train_state, batch)
                else:
                    gradient_accumulator, micro_info = paccumulate_train_step(
                        train_rng,
                        train_state,
                        gradient_accumulator,
                        batch,
                        jnp.array(micro_step, dtype=jnp.int32),
                    )
                    jax.block_until_ready(micro_info)
                    micro_infos.append(micro_info)
            if accumulation_steps == 1:
                jax.block_until_ready(info)
            train_step_s += time.perf_counter() - train_start

            has_next_batch = step + 1 < config.num_train_steps or micro_step + 1 < accumulation_steps
            if has_next_batch:
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_wait_s += time.perf_counter() - data_wait_start

        if accumulation_steps > 1:
            apply_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                train_state, gradient_accumulator, update_info = papply_accumulated_gradients(
                    train_state, gradient_accumulator
                )
                jax.block_until_ready(update_info)
            train_step_s += time.perf_counter() - apply_start
            info = jax.tree.map(jnp.mean, common_utils.stack_forest(micro_infos))
            info.update(update_info)

        infos.append(info)
        timing_info = {
            "train_step_s": train_step_s,
            "data_wait_s": data_wait_s,
            "samples_per_s": config.effective_batch_size / train_step_s,
            "steps_per_h": 3600.0 / train_step_s,
        }
        timing_infos.append(timing_info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if timing_infos:
                reduced_info.update(
                    {key: float(np.mean([info[key] for info in timing_infos])) for key in timing_infos[0]}
                )
            time_info = _distributed.progress_time_info(
                loop_start=loop_start,
                start_step=start_step,
                step=step,
                total_steps=config.num_train_steps,
            )
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            if dist_info.is_primary:
                logging.info("Step %s: %s, %s", step, time_info, info_str)
                if show_tqdm:
                    pbar.write(f"Step {step}: {time_info}, {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
            timing_infos = []

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
