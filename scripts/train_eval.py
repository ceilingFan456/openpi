import argparse
import dataclasses
import functools
import logging
import platform
import sys
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb
from PIL import Image, ImageDraw

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class EvalRuntimeConfig:
    val_fraction: float = 0.1
    val_interval: int = 1000
    val_num_batches: int = 20
    val_seed: int = 0
    eval_use_ema: bool = True


def parse_cli() -> tuple[EvalRuntimeConfig, _config.TrainConfig]:
    defaults = EvalRuntimeConfig()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--val-interval", type=int, default=defaults.val_interval)
    parser.add_argument("--val-num-batches", type=int, default=defaults.val_num_batches)
    parser.add_argument("--val-seed", type=int, default=defaults.val_seed)
    parser.add_argument("--eval-use-ema", action=argparse.BooleanOptionalAction, default=defaults.eval_use_ema)

    eval_args, remaining = parser.parse_known_args(sys.argv[1:])
    runtime_cfg = EvalRuntimeConfig(
        val_fraction=eval_args.val_fraction,
        val_interval=eval_args.val_interval,
        val_num_batches=eval_args.val_num_batches,
        val_seed=eval_args.val_seed,
        eval_use_ema=eval_args.eval_use_ema,
    )

    if not (0.0 <= runtime_cfg.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in [0, 1).")
    if runtime_cfg.val_interval <= 0:
        raise ValueError("--val-interval must be > 0.")
    if runtime_cfg.val_num_batches <= 0:
        raise ValueError("--val-num-batches must be > 0.")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        train_cfg = _config.cli()
    finally:
        sys.argv = original_argv

    return runtime_cfg, train_cfg


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


def init_wandb(
    config: _config.TrainConfig,
    runtime_cfg: EvalRuntimeConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
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
            name=config.exp_name,
            config={
                **dataclasses.asdict(config),
                "train_eval": dataclasses.asdict(runtime_cfg),
            },
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

    # Make step semantics explicit in the UI for both train and validation metrics.
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("split/*", step_metric="global_step")


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

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
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
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

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions = batch
    eval_rng = jax.random.fold_in(rng, state.step)
    chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False)
    loss_sum = jnp.mean(chunked_loss)

    batch_shape = observation.state.shape[:-1]
    zero_gate = jnp.zeros(batch_shape, dtype=jnp.bool_)

    # Policy-only contribution (with all auxiliary supervision disabled).
    policy_only_observation = dataclasses.replace(
        observation,
        aux_keypoints_2d=None,
        aux_keypoints_mask=None,
        use_auxiliary=zero_gate,
    )
    policy_loss = jnp.mean(model.compute_loss(eval_rng, policy_only_observation, actions, train=False))

    # Auxiliary-only contribution (policy term gated off).
    aux_only_observation = dataclasses.replace(observation, use_policy=zero_gate)
    aux_loss = jnp.mean(model.compute_loss(eval_rng, aux_only_observation, actions, train=False))

    return {
        "loss": loss_sum,  # Backward-compatible alias.
        "loss_sum": loss_sum,
        "loss_policy": policy_loss,
        "loss_aux_2d": aux_loss,
    }


def create_train_val_loaders(
    config: _config.TrainConfig,
    data_sharding: jax.sharding.Sharding,
    runtime_cfg: EvalRuntimeConfig,
):
    def _unwrap_hf_dataset(dataset_obj):
        base = dataset_obj
        for _ in range(8):
            if hasattr(base, "hf_dataset"):
                return base.hf_dataset
            if hasattr(base, "_dataset"):
                base = base._dataset
                continue
            break
        return None

    def _to_bool_scalar(x) -> bool:
        arr = np.asarray(x)
        if arr.size == 0:
            return False
        return bool(arr.reshape(-1)[0])

    def _count_supervision_flags(dataset_obj, indices: np.ndarray) -> tuple[int, int]:
        hf_dataset = _unwrap_hf_dataset(dataset_obj)
        if hf_dataset is not None and len(indices) > 0:
            selected = hf_dataset.select(indices.tolist())
            aux_count = 0
            policy_count = 0
            if "use_auxiliary" in selected.column_names:
                aux_count = sum(_to_bool_scalar(v) for v in selected["use_auxiliary"])
            if "use_policy" in selected.column_names:
                policy_count = sum(_to_bool_scalar(v) for v in selected["use_policy"])
            return aux_count, policy_count

        # Fallback path for non-HF datasets.
        aux_count = 0
        policy_count = 0
        for i in indices.tolist():
            item = dataset_obj[int(i)]
            aux_count += int(_to_bool_scalar(item.get("use_auxiliary", False)))
            policy_count += int(_to_bool_scalar(item.get("use_policy", True)))
        return aux_count, policy_count

    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    dataset_size = len(dataset)
    local_batch_size = config.batch_size // jax.process_count()
    if dataset_size < 2 * local_batch_size:
        raise ValueError(
            f"Dataset too small for train/val split with batch size. dataset={dataset_size}, "
            f"local_batch_size={local_batch_size}"
        )

    val_size = int(round(dataset_size * runtime_cfg.val_fraction))
    val_size = max(val_size, local_batch_size)
    val_size = min(val_size, dataset_size - local_batch_size)

    print(f"Dataset size: {dataset_size}, Train size: {dataset_size - val_size}, Val size: {val_size}, Local batch size: {local_batch_size}")

    rng = np.random.default_rng(config.seed + runtime_cfg.val_seed)
    perm = rng.permutation(dataset_size)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_aux_enabled, train_policy_enabled = _count_supervision_flags(raw_dataset, train_indices)
    val_aux_enabled, val_policy_enabled = _count_supervision_flags(raw_dataset, val_indices)

    train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_dataset = torch.utils.data.Subset(dataset, val_indices.tolist())

    train_torch_loader = _data_loader.TorchDataLoader(
        train_dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_torch_loader = _data_loader.TorchDataLoader(
        val_dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_batches=runtime_cfg.val_num_batches,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    return (
        _data_loader.DataLoaderImpl(data_config, train_torch_loader),
        _data_loader.DataLoaderImpl(data_config, val_torch_loader),
        {
            "dataset_size": dataset_size,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "local_batch_size": local_batch_size,
            "train_aux_enabled": train_aux_enabled,
            "val_aux_enabled": val_aux_enabled,
            "train_policy_enabled": train_policy_enabled,
            "val_policy_enabled": val_policy_enabled,
        },
    )


def run_validation(
    train_state: training_utils.TrainState,
    val_loader: _data_loader.DataLoaderImpl,
    peval_step,
    mesh: jax.sharding.Mesh,
    val_rng: at.KeyArrayLike,
    *,
    step: int,
    use_ema: bool,
) -> dict[str, float]:
    eval_state = train_state
    if use_ema and train_state.ema_params is not None:
        eval_state = dataclasses.replace(train_state, params=train_state.ema_params)

    infos = []
    aux_valid_point_count = 0.0
    aux_enabled_sample_count = 0.0
    for i, batch in enumerate(val_loader):
        observation, _ = batch
        if observation.aux_keypoints_mask is not None:
            aux_valid_point_count += float(jax.device_get(jnp.sum(observation.aux_keypoints_mask.astype(jnp.float32))))
        if observation.use_auxiliary is not None:
            aux_enabled_sample_count += float(jax.device_get(jnp.sum(observation.use_auxiliary.astype(jnp.float32))))

        batch_rng = jax.random.fold_in(val_rng, step * 1_000_000 + i)
        with sharding.set_mesh(mesh):
            info = peval_step(batch_rng, eval_state, batch)
        infos.append(info)

    if not infos:
        return {
            "val/loss": float("nan"),
            "val/aux_valid_point_count": 0.0,
            "val/aux_enabled_sample_count": 0.0,
        }

    stacked_infos = common_utils.stack_forest(infos)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    out = {f"val/{k}": float(v) for k, v in reduced_info.items()}
    out["val/aux_valid_point_count"] = float(aux_valid_point_count)
    out["val/aux_enabled_sample_count"] = float(aux_enabled_sample_count)
    return out


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if -1.5 <= vmin <= 1.5 and -1.5 <= vmax <= 1.5:
            if vmin < 0.0:
                arr = (arr + 1.0) * 127.5
            else:
                arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _draw_points_overlay(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    h, w = image_rgb.shape[:2]
    for (x, y), is_valid in zip(points_xy, valid_mask, strict=False):
        if not bool(is_valid):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color, outline=color)
    return np.asarray(img)


def _letterbox_map_points(points_xy: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw

    scale = min(target_w / source_w, target_h / source_h)
    new_w = int(source_w * scale)
    new_h = int(source_h * scale)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    mapped = np.asarray(points_xy, dtype=np.float64).copy()
    mapped[:, 0] = mapped[:, 0] * scale + pad_left
    mapped[:, 1] = mapped[:, 1] * scale + pad_top
    return mapped


def build_aux_visualization_log(
    train_state: training_utils.TrainState,
    val_loader: _data_loader.DataLoaderImpl,
    *,
    step: int,
    use_ema: bool,
) -> dict[str, Any]:
    eval_state = train_state
    if use_ema and train_state.ema_params is not None:
        eval_state = dataclasses.replace(train_state, params=train_state.ema_params)

    # Choose a random validation batch and sample so overlays are not always from index 0.
    viz_rng = np.random.default_rng(step)
    val_iter = iter(val_loader)
    skip_batches = int(viz_rng.integers(0, 4))
    batch = next(val_iter)
    for _ in range(skip_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
    observation, _ = batch

    model = nnx.merge(eval_state.model_def, eval_state.params)
    model.eval()
    if not bool(getattr(model, "enable_aux_2d", False)) or not hasattr(model, "_compute_aux_predictions"):
        return {}
    if observation.aux_keypoints_2d is None:
        return {}

    pred_2d_all, _ = model._compute_aux_predictions(observation, train=False)
    batch_size = int(observation.state.shape[0])
    num_viz = min(5, max(batch_size, 1))
    sample_indices = viz_rng.choice(batch_size, size=num_viz, replace=False)

    pred_images: list[wandb.Image] = []
    gt_images: list[wandb.Image] = []

    for sample_idx in sample_indices:
        pred_2d = np.asarray(jax.device_get(pred_2d_all[sample_idx]))
        gt_2d = np.asarray(jax.device_get(observation.aux_keypoints_2d[sample_idx]))

        if observation.aux_keypoints_mask is None:
            valid_mask = np.ones((gt_2d.shape[0],), dtype=bool)
        else:
            valid_mask = np.asarray(jax.device_get(observation.aux_keypoints_mask[sample_idx])).astype(bool)

        image_keys = list(observation.images.keys())
        if observation.image_masks is not None:
            valid_image_keys = []
            for key in image_keys:
                key_mask = np.asarray(jax.device_get(observation.image_masks[key][sample_idx])).astype(bool)
                if bool(np.reshape(key_mask, (-1,))[0]):
                    valid_image_keys.append(key)
            if valid_image_keys:
                image_keys = valid_image_keys

        # Auxiliary labels in our lab dataset are projected from the base/front camera.
        # For correct visualization, prioritize the aligned camera instead of random view selection.
        if "base_0_rgb" in image_keys:
            image_key = "base_0_rgb"
        else:
            image_key = image_keys[int(viz_rng.integers(0, len(image_keys)))]
        base_image = np.asarray(jax.device_get(observation.images[image_key][sample_idx]))
        base_image = _to_uint8_rgb(base_image)

        h = min(pred_2d.shape[0], gt_2d.shape[0], valid_mask.shape[0])
        pred_2d = pred_2d[:h]
        gt_2d = gt_2d[:h]
        valid_mask = valid_mask[:h]

        target_hw = base_image.shape[:2]
        # Aux labels are generated in converter image space (H,W)=(180,320).
        source_hw = (180, 320)
        caption_suffix = ""
        if target_hw != source_hw:
            pred_2d = _letterbox_map_points(pred_2d, source_hw, target_hw)
            gt_2d = _letterbox_map_points(gt_2d, source_hw, target_hw)
            caption_suffix = (
                f" (points remapped {source_hw[1]}x{source_hw[0]} -> {target_hw[1]}x{target_hw[0]})"
            )

        pred_overlay = _draw_points_overlay(base_image, pred_2d, valid_mask, color=(255, 0, 0))
        gt_overlay = _draw_points_overlay(base_image, gt_2d, valid_mask, color=(0, 255, 0))

        pred_images.append(
            wandb.Image(
                pred_overlay,
                caption=f"step={step} sample={int(sample_idx)} predicted aux 2D points (red) on {image_key}{caption_suffix}",
            )
        )
        gt_images.append(
            wandb.Image(
                gt_overlay,
                caption=f"step={step} sample={int(sample_idx)} ground-truth aux 2D points (green) on {image_key}{caption_suffix}",
            )
        )

    return {
        "val/aux_pred_points": pred_images,
        "val/aux_gt_points": gt_images,
    }


def main(runtime_cfg: EvalRuntimeConfig, config: _config.TrainConfig):
    print("Training configuration:")
    print(config)
    print("Runtime validation config:")
    print(runtime_cfg)

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng, val_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, runtime_cfg, resuming=resuming, enabled=config.wandb_enabled)

    train_loader, val_loader, split_info = create_train_val_loaders(config, data_sharding, runtime_cfg)
    logging.info(f"Train/val split info: {split_info}")

    train_iter = iter(train_loader)
    batch = next(train_iter)
    logging.info(f"Initialized train data loader:\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"global_step": 0, "camera_views": images_to_log, **{f"split/{k}": v for k, v in split_info.items()}}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        eval_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    lr_schedule = config.lr_schedule.create()

    train_infos = []
    log_window_start = time.time()
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        train_infos.append(info)

        if step % config.log_interval == 0:
            now = time.time()
            elapsed = max(now - log_window_start, 1e-8)
            steps_in_window = max(len(train_infos), 1)
            steps_per_sec = steps_in_window / elapsed
            learning_rate = float(np.asarray(lr_schedule(step)))

            stacked_infos = common_utils.stack_forest(train_infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            info_str += f", lr={learning_rate:.2e}, steps_per_sec={steps_per_sec:.2f}"
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(
                {
                    "global_step": step,
                    **{f"train/{k}": v for k, v in reduced_info.items()},
                    "train/learning_rate": learning_rate,
                    "train/steps_per_sec": steps_per_sec,
                    # Flat aliases make it easier to find quickly in W&B.
                    "learning_rate": learning_rate,
                    "steps_per_sec": steps_per_sec,
                },
                step=step,
            )
            train_infos = []
            log_window_start = now

        do_eval = (step % runtime_cfg.val_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        if do_eval:
            val_metrics = run_validation(
                train_state,
                val_loader,
                peval_step,
                mesh,
                val_rng,
                step=step,
                use_ema=runtime_cfg.eval_use_ema,
            )
            pbar.write(
                f"Step {step}: "
                + ", ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
            )
            aux_vis = build_aux_visualization_log(
                train_state,
                val_loader,
                step=step,
                use_ema=runtime_cfg.eval_use_ema,
            )
            wandb.log({"global_step": step, **val_metrics, **aux_vis}, step=step)

        batch = next(train_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    runtime_cfg, train_cfg = parse_cli()
    main(runtime_cfg, train_cfg)
