# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy
from functools import partial
from typing import Dict, Any

from datasets import Dataset
import torch

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.vlm_model import VLMModel
from mindspeed_mm.patchs import dummy_optimizer_patch
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.utils.hetero_parallel import change_parallel_state, apply_hetero_parallel_hooks
from mindspeed_mm.utils.utils import EncoderBalanceComm
from mindspeed_mm.utils.hetero_parallel import hetero_align_config
from mindspeed_mm.utils.utils import compute_token_level_loss
mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor

_VLA_CONFIG_VALIDATED = False
_VLA_BATCH_CONTRACT_VALIDATED = False


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _unwrap_model(model):
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def _is_vla_dataset(args):
    dataset_param = _cfg_get(_cfg_get(args.mm, "data", None), "dataset_param", None)
    dataset_type = str(_cfg_get(dataset_param, "dataset_type", "")).lower()
    return dataset_type == "rlds_vla"


def _validate_vla_config(args):
    global _VLA_CONFIG_VALIDATED
    if _VLA_CONFIG_VALIDATED or not _is_vla_dataset(args):
        return
    model_cfg = _cfg_get(args.mm, "model", None)
    action_head_cfg = _cfg_get(model_cfg, "action_head", None)
    if action_head_cfg is None:
        raise ValueError("VLA training requires mm_model.action_head config when dataset_type is `rlds_vla`.")
    if not bool(_cfg_get(action_head_cfg, "enable", False)):
        raise ValueError("VLA training requires mm_model.action_head.enable=true when dataset_type is `rlds_vla`.")
    action_dim = int(_cfg_get(action_head_cfg, "action_dim", 0) or 0)
    action_horizon = int(_cfg_get(action_head_cfg, "action_horizon", _cfg_get(action_head_cfg, "num_queries", 0)) or 0)
    if action_dim <= 0 or action_horizon <= 0:
        raise ValueError("VLA training requires positive mm_model.action_head.action_dim and action_horizon/num_queries.")
    repeated_steps = int(_cfg_get(action_head_cfg, "repeated_diffusion_steps", 1) or 1)
    if repeated_steps <= 0:
        raise ValueError("mm_model.action_head.repeated_diffusion_steps must be >= 1.")
    data_cfg = _cfg_get(args.mm, "data", None)
    collate_param = _cfg_get(_cfg_get(data_cfg, "dataloader_param", None), "collate_param", None)
    collate_name = str(_cfg_get(collate_param, "model_name", ""))
    if collate_name not in {"vla_seq", "qwen2vl_vla"}:
        raise ValueError("VLA training requires dataloader_param.collate_param.model_name in {`vla_seq`, `qwen2vl_vla`}.")
    basic_param = _cfg_get(_cfg_get(data_cfg, "dataset_param", None), "basic_parameters", None)
    use_proprio = bool(_cfg_get(basic_param, "use_proprio", True))
    state_dim = int(_cfg_get(action_head_cfg, "state_dim", 0) or 0)
    if use_proprio and state_dim <= 0:
        raise ValueError("VLA training with use_proprio=true requires mm_model.action_head.state_dim > 0.")
    _VLA_CONFIG_VALIDATED = True


def _validate_vla_batch_contract(batch, model):
    global _VLA_BATCH_CONTRACT_VALIDATED
    if _VLA_BATCH_CONTRACT_VALIDATED:
        return
    core_model = _unwrap_model(model)
    if not bool(getattr(core_model, "post_process", False)):
        return
    if not bool(getattr(core_model, "enable_action_head", False)):
        raise ValueError("VLA training requires action_head enabled in model config.")
    action_head = getattr(core_model, "action_head", None)
    if action_head is None:
        raise ValueError("VLA training requires action_head module initialized on model.")
    if "action" not in batch or batch["action"] is None:
        raise ValueError("VLA batch must contain non-empty `action` tensor.")
    action_tensor = batch["action"]
    if action_tensor.ndim != 3:
        raise ValueError(f"VLA `action` tensor must be 3D [B,T,D], got shape={tuple(action_tensor.shape)}")
    expected_horizon = int(getattr(action_head, "action_horizon", 0) or 0)
    expected_dim = int(getattr(action_head, "action_dim", 0) or 0)
    if expected_horizon > 0 and action_tensor.shape[1] != expected_horizon:
        raise ValueError(
            f"VLA action_horizon mismatch: batch={action_tensor.shape[1]}, action_head={expected_horizon}"
        )
    if expected_dim > 0 and action_tensor.shape[2] != expected_dim:
        raise ValueError(
            f"VLA action_dim mismatch: batch={action_tensor.shape[2]}, action_head={expected_dim}"
        )
    action_head_cfg = getattr(core_model, "action_head_config", None)
    expected_state_dim = int(_cfg_get(action_head_cfg, "state_dim", 0) or 0)
    state_tensor = batch.get("state", None)
    if expected_state_dim > 0 and state_tensor is None:
        raise ValueError("VLA batch must contain `state` because action_head.state_dim > 0.")
    if state_tensor is not None:
        if state_tensor.ndim not in {2, 3}:
            raise ValueError(f"VLA `state` tensor must be 2D/3D, got shape={tuple(state_tensor.shape)}")
        state_last_dim = int(state_tensor.shape[-1])
        if expected_state_dim > 0 and state_last_dim != expected_state_dim:
            raise ValueError(
                f"VLA state_dim mismatch: batch={state_last_dim}, action_head={expected_state_dim}"
            )
    _VLA_BATCH_CONTRACT_VALIDATED = True


def model_provider(pre_process=True, post_process=True, modules=None):
    """Builds the model."""
    if modules is None:
        modules = ['image_encoder', 'audio_encoder', 'text_decoder']

    args = get_args()
    _validate_vla_config(args)
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    _configure_modules(vlm_config, modules)

    model = VLMModel(vlm_config)

    if args.hetero_parallel:
        print_rank_0("apply hetero parallel ...")
        apply_hetero_parallel_hooks(model)

    _apply_freezing(model, vlm_config)

    return model


def _configure_modules(vlm_config, modules):
    """Configure each module based on the modules list."""
    module_configs = {
        'image_encoder': _configure_image_encoder,
        'audio_encoder': _configure_audio_encoder,
        'text_decoder': _configure_text_decoder
    }

    for module_name, config_func in module_configs.items():
        if module_name in modules and hasattr(vlm_config, module_name):
            config_func(vlm_config)
        else:
            setattr(vlm_config, module_name, None)


def _configure_image_encoder(vlm_config):
    """Configure image encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.image_encoder.vision_encoder, vlm_config.image_encoder)
        hetero_align_config(vlm_config.image_encoder.vision_projector, vlm_config.image_encoder)

    # MindSpeed needs to validate the CP configuration; the attention head must be divisible by the CP sizes.
    # However, since the vision projector does not have an attention head, special handling is required.
    vlm_config.image_encoder.vision_projector.context_parallel_size = 1
    vlm_config.image_encoder.vision_encoder.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_projector.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)


def _configure_audio_encoder(vlm_config):
    """Configure audio encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.audio_encoder.audio_encoder, vlm_config.audio_encoder)

    vlm_config.audio_encoder.audio_encoder = get_model_config(vlm_config.audio_encoder.audio_encoder)


def _configure_text_decoder(vlm_config):
    """Configure text decoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.text_decoder, vlm_config.text_decoder)
        
    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)


def _apply_freezing(model, vlm_config):
    """Apply freezing settings to the model."""
    has_image = hasattr(vlm_config, 'image_encoder') and vlm_config.image_encoder is not None
    freeze_image_encoder = has_image and getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True)
    freeze_image_projection = has_image and getattr(vlm_config.image_encoder.vision_projector, 'freeze', False)

    has_audio = hasattr(vlm_config, 'audio_encoder') and vlm_config.audio_encoder is not None
    freeze_audio_encoder = has_audio and getattr(vlm_config.audio_encoder.audio_encoder, 'freeze', True)

    model.freeze(
        freeze_image_encoder=freeze_image_encoder,
        freeze_image_projection=freeze_image_projection,
        freeze_audio_encoder=freeze_audio_encoder
    )


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
    if has_video:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None
    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def loss_func(output_tensor):
    """
    计算并返回最终用于反向传播的 loss，同时收集日志所需的统计信息。
    
    支持两种模式：
    1. 逐 token loss（args.calculate_per_token_loss=True）：
       - 调用 compute_token_level_loss 计算每个 token 的 loss，返回加权后的总 loss、
         本地有效 token 数以及用于日志的 (numerator, denominator) 二元组。
    2. 默认模式（args.calculate_per_token_loss=False）：
       - 直接取 loss_dict['loss']，在数据并行组内做 all-reduce 平均，
         再按上下文并行度做归一化，保证各并行组梯度一致。
    
    额外功能：
    - 若开启 args.log_tps，则计算并记录“每个样本的平均 token 数”。
    
    返回值：
    - 逐 token 模式：(loss_tensor, local_num_tokens, loss_dir)
    - 默认模式：(loss_tensor, loss_dir)
      其中 loss_dir 为字典，用于训练日志打印。
    """
    args = get_args()
    loss_dir = {}

    action_loss = output_tensor.get("action_loss", None)
    if action_loss is not None:
        averaged_action_loss = average_losses_across_data_parallel_group([action_loss])[0]
        loss_dir["loss"] = averaged_action_loss
        loss_dir["action_loss"] = averaged_action_loss

        loss_dict = output_tensor.get("loss_dict", None)
        if isinstance(loss_dict, dict) and "loss" in loss_dict:
            averaged_lm_loss = average_losses_across_data_parallel_group([loss_dict["loss"]])[0]
            loss_dir["lm_loss"] = averaged_lm_loss

        action_loss = action_loss.unsqueeze(0).clone() / mpu.get_context_parallel_world_size()
        if args.calculate_per_token_loss:
            local_num_tokens = torch.ones((), dtype=torch.long, device=action_loss.device)
            loss_dir["loss"] = (averaged_action_loss, torch.ones_like(averaged_action_loss))
            return action_loss[0].clone(), local_num_tokens, loss_dir
        return action_loss, loss_dir

    if "loss_dict" not in output_tensor:
        raise ValueError(
            "output_tensor must include `action_loss` or `loss_dict` for training loss computation."
        )
    loss_dict = output_tensor["loss_dict"]

    # 如果需要记录“每样本 token 数”，则计算并写入日志字典
    if args.log_tps:
        tokens_per_sample = get_tps(output_tensor['logits'])
        loss_dir["tokens per sample"] = tokens_per_sample

    # 逐 token loss 模式：返回加权 loss、本地 token 数及日志用的 (numerator, denominator)
    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(loss_dict)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),          # 用于反向传播的标量 loss
            local_num_tokens,        # 本地实际参与计算的 token 数量
            loss_dir                  # 日志信息
        )

    # 默认模式：对 loss 做 DP 平均，再按 CP 规模归一化
    loss = loss_dict['loss']
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    # 除以上下文并行世界大小，保证各 CP 组梯度一致
    return loss / mpu.get_context_parallel_world_size(), loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    _validate_vla_config(args)
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    batch = get_batch(data_iterator, is_vit_last_stage)
    if _is_vla_dataset(args):
        batch["compute_action_loss"] = True
        _validate_vla_batch_contract(batch, model)
    output_tensor = model(**batch)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    if args.hetero_parallel:
        print_rank_0("change parallel state for data loader ...")
        change_parallel_state("text_decoder")

        if args.hetero_encoder_mbs_scale > 1:
            pp_mbs = args.micro_batch_size
            args.micro_batch_size = pp_mbs * args.hetero_encoder_mbs_scale

    datasets = build_mm_dataset(data_config.dataset_param)
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )

    micro_batch_size = args.micro_batch_size
    if args.use_data_balance:
        global_batch_size = args.micro_batch_size * get_num_microbatches()
        if args.hetero_encoder_mbs_scale > 1:
            global_batch_size = global_batch_size // args.hetero_encoder_mbs_scale
        args.micro_batch_size = global_batch_size

    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        args.micro_batch_size = micro_batch_size
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            valid_dataloader = build_dataloader(valid_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    if args.hetero_parallel and args.hetero_encoder_mbs_scale > 1:
        args.micro_batch_size = pp_mbs

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import ring_attn_patch, ulysses_patches, torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
