import argparse
import importlib
import os
from typing import Any, Dict

import torch
from torch.utils.data import IterableDataset
from transformers import AutoProcessor


class RLDSVLAAdapterDataset(IterableDataset):
    def __init__(self, basic_param: Dict[str, Any], preprocess_param: Dict[str, Any], **dataset_param):
        super().__init__()

        model_name_or_path = preprocess_param.get("model_name_or_path")
        if model_name_or_path is None:
            raise ValueError("preprocess_parameters.model_name_or_path is required for rlds_vla dataset.")

        processor = AutoProcessor.from_pretrained(model_name_or_path)
        processor.tokenizer.padding_side = "left"

        use_wrist_image = basic_param.get("use_wrist_image", False)
        use_proprio = basic_param.get("use_proprio", True)
        resize_resolution = tuple(basic_param.get("resize_resolution", [224, 224]))
        shuffle_buffer_size = basic_param.get("shuffle_buffer_size", 10000)
        image_aug = basic_param.get("image_aug", False)
        window_size = basic_param.get("window_size", 1)
        train = basic_param.get("train", True)
        data_root_dir = basic_param.get("data_root_dir")
        data_mix = basic_param.get("data_mix")
        env_data_root_dir = os.getenv("VLA_DATA_ROOT_DIR") or os.getenv("OXE_DATA_ROOT")
        env_data_mix = os.getenv("VLA_DATA_MIX") or os.getenv("DATA_MIX")
        if env_data_root_dir:
            data_root_dir = env_data_root_dir
        if env_data_mix:
            data_mix = env_data_mix
        if data_root_dir is None or data_mix is None:
            raise ValueError("basic_parameters.data_root_dir and basic_parameters.data_mix are required for rlds_vla dataset.")

        batch_transform_type = basic_param.get("batch_transform_type", "qwen2vl")
        batch_transform_kwargs = basic_param.get("batch_transform_kwargs", {})
        batch_transform = build_batch_transform(
            batch_transform_type,
            processor,
            use_wrist_image=use_wrist_image,
            use_proprio=use_proprio,
            **batch_transform_kwargs,
        )

        core_rlds_dataset_cls, _ = _load_rlds_core_classes()
        self.dataset = core_rlds_dataset_cls(
            data_root_dir=data_root_dir,
            data_mix=data_mix,
            batch_transform=batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
            window_size=window_size,
        )

    def __iter__(self):
        for sample in self.dataset:
            action = sample.pop("actions", None)
            if action is not None:
                sample["action"] = action
            state = sample.pop("proprio", None)
            if state is not None:
                sample["state"] = state
            yield sample


def get_rlds_vla_dataset(basic_param, preprocess_param, dataset_param):
    if not isinstance(basic_param, dict):
        basic_param = basic_param.to_dict()
    if not isinstance(preprocess_param, dict):
        preprocess_param = preprocess_param.to_dict()
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    return RLDSVLAAdapterDataset(basic_param, preprocess_param, **dataset_param)


_BATCH_TRANSFORM_BUILDERS = {}


def register_batch_transform_builder(name: str, builder):
    _BATCH_TRANSFORM_BUILDERS[name] = builder


def build_batch_transform(name: str, processor, **kwargs):
    if name not in _BATCH_TRANSFORM_BUILDERS:
        raise ValueError(f"Unknown batch_transform_type `{name}`. Available: {list(_BATCH_TRANSFORM_BUILDERS.keys())}")
    return _BATCH_TRANSFORM_BUILDERS[name](processor=processor, **kwargs)


def _build_qwen2vl_batch_transform(processor, use_wrist_image=False, use_proprio=True, **kwargs):
    _, core_rlds_batch_transform_cls = _load_rlds_core_classes()
    return core_rlds_batch_transform_cls(
        processor=processor,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
    )


register_batch_transform_builder("qwen2vl", _build_qwen2vl_batch_transform)


_RLDS_CORE_CACHE = None


def _load_rlds_core_classes():
    global _RLDS_CORE_CACHE
    if _RLDS_CORE_CACHE is not None:
        return _RLDS_CORE_CACHE
    _validate_tensorflow_runtime()
    module = importlib.import_module("mindspeed_mm.data.rlds_dataloader.datasets.datasets")
    _RLDS_CORE_CACHE = (module.RLDSDataset, module.RLDSBatchTransform)
    return _RLDS_CORE_CACHE


def _validate_tensorflow_runtime():
    tf = importlib.import_module("tensorflow")
    if not hasattr(tf, "image"):
        tf_path = getattr(tf, "__file__", "unknown")
        tf_ver = getattr(tf, "__version__", "unknown")
        raise RuntimeError(
            "Invalid TensorFlow runtime detected for RLDS pipeline. "
            f"module=tensorflow, version={tf_ver}, path={tf_path}, has_tf_image=False. "
            "Please install a full TensorFlow build (e.g. tensorflow-cpu/tensorflow >= 2.x), "
            "and make sure no local module shadows `tensorflow`."
        )


class _DictWrapper:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def to_dict(self):
        return self._data


class _DatasetParamWrapper:
    def __init__(self, preprocess_parameters: Dict[str, Any]):
        self.preprocess_parameters = _DictWrapper(preprocess_parameters)


def _print_batch_summary(batch: Dict[str, Any]):
    summary = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
            }
        elif v is None:
            summary[k] = None
        else:
            summary[k] = {"type": str(type(v))}
    print(summary)


def _build_debug_dataloader(args):
    basic_param = {
        "data_root_dir": args.data_root_dir,
        "data_mix": args.data_mix,
        "use_wrist_image": args.use_wrist_image,
        "use_proprio": args.use_proprio,
        "resize_resolution": [args.resize_h, args.resize_w],
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "image_aug": args.image_aug,
        "window_size": args.window_size,
        "train": args.train,
        "batch_transform_type": args.batch_transform_type,
    }
    preprocess_param = {
        "model_name_or_path": args.model_name_or_path,
    }
    dataset = RLDSVLAAdapterDataset(
        basic_param,
        preprocess_param,
    )
    from mindspeed_mm.data.dataloader.data_collator import DataCollatorForVLASequence

    collator = DataCollatorForVLASequence(dataset_param=_DatasetParamWrapper(preprocess_param))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--data_mix", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--use_wrist_image", action="store_true")
    parser.add_argument("--use_proprio", action="store_true")
    parser.add_argument("--image_aug", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_transform_type", type=str, default="qwen2vl")
    return parser.parse_args()


def _run_debug():
    args = _parse_args()
    dataloader = _build_debug_dataloader(args)
    for idx, batch in enumerate(dataloader):
        _print_batch_summary(batch)
        if idx + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    _run_debug()
