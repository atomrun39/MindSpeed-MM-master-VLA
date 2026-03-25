import os
import warnings
import copy
import threading
import queue

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers.training_args import TrainingArguments

from megatron.training import get_args
from megatron.training.utils import is_rank0
from megatron.core import mpu

from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DataArgumentsForRewardVideo,
    DatasetAttr,
    load_tokenizer,
    align_dataset,
    SupervisedDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.data_utils.video_processor import VideoProcessor
from mindspeed_mm.data.data_utils.video_reader import VideoReader
from mindspeed_mm.data.data_utils.reward_preprocess import build_reward_data

logger = get_logger(__name__)


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank=None):


        self.dataset = dataset
        self.num_dp = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if idx % self.num_dp == self.dp_rank:
                yield item


class AsyncPreprocessIterableDataset(IterableDataset):
    def __init__(self, dataset, preprocess_fn, buffer_size=4):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn
        self.buffer_size = buffer_size
    
    def __iter__(self):
        q = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()

        def worker():
            batch = []
            for item in self.dataset:
                if stop_event.is_set():
                    break
                batch.append(item)
                if len(batch) == 1:  # batch_size=1
                    try:
                        batch_dict = {k: [v] for k, v in batch[0].items()}
                        processed = self.preprocess_fn(batch_dict)
                        for i in range(len(next(iter(processed.values())))):
                            q.put({k: v[i] for k, v in processed.items()})
                    except Exception as e:
                        raise RuntimeError("Preprocessing failed. Check input data and preprocessing function.") from e
                    batch = []
            if batch:
                batch_dict = {k: [v] for k, v in batch[0].items()}
                processed = self.preprocess_fn(batch_dict)
                for i in range(len(next(iter(processed.values())))):
                    q.put({k: v[i] for k, v in processed.items()})
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    """
    为 Qwen2VL 模型准备训练/验证数据集。
    
    整体流程：
    1. 解析参数：将外部 dict 转成内部数据类（DataArguments、ProcessorArguments、DatasetAttr）。
    2. 加载 tokenizer & processor，并依据 template 修复 tokenizer。
    3. 读取 JSON 格式的原始数据，支持流式(streaming)与非流式。
    4. 可选：从 checkpoint 断点处继续训练（跳过已消费的样本）。
    5. 将原始数据对齐成 ShareGPT 风格（align_dataset）。
    6. 根据任务类型选择处理器：
       - ranking → PairwiseDatasetProcessor（对比学习）
       - packing → PackedSupervisedDatasetProcessor（长度打包）
       - 其他 → SupervisedDatasetProcessor（普通 SFT）
    7. 对数据集进行 tokenize / preprocess：
       - streaming=True 时，可启用异步预处理（AsyncPreprocessIterableDataset）提升 IO 效率，
         否则用 `.map()` 在线处理；最后统一包一层 DistributedIterableDataset 做分布式采样。
       - streaming=False 时，可选择“即时 transform”或一次性 `.map()` 处理；
         多进程/缓存策略通过 LOCAL_RANK 控制，仅主进程真正计算，其余进程直接读缓存。
    8. 若提供了验证集，同样走一遍上述流程，最后返回 (train_dataset, val_dataset)；
       否则只返回 train_dataset。
    9. 仅在 rank0 打印一条样本示例，方便调试。
    
    参数
    ----
    basic_param : dict
        数据集基础配置，例如 {"dataset": "xxx.json", "max_samples": 10000, "template": "qwen2vl", ...}
    preprocess_param : dict
        预处理配置，会传给 ProcessorArguments，例如 {"image_resolution": 448, "video_fps": 2, ...}
    dataset_param : dict
        数据集属性配置，必须包含 "attr" 字段，例如 {"attr": {"ranking": False, "packing": True}}
    
    返回
    ----
    train_dataset : Dataset/IterableDataset
        处理好的训练集（已 tokenize、对齐、分布式包装）。
    val_dataset : Dataset/IterableDataset, optional
        处理好的验证集；当 data_args.val_dataset 不存在时返回 None。
    """
    
    # 1. 参数校验与转换
    # 兼容旧接口：若仍使用 cutoff_len 则抛错提示改用 seq_length
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    
    # 将外部 dict 转换为内部数据类，方便后续字段提示与类型检查
    data_args = DataArguments(**basic_param)
    # 用 megatron 全局参数中的 seq_length 覆盖真正用于截断的长度
    data_args.cutoff_len = get_args().seq_length
    
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    # 2. 加载 tokenizer、processor 并修复 template
    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    # 根据 template 对 tokenizer 做特殊补丁（如添加 special tokens）
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    # 3. 读取已消费样本数，用于断点续训
    args = get_args()
    consumed_samples = args.consumed_train_samples

    # 4. 仅在主进程真正执行数据计算，其余进程复用缓存，减少冗余
    # 与 LLaMA-Factory 策略保持一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # 4.1 加载训练集（JSON 格式）
        train_dataset = load_dataset(
            path="json",
            data_files=data_args.dataset,
            split="train",
            cache_dir=data_args.cache_dir,
            streaming=data_args.streaming
        )
        # 若指定最大样本数且非流式，则直接 select 前 max_samples 条
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))
        
        # 断点续训：跳过已消费的样本
        if consumed_samples > 0:
            logger.info(f"Skipping first {consumed_samples} samples to resume from checkpoint.")
            train_dataset.skip(consumed_samples)

        # 4.2 加载验证集（若提供）
        val_dataset = None
        if data_args.val_dataset:
            val_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming
            )
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            # 若同时给出 val_dataset 与 val_rate，优先使用前者并警告
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. "
                    "The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning
                )

        # 4.3 构造 map 时的通用参数
        # 流式数据无需多进程，非流式可通过 num_proc 加速；
        # load_from_cache_file 控制缓存：主进程计算，其余进程直接读缓存
        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug('Rank: %s, kwargs: %s', local_process_index, kwargs)

        # 5. 将原始数据转换成 ShareGPT 风格格式
        train_dataset = align_dataset(train_dataset, dataset_attr, data_args)
        if val_dataset:
            val_dataset = align_dataset(val_dataset, dataset_attr, data_args)

        # 6. 根据任务类型选择对应的 DatasetProcessor
        if dataset_attr.ranking:
            # 对比学习（如 DPO / Reward Model）
            dataset_processor_cls = PairwiseDatasetProcessor
        elif dataset_attr.packing:
            # 打包多个样本到同一序列，节省 padding
            data_args.cutoff_len -= 1  # 预留一个位置给打包分隔符
            dataset_processor_cls = PackedSupervisedDatasetProcessor
        else:
            # 普通监督微调
            dataset_processor_cls = SupervisedDatasetProcessor
        
        dataset_processor = dataset_processor_cls(
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args
        )
        preprocess_func = dataset_processor.preprocess_dataset

        # 7. 执行 tokenize / 预处理
        if data_args.streaming:
            # 7.1 流式场景
            if data_args.async_preprocess:
                # 异步预处理：后台线程实时处理，主线程迭代消费
                train_dataset = DistributedIterableDataset(train_dataset)          # 先按数据并行 rank 分片
                train_dataset = AsyncPreprocessIterableDataset(
                    train_dataset, preprocess_func, buffer_size=8
                )
            else:
                # 在线 map：批处理完后，再包一层分布式迭代器
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=list(next(iter(train_dataset)).keys()),
                    **kwargs,
                )
                train_dataset = DistributedIterableDataset(train_dataset)
            
            # 验证集同理
            if val_dataset:
                if data_args.async_preprocess:
                    val_dataset = DistributedIterableDataset(val_dataset)
                    val_dataset = AsyncPreprocessIterableDataset(
                        val_dataset, preprocess_func, buffer_size=8
                    )
                else:
                    val_dataset = val_dataset.map(
                        preprocess_func,
                        batched=True,
                        batch_size=data_args.preprocessing_batch_size,
                        remove_columns=list(next(iter(val_dataset)).keys()),
                        **kwargs,
                    )
                    val_dataset = DistributedIterableDataset(val_dataset)
                return train_dataset, val_dataset
        else:
            # 7.2 非流式场景
            if data_args.preprocess_on_fly:
                # 即时 transform：每次迭代时实时调用 preprocess_func
                train_dataset.set_transform(preprocess_func, output_all_columns=True)
            else:
                # 一次性 map 处理
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=list(next(iter(train_dataset)).keys()),
                    desc=f"Rank {local_process_index}, running tokenizer on train_dataset",
                    **kwargs,
                )
            
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=list(next(iter(val_dataset)).keys()),
                    desc=f"Rank {local_process_index}, running tokenizer on val_dataset",
                    **kwargs,
                )
                return train_dataset, val_dataset

        # 8. 打印样本示例（仅 rank0）
        if is_rank0():
            print("training example:")
            dataset_processor.print_data_example(next(iter(train_dataset)))
        
        # 若未提供验证集，则只返回训练集
        return train_dataset


def process_reward_dataset(dataset, data_folder, preprocess_param):
    def add_idx(example, idx):
        example['metainfo_idx'] = idx
        return example

    dataset = dataset.map(lambda example, idx: add_idx(example, idx), with_indices=True)
    if not preprocess_param.get('use_tied_data', True):
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in dataset.eval_dim)
        dataset = dataset.filter(filter_func)

    convert_func = lambda example: build_reward_data(example, data_folder, **preprocess_param)
    dataset = dataset.map(convert_func, remove_columns=dataset.column_names)
    return dataset


def get_reward_video_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")

    data_args = DataArgumentsForRewardVideo(**basic_param)

    # Ensure main process handles data processing, while other processes reuse cache to avoid redundant calculations.
    # This strategy is consistent with the data processing strategy used by LLaMA Factory.
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # load dataset from file
        train_dataset = load_dataset(path="csv", data_files=data_args.data_path, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))
        
        train_dataset = process_reward_dataset(train_dataset, data_args.data_folder, preprocess_param)
        
        val_dataset = None
        if data_args.val_dataset:
            if data_args.data_path_val:
                val_dataset = load_dataset(path="csv", data_files=data_args.data_path_val, split="train",
                                        cache_dir=data_args.cache_dir,
                                        streaming=data_args.streaming
                )
                val_dataset = process_reward_dataset(val_dataset, data_args.data_folder, preprocess_param)
                train_dataset = train_dataset
            else:
                dataset = train_dataset.train_test_split(test_size=0.02, seed=42)
                train_dataset = dataset['train']
                val_dataset = dataset['test']
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)
        else:
            train_dataset = train_dataset

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # If overwrite_cache is false (default), only non-rank-0 nodes load cache without map processing.
                # If overwrite_cache is true, all nodes read the cache and none of them perform map processing.
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)

        if data_args.streaming:
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = DistributedIterableDataset(val_dataset)

        return [train_dataset, val_dataset]