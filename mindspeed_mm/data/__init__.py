__all__ = [
    "build_mm_dataset", "build_mm_dataloader"
]

import copy

from torch.utils.data import ConcatDataset
from torch.distributed.distributed_c10d import _get_default_group

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from mindspeed_mm.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
    prepare_variable_dataloader,
)
from mindspeed_mm.data.datasets.multimodal_dataset import DeepSeekVLDataset, MultiModalChatDataset
from mindspeed_mm.data.datasets.t2i_dataset import T2IDataset
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset, DynamicVideoTextDataset
from mindspeed_mm.data.datasets.i2v_dataset import I2VDataset
from mindspeed_mm.data.datasets.feature_dataset import FeatureDataset
from mindspeed_mm.data.datasets.audio_dataset import AudioDataset
from mindspeed_mm.data.datasets.qwen2vl_dataset import get_qwen2vl_dataset, get_reward_video_dataset
from mindspeed_mm.data.datasets.rlds_vla_dataset import get_rlds_vla_dataset
from mindspeed_mm.data.datasets.ae_dataset import TrainVideoDataset
from mindspeed_mm.models.ae.training.global_vars import get_ae_args



def build_mm_dataset(dataset_param):
    """
    根据任务类型构建多模态数据集。

    本函数是“工厂函数”——根据 dataset_param 中指定的 dataset_type，动态地选择并实例化对应的数据集类。
    支持以下任务类型：
        t2v  : Text-to-Video 文生视频
        i2v  : Image-to-Video 图生视频
        t2i  : Text-to-Image 文生图
        dt2v : Dynamic-resolution Text-to-Video 动态分辨率文生视频
        feature     : 纯特征数据集
        multimodal  : 多轮对话多模态数据集（支持多子集合并）
        audio       : 音频数据集
        huggingface : HuggingFace 格式（内部调用 get_qwen2vl_dataset）
        deepseekvl2 : DeepSeekVL2 专用数据集（支持多子集合并）
        rewardvideo : 奖励模型训练用的视频数据集
        lumina      : Lumina 模型对话数据集
        bagel       : Bagel 多模态数据集

    Args:
        dataset_param (dict|object): 多模态数据集配置，必须包含以下核心键：
            - dataset_type        : 字符串，指定上述任务类型之一
            - basic_parameters    : dict 或 list，数据集基础参数
            - preprocess_parameters: dict，预处理超参

    Returns:
        dataset: 与 dataset_type 对应的数据集实例（单个 Dataset 或 ConcatDataset）

    Raises:
        AssertionError: 缺少任意核心键时触发
        NotImplementedError: 遇到不支持的 dataset_type 时触发
    """
    # 1. 统一转成 dict，方便后续处理
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()

    # 2. 必填字段检查：三缺一就直接抛错，提前失败
    for check_key in ["dataset_type", "basic_parameters", "preprocess_parameters"]:
        if check_key not in dataset_param:
            raise AssertionError(f"Key parameter missing: {check_key}")

    # 3. 提取公共字段，减少重复索引
    dataset_type = dataset_param["dataset_type"]      # 任务类型
    basic_param = dataset_param["basic_parameters"]   # 基础参数
    preprocess_param = dataset_param["preprocess_parameters"]  # 预处理参数

    # 4. 根据任务类型分发到具体数据集类
    # 4.1 文生视频
    if dataset_type == "t2v":
        return T2VDataset(basic_param, preprocess_param, **dataset_param)

    # 4.2 图生视频
    elif dataset_type == "i2v":
        return I2VDataset(basic_param, preprocess_param, **dataset_param)

    # 4.3 文生图
    elif dataset_type == "t2i":
        return T2IDataset(basic_param, preprocess_param, **dataset_param)

    # 4.4 动态分辨率文生视频
    elif dataset_type == "dt2v":
        return DynamicVideoTextDataset(basic_param, preprocess_param, **dataset_param)

    # 4.5 纯特征数据集（无需预处理参数）
    elif dataset_type == "feature":
        return FeatureDataset(basic_param)

    # 4.6 多轮对话多模态数据集
    # 支持传入多个子配置，每个子配置可单独设置 repeat_time，最终用 ConcatDataset 合并
    elif dataset_type == "multimodal":
        # 保证 basic_param 是 list，方便统一遍历
        if not isinstance(basic_param, list):
            basic_param = [basic_param]

        datasets = []
        for single_param in basic_param:
            # 如果子配置里写了 repeat_time 就用它，否则默认 1 次
            dataset_param["repeat_time"] = single_param.get("repeat_time", 1)
            # 深拷贝一份，防止不同子集之间互相污染
            dataset_param_copy = copy.deepcopy(dataset_param)
            dataset = MultiModalChatDataset(single_param, preprocess_param, **dataset_param_copy)
            datasets.append(dataset)
        # 将所有子集顺序拼接，训练时等价于一个大数据集
        return ConcatDataset(datasets)

    # 4.7 音频数据集
    elif dataset_type == "audio":
        return AudioDataset(basic_param, preprocess_param, **dataset_param)

    # 4.8 HuggingFace 格式（内部会读取 qwen2vl 专用逻辑）
    elif dataset_type == "huggingface":
        return get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param)

    # 4.8.1 RLDS VLA 格式（桥接 UnifoLM-VLA 的 RLDS pipeline）
    elif dataset_type == "rlds_vla":
        return get_rlds_vla_dataset(basic_param, preprocess_param, dataset_param)

    # 4.9 DeepSeekVL2 专用数据集，同样支持多子集合并
    elif dataset_type == "deepseekvl2":
        if not isinstance(basic_param, list):
            basic_param = [basic_param]

        datasets = []
        for single_param in basic_param:
            dataset_param["repeat_time"] = single_param.get("repeat_time", 1)
            dataset_param_copy = copy.deepcopy(dataset_param)
            dataset = DeepSeekVLDataset(single_param, **dataset_param_copy)
            datasets.append(dataset)
        return ConcatDataset(datasets)

    # 4.10 奖励模型训练用的视频数据集
    elif dataset_type == "rewardvideo":
        return get_reward_video_dataset(basic_param, preprocess_param, dataset_param)

    # 4.11 Lumina 模型对话数据集
    if dataset_type == "lumina":
        # 延迟导入，避免无关依赖
        from mindspeed_mm.data.datasets.lumina_dataset import LuminaConversationDataset
        return LuminaConversationDataset(basic_param, **dataset_param)

    # 4.12 Bagel 多模态数据集
    elif dataset_type == "bagel":
        from mindspeed_mm.data.datasets.bagel_dataset import BagelMultiDataset
        return BagelMultiDataset(basic_param, preprocess_param, **dataset_param)

    # 5. 走到这里说明 dataset_type 没匹配上，直接抛错
    else:
        raise NotImplementedError(dataset_type)


def build_mm_dataloader(dataset, dataloader_param, process_group=None, consumed_samples=0, dataset_param=None, generator=None):
    """
    根据任务类型构建多模态 DataLoader。

    本函数是“工厂函数”——根据 dataloader_param 中指定的 dataloader_mode，动态地选择并返回对应的 DataLoader。
    支持三种模式：
        base    : 最基础的 PyTorch DataLoader，无分布式采样器
        sampler : 为分布式训练定制的 DataLoader，内部会构造 Megatron 专用采样器
        variable: 支持“动态样本数”场景，通常用于变长视频/图文混合训练

    Args:
        dataset (Dataset): 已经构建好的多模态数据集对象
        dataloader_param (dict|object): DataLoader 配置，必须包含 dataloader_mode 字段
        process_group (ProcessGroup, optional): 分布式进程组；若为 None，则自动取 Megatron 的数据并行组
        consumed_samples (int, optional): 已经消费过的样本数，用于断点续训；默认 0 表示从头开始
        dataset_param (dict, optional): 数据集的原始配置，透传给底层 prepare 函数，用于日志或调试
        generator (torch.Generator, optional): 可复现的随机数生成器，主要供采样器使用

    Returns:
        DataLoader: 与 dataloader_mode 匹配的 DataLoader 实例

    Raises:
        AssertionError: dataloader_param 中缺少 dataloader_mode 时触发
        NotImplementedError: 遇到不支持的 dataloader_mode 时触发
    """
    # 1. 统一转成 dict，方便后续处理
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()

    # 2. 必填字段检查：缺少 dataloader_mode 直接抛错，提前失败
    if "dataloader_mode" not in dataloader_param:
        raise AssertionError("Key parameter missing: dataloader_mode")

    # 3. 取出模式并“弹出”，避免透传时重复
    dataloader_mode = dataloader_param.pop("dataloader_mode")

    # 4. 若未显式指定进程组，则默认使用 Megatron 的数据并行组
    if process_group is None:
        process_group = mpu.get_data_parallel_group()

    # 5. 从全局参数里统一读取 batch_size / num_workers / seed，保证与训练主流程一致
    args = get_args()
    dataloader_param.update(
        {
            "batch_size": args.micro_batch_size,  # 全局微批次大小
            "num_workers": args.num_workers,      # 多进程加载线程数
            "seed": args.seed,                    # 全局随机种子，保证可复现
        }
    )
    # 6. 打印提示：当前使用命令行参数覆盖 data.json 中的同名字段
    print_rank_0('[INFO] initialize `batch_size`/`num_workers`/`seed` from argument parser rather than `data.json`')

    # 7. 根据模式分发到具体 DataLoader 构建函数
    if dataloader_mode == "base":
        # 7.1 基础模式：直接封装成最朴素的 DataLoader
        data_loader = prepare_base_dataloader(dataset, **dataloader_param, dataset_param=dataset_param)
        return data_loader

    elif dataloader_mode == "sampler":
        # 7.2 采样器模式：构造带分布式采样器的 DataLoader，支持断点续训与随机种子
        data_loader = prepare_sampler_dataloader(
            dataset,
            **dataloader_param,
            process_group=process_group,
            consumed_samples=consumed_samples,
            dataset_param=dataset_param,
            generator=generator
        )
        return data_loader

    elif dataloader_mode == "variable":
        # 7.3 可变样本模式：用于变长视频或动态样本数场景，内部会做额外的样本数对齐
        data_loader = prepare_variable_dataloader(
            dataset,
            **dataloader_param,
            process_group=process_group,
            consumed_samples=consumed_samples
        )
        return data_loader

    else:
        # 8. 走到这里说明 dataloader_mode 没匹配上，直接抛错
        raise NotImplementedError(dataloader_mode)


def build_ae_dataset(dataset_param):
    """
    Build an AE dataset based on different tasks.

    Args:
        dataset_param: config with necessary parameters for AE dataset construction
    Return:
        dataset: an AE training dataset object
    """
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    return TrainVideoDataset(**dataset_param)


def build_ae_dataloader(dataset, dataloader_param, process_group=None):
    """
    Build an AE dataloader based on different tasks.

    Args:
        dataset: AE dataset object
        dataloader_param: config of AE dataloader
    Return:
        dataloader: an AE dataloader object matched with the given dataloader_mode
    Optional parameters:
        process_group: if it is absent or None, use default process group

    Raises:
        NotImplementedError: An error raised when the given `dataloader_mode` is not supported
    """
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()
    dataloader_mode = dataloader_param.pop("dataloader_mode")
    process_group = process_group if process_group is not None else _get_default_group()

    if dataloader_mode == "sampler":
        args = get_ae_args()
        batch_size = args.micro_batch_size
        num_workers = args.num_workers
        data_loader = prepare_sampler_dataloader(
            dataset, batch_size=batch_size, num_workers=num_workers, **dataloader_param, process_group=process_group
        )
        return data_loader
    else:
        raise NotImplementedError(dataloader_mode)
