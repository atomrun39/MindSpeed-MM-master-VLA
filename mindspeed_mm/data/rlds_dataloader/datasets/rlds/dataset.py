"""
dataset.py

Core interface script for configuring and initializing RLDS datasets.
"""

import copy
import inspect
import json
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# from overwatch import initialize_overwatch
from mindspeed_mm.data.rlds_dataloader.datasets.rlds import obs_transforms, traj_transforms
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.utils import goal_relabeling, task_augmentation
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.utils.data_utils import (
    NormalizationType,
    allocate_threads,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
    convert_quaternion_to_euler,
)
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
import logging
# Initialize Overwatch =>> Wraps `logging.Logger`
# overwatch = initialize_overwatch(__name__)


# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")





# ruff: noqa: B006
def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    **kwargs,
) -> Tuple[dl.DLataset, dict]:
    """
    从 TFDS/RLDS 加载一个数据集，把它整理成“统一格式”的轨迹数据集，并返回：
      1. 整理后的 dl.DLataset（每条轨迹已经拆成 dict，包含 observation/task/action 等字段）
      2. 该数据集的统计量 dict（用于后续归一化、混合采样等）

    本函数只做“非 CPU 密集型”操作：
      - 不解码图像
      - 不做数据增强
      - 只做字段重排、缺失字段补零、语言提取、归一化系数计算/加载等

    设计思想：
      a) 先通过 `standardize_fn`（可选）把原始轨迹洗成至少包含 observation/action 的“最小标准格式”
      b) 用 `restructure()` 把 observation 里的原始字段按用户给出的 key-mapping 拆成
         image_*/depth_*/proprio/timestep 等统一名字
      c) 如果给了 language_key，就把它抽到 task["language_instruction"]
      d) 计算或加载归一化统计量，最后把 action & proprio 归一化掉
    """

    # --- 1. 定义“最小必需字段” -------------------------------------------------
    # 如果用户想抽语言，还必须额外提供 language_key
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    # --- 2. 定义单条轨迹的重整函数 ---------------------------------------------
    def restructure(traj: dict) -> dict:
        """
        把一条轨迹 dict 洗成统一格式：
          - observation 里出现 image_*/depth_*/proprio/timestep
          - task 里可选 language_instruction
          - action 保持原样（后面再归一化）
          - dataset_name 写入轨迹每步，方便后续追踪来源
        如果某字段在原始轨迹里缺失，就补零或补空字符串（padding）
        """
        # 2.1 可选：先让用户自定义的 standardize_fn 过一遍
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        # 2.2 再次检查必需字段
        missing = REQUIRED_KEYS - set(traj.keys())
        if missing:
            raise ValueError(
                f"Trajectory 缺少字段 {missing}。请检查 standardize_fn 是否输出正确。"
            )

        # 2.3 取出轨迹长度，方便后续补 padding
        traj_len = tf.shape(traj["action"])[0]  # 时间维度
        old_obs = traj["observation"]           # 原始 observation 子 dict

        # 2.4 按 image_obs_keys 映射 RGB 图像
        #     old 为 None  =>  补空字符串（TF 里用空串代表“padding 图像”）
        new_obs = {}
        for new_key, old_key in image_obs_keys.items():
            if old_key is None:
                new_obs[f"image_{new_key}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new_key}"] = old_obs[old_key]

        # 2.5 同理映射深度图
        for new_key, old_key in depth_obs_keys.items():
            if old_key is None:
                new_obs[f"depth_{new_key}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"depth_{new_key}"] = old_obs[old_key]

        # 2.6 把一维 proprio 字段拼接成 [T, D] 矩阵
        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # 2.7 加入 timestep 信息（0…T-1）
        new_obs["timestep"] = tf.range(traj_len)

        # 2.8 提取语言指令到 task 字段
        task = {}
        if language_key is not None:
            lang_tensor = traj[language_key]
            if lang_tensor.dtype != tf.string:
                raise ValueError(
                    f"language_key={language_key} 对应字段 dtype={lang_tensor.dtype}，必须是 tf.string。"
                )
            task["language_instruction"] = traj.pop(language_key)

        # 2.9 组装最终轨迹 dict
        new_traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),  # 每步都写数据集名字
        }

        # 2.10 如果用户指明哪些 action 维度是绝对量，就生成同形状 mask
        if absolute_action_mask is not None:
            act_dim = new_traj["action"].shape[-1]
            if len(absolute_action_mask) != act_dim:
                raise ValueError(
                    f"absolute_action_mask 长度 ({len(absolute_action_mask)}) 与 action 维度 ({act_dim}) 不符。"
                )
            new_traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],  # [1, D]
                [traj_len, 1]  # => [T, D]
            )

        return new_traj

    # --- 3. 构建 TFDS builder --------------------------------------------------
    builder = tfds.builder(name, data_dir=data_dir, version="1.0.0")

    # --- 4. 获得归一化统计量 ---------------------------------------------------
    # 4.1 用户直接给了 json 文件
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    # 4.2 用户没给，就现场算：先加载“全拆分”数据，过一遍 restructure，再算均值方差
    elif dataset_statistics is None:
        full_dataset = (
            dl.DLataset.from_rlds(
                builder,
                split="all",
                shuffle=False,
                num_parallel_reads=num_parallel_reads,
            )
            .traj_map(restructure, num_parallel_calls)
        )
        # get_dataset_statistics 会优先读缓存，缓存不存在才真算
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn else "",
            ),
            save_dir=builder.data_dir,
        )
    # 4.3 把统计量里所有 list 转 np.array，方便后续矩阵运算
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # --- 5. 如果用户想跳过某些 action 维度的归一化，就写入 mask ------------------
    if action_normalization_mask is not None:
        act_mean = dataset_statistics["action"]["mean"]
        if len(action_normalization_mask) != act_mean.shape[-1]:
            raise ValueError(
                f"action_normalization_mask 长度 ({len(action_normalization_mask)}) "
                f"与 action 维度 ({act_mean.shape[-1]}) 不符。"
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # --- 6. 决定训练 / 验证拆分 -----------------------------------------------
    if "val" not in builder.info.splits:
        # 没有官方验证集，就把 train 后 5% 当验证
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    # --- 7. 真正组装数据集 -----------------------------------------------------
    dataset = (
        dl.DLataset.from_rlds(
            builder,
            split=split,
            shuffle=shuffle,  # 只打乱“文件”顺序，不是全局打乱
            num_parallel_reads=num_parallel_reads,
        )
        .traj_map(restructure, num_parallel_calls)  # 重整字段
        .traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls,
        )
    )

    return dataset, dataset_statistics


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    对轨迹级（trajectory-level）数据进行一系列通用变换，返回变换后的 DLataset。

    设计原则：
    1. 所有变换必须在整条轨迹可见的前提下完成，无法逐帧独立处理；
    2. 计算量低，主要做“重排/截取/过滤/复制”类操作，不解码图像；
    3. 执行顺序有依赖：先过滤 -> 打 padding 标记 -> 目标重标记 -> 任务增广 -> 分窗 -> 训练期再抽样。

    主要功能：
    - 过滤掉无语言标签或动作/本体感知超限的轨迹；
    - 给观测与任务字典打“padding”标记，方便后续代码识别填充位；
    - 根据指定策略对任务目标进行重标记（如把最后几帧设为“目标帧”）；
    - 训练模式下可做任务增广（如随机丢弃部分语言指令）；
    - 将整条轨迹切成固定长度 window 的片段，支持额外附带未来动作；
    - 训练模式下若指定长度，可对过长片段做随机子采样，增加多样性。

    参数说明
    ----------
    dataset : dl.DLataset
        输入的轨迹数据集，要求每条样本为一条完整轨迹 dict。
    train : bool
        是否为训练模式。影响两项逻辑：
        1) 任务增广仅在训练期执行；
        2) 仅在训练期启用子采样（subsample）。
    goal_relabeling_strategy : str, optional
        目标重标记策略名，对应 goal_relabeling 模块下的函数名，如
        "uniform_random" / "last_chunk" 等；为 None 则跳过。
    goal_relabeling_kwargs : dict, optional
        传给目标重标记函数的关键字参数。
    window_size : int, optional
        每条输出片段的时序长度 T。默认 1 表示不切片，仅加轴。
    future_action_window_size : int, optional
        在 window 之后额外再取多少步未来动作，用于动作序列预测。
    subsample_length : int, optional
        若给定且轨迹长度 > 该值，则在训练期随机截取 subsample_length 步，
        防止长轨迹占比过大并增加数据多样性。
    skip_unlabeled : bool, optional
        为 True 时跳过所有语言标签为空的轨迹；数据集必须含 language_instruction。
    max_action : float, optional
        若给定，只要轨迹中任意动作维度出现绝对值超过该阈值，则整段丢弃。
    max_proprio : float, optional
        同上，但检查本体感知 proprio 值。
    task_augment_strategy : str, optional
        任务增广策略名，对应 task_augmentation 模块下的函数；训练期生效。
    task_augment_kwargs : dict, optional
        传给任务增广函数的额外参数。
    num_parallel_calls : int, optional
        每个 traj_map 的并行度，默认 tf.data.AUTOTUNE 由系统自动决定。

    返回
    -------
    dl.DLataset
        经过所有轨迹级变换后的数据集，元素仍为 dict，但已被切片/重标记/过滤。
    """

    # ---- 1. 过滤无语言标签的轨迹 -------------------------------------------------
    if skip_unlabeled:
        # 先检查数据集是否真的包含语言字段，防止误用
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")

        # 只要该轨迹里任意一步出现非空字符串，就保留；否则整段丢弃
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    # ---- 2. 根据动作值过滤 -------------------------------------------------------
    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action))

    # ---- 3. 根据本体感知值过滤 ----------------------------------------------------
    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["observation"]["proprio"]) <= max_proprio))

    # ---- 4. 给观测和任务字典打 padding 标记 ---------------------------------------
    # 后续代码可通过该标记判断哪些位置是填充值，从而跳过不必要计算
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # ---- 5. 目标重标记（goal relabeling） ----------------------------------------
    # 例如把 future_chunk 或最后若干帧设为“目标帧”，供目标条件策略使用
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    # ---- 6. 任务增广（task augmentation） ----------------------------------------
    # 需在 chunk 之前完成，因为增广可能改变目标时间戳或语言内容
    if train and task_augment_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # ---- 7. 将轨迹切成固定长度片段（chunk） ---------------------------------------
    # 输出形状：observation 新增第二维 [T, window_size, ...]，
    #          action 新增第二维 [T, window_size + future_action_window_size, ...]
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    # ---- 8. 训练期子采样（subsample） --------------------------------------------
    # 对仍过长的片段随机截取，进一步均衡不同长度轨迹的采样概率
    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    # ---- 9. 返回变换后的数据集 ----------------------------------------------------
    return dataset


def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    对“帧级别”的数据做通用变换，通常是 CPU 密集操作：解码、resize、数据增强等。
    输入已经是“chunk”后的数据：每条样本的 observation 里每个字段都是 [T, ...] 的时序张量。

    主要流程：
    1. 统一解码并 resize RGB/深度图；
    2. 训练模式下再做随机数据增强（所有图像共享同一随机种子，保证同步）；
    3. 所有操作通过 frame_map 并行执行，支持逐帧加速。

    Args:
        train (bool): 是否为训练模式，决定要不要做数据增强。
        dataset (dl.DLataset): 输入的帧级数据集，元素为 dict，含 observation/task/action 等。
        image_augment_kwargs (dict | Mapping[str, dict]): 传给图像增强函数的参数。
            若为 dict 嵌套 dict，则外层 key 对应 image_{key}；缺失 key 则跳过增强。
        resize_size (Tuple[int, int] | Mapping[str, Tuple[int, int]]): 目标 resize 尺寸。
            支持按图像 key 单独指定；缺失 key 则跳过 resize。
        depth_resize_size: 同上，但针对深度图。
        num_parallel_calls: 并行度，默认 AUTOTUNE。
    """

    # 工具函数：把“单帧/单任务”函数批量应用到 chunked 数据上
    #   - frame["task"] 直接 fn
    #   - frame["observation"] 里每个字段都是 [T, ...]，用 dl.vmap 沿 T 维批量 fn
    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame["task"] = fn(frame["task"])                       # task 字段本身非 chunked
        frame["observation"] = dl.vmap(fn)(frame["observation"])  # 对 T 维批量映射
        return frame

    # 第1步：解码 + resize（RGB 与深度图）
    # 使用 obs_transforms.decode_and_resize 统一处理，支持按 key 差异化尺寸
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    # 第2步：训练模式下再做随机数据增强
    if train:
        # 为保证同一帧内所有图像同步增强，先生成一个全局随机种子
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    return dataset, dataset_statistics["num_trajectories"], dataset_statistics


# === 核心初始化器：将多个数据集按权重混合成一个可迭代的数据集 ===
def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],          # 每个元素是传给 make_dataset_from_rlds 的参数字典
    sample_weights: Optional[List[float]] = None,  # 每个数据集的采样权重，None 则均匀采样
    *,
    train: bool,                             # True=训练模式，False=验证模式
    shuffle_buffer_size: int,                # 打洗缓存的帧数
    traj_transform_kwargs: Optional[Dict] = None,   # 轨迹级变换参数
    frame_transform_kwargs: Optional[Dict] = None,   # 帧级变换参数
    batch_size: Optional[int] = None,         # 若给出则最后按此 batch_size 打包
    balance_weights: bool = False,           # 是否用“数据集大小”加权，使各数据集期望被完整遍历一次
    traj_transform_threads: Optional[int] = None,  # 轨迹变换总线程数，会按权重分到各数据集
    traj_read_threads: Optional[int] = None,        # 读取总线程数，会按权重分到各数据集
) -> Tuple[dl.DLataset, int, dict]:
    """
    将多个 RLDS 数据集按指定权重混合（interleave）成一个数据集，返回帧级别的连续流。
    支持训练/验证两种模式，自动分配线程，可选 batch。
    """

    # 1. 若用户未指定采样权重，默认均匀采样
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    # 2. 简单校验
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights 必须为 None 或长度等于 {len(dataset_kwargs_list)}。")
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("必须提供 traj_transform_kwargs 与 frame_transform_kwargs！")

    # 3. 先遍历一次数据集，拿到每个数据集的“帧数”和统计信息
    dataset_sizes = []               # 每个数据集的帧数（transition 数）
    all_dataset_statistics = {}      # 按数据集名字缓存统计信息，避免重复计算
    for dataset_kwargs in dataset_kwargs_list:
        # 深拷贝，防止 pop 影响外部
        data_kwargs = copy.deepcopy(dataset_kwargs)
        # 如果某个数据集有私有的帧变换参数，先拿出来，后面单独用
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        # 仅为了拿统计信息，不真正构建大图
        _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # 4. 找出“主数据集”——即权重恰好为 1.0 的那些，用于后面计算“有效长度”
    primary_dataset_indices = np.array(
        [idx for idx, w in enumerate(sample_weights) if w == 1.0], dtype=int
    )

    # 5. 权重平衡：若 balance_weights=True，把原始权重乘以数据集大小，使得大小不同的数据集也能“公平”地被完整遍历一次
    if balance_weights:
        sample_weights = np.array(sample_weights, dtype=float) * np.array(dataset_sizes, dtype=float)
    # 归一化到概率分布
    sample_weights = np.array(sample_weights, dtype=float)
    sample_weights /= np.sum(sample_weights)
    # 打印当前数据混合比例，方便调试
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # 6. 计算“有效长度”：在 primary 数据集上，按权重折算后所需的最大采样数
    #    保证期望下每个主数据集至少被完整遍历一次
    if len(primary_dataset_indices) == 0:
        # 没有主数据集时，退而求其次用全部数据集
        primary_dataset_indices = np.arange(len(dataset_sizes))
    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

    # 7. 按权重比例分配线程（读取线程 & 变换线程）
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset   = allocate_threads(traj_read_threads,   sample_weights)
    logging.info("Threads per Dataset: %s", threads_per_dataset)
    logging.info("Reads per Dataset: %s",   reads_per_dataset)

    # 8. 真正构造每个数据集的“无限”流
    datasets = []
    for dataset_kwargs, n_thread, n_read in zip(
        dataset_kwargs_list, threads_per_dataset, reads_per_dataset
    ):
        # 8.1 取出私有帧变换参数（如果有）
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs")
            if "dataset_frame_transform_kwargs" in dataset_kwargs
            else {}
        )
        # 8.2 构建数据集（带缓存的统计信息，避免重复计算）
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=n_thread,
            num_parallel_reads=n_read,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        # 8.3 应用轨迹级变换：repeat() 让数据集无限循环，flatten 把轨迹拆成帧
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=n_thread,
            train=train,
        ).flatten(num_parallel_calls=n_thread)
        # 8.4 应用该数据集私有的帧级变换（如果有）
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    # 9. 按权重混合多个数据集的帧流
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # 10. 验证模式：先取固定缓存大小的数据并缓存到 RAM，防止内存逐渐增长
    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()

    # 11. 打洗：注意必须在 .cache() 之后，否则内存仍可能泄漏
    dataset = dataset.shuffle(shuffle_buffer_size)

    # 12. 应用通用帧级变换（解码、resize、数据增强等）
    logging.info("Applying frame transforms on dataset...")
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # 13. 可选：按 batch_size 打包
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # 14. 给数据集加 RAM 预算限制，降低内存占用
    dataset = dataset.with_ram_budget(1)

    # 15. 把采样权重挂在数据集对象上，方便外部读取
    dataset.sample_weights = sample_weights

    # 返回：混合后的数据集、有效长度、所有统计信息
    return dataset, dataset_len, all_dataset_statistics
