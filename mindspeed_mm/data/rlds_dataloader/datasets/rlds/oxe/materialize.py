"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mindspeed_mm.data.rlds_dataloader.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.utils.data_utils import NormalizationType
import logging

overwatch = logging.getLogger(__name__)


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Dict[str, Any]:
    """
    根据指定的 Open-X Embodiment 数据集名称，生成该数据集在训练/推理时所需的完整配置字典（kwargs）。

    步骤概览：
    1. 深拷贝预定义配置，避免污染全局配置；
    2. 校验动作编码是否受支持；
    3. 根据动作编码类型，生成「绝对值掩码」与「归一化掩码」：
       - 对于 EEF_* 类动作，除末端夹爪外其余维度均做归一化；
       - 对于 JOINT_* 类动作，所有维度均做归一化；
    4. 过滤用户指定的相机视角，若缺失则抛错；
    5. 按需删除本次训练不需要的字段（深度图、本体感受状态、动作/状态编码等）；
    6. 若需要语言指令，则注入统一 key；
    7. 绑定该数据集对应的标准化函数；
    8. 展开辅助参数（aux_kwargs）并追加根目录与数据集名，最终返回可直接喂给 DataLoader 的配置 dict。
    """
    # 1. 深拷贝全局配置，确保后续改动仅影响本次调用
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])

    # 2. 动作编码合法性检查：仅支持列表中的 6 种编码
    supported_action_encodings = [
        ActionEncoding.EEF_POS,
        ActionEncoding.EEF_R6,
        ActionEncoding.JOINT_POS_BIMANUAL,
        ActionEncoding.JOINT_POS,
        ActionEncoding.JOINT_G1,
        ActionEncoding.EE_R6_G1,
    ]
    if dataset_kwargs["action_encoding"] not in supported_action_encodings:
        raise ValueError(
            f"Cannot load `{dataset_name}`; only {supported_action_encodings} actions supported!"
        )

    # 3. 根据动作编码生成「绝对值掩码」与「归一化掩码」
    #    掩码长度 = 动作维度，True 表示该维度采用对应策略
    action_encoding = dataset_kwargs["action_encoding"]
    if action_encoding is ActionEncoding.EEF_POS:
        # 6D 末端位姿 + 1D 夹爪：仅夹爪维度使用绝对值，不做归一化
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif action_encoding is ActionEncoding.EEF_R6:
        # 9D 旋转表示 + 1D 夹爪：同上
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    elif action_encoding is ActionEncoding.JOINT_POS_BIMANUAL:
        # 双臂关节角：全部使用绝对值并归一化
        dataset_kwargs["absolute_action_mask"] = [True] * 14
        dataset_kwargs["action_normalization_mask"] = [True] * 14
    elif action_encoding is ActionEncoding.JOINT_POS:
        # 单臂 7D 关节角：全部使用绝对值并归一化
        dataset_kwargs["absolute_action_mask"] = [True] * 7
        dataset_kwargs["action_normalization_mask"] = [True] * 7
    elif action_encoding is ActionEncoding.JOINT_G1:
        # G1 机器人 19D 关节角：全部使用绝对值并归一化
        dataset_kwargs["absolute_action_mask"] = [True] * 19
        dataset_kwargs["action_normalization_mask"] = [True] * 19
    elif action_encoding is ActionEncoding.EE_R6_G1:
        # G1 机器人 23D 末端旋转+关节角：全部使用绝对值并归一化
        dataset_kwargs["absolute_action_mask"] = [True] * 23
        dataset_kwargs["action_normalization_mask"] = [True] * 23

    # 4. 记录本体感受归一化类型（NORMAL / NONE / …）
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # 5. 校验用户请求的相机视角是否都存在
    missing_camera_views = set(load_camera_views) - set(dataset_kwargs["image_obs_keys"])
    if missing_camera_views:
        raise ValueError(
            f"Cannot load `{dataset_name}`; missing camera views `{missing_camera_views}`"
        )

    # 6. 仅保留用户指定的相机视角对应的图像/深度 key
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # 7. 删除后续流程不再需要的字段，减少配置体积
    dataset_kwargs.pop("state_encoding")  # 状态编码类型已用不到
    dataset_kwargs.pop("action_encoding")  # 动作编码类型已用不到
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")  # 不加载深度图时直接删除
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")  # 不加载本体感受时直接删除

    # 8. 若需要语言模态，则注入统一 key
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # 9. 绑定该数据集在 OXE 中注册的标准化变换函数
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # 10. 展开辅助参数字典（若存在），避免嵌套
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    # 11. 最终返回：数据集名 + 数据根目录 + 其余所有配置
    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}

def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    根据给定的 Open X-Embodiment 数据集混合配比，生成每个子数据集的配置（kwargs）以及对应的采样权重。
    返回结果可直接喂给 `make_interleaved_dataset`，用于构建多数据集混合采样器。

    :param data_root_dir: 存放 RLDS/TFDS 格式数据集的根目录
    :param mixture_spec: 列表，元素为 (dataset_name, sampling_weight)，来自 `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: 需要加载的相机视角，默认只加载主视角；详见 `oxe.dataset_configs.py`
    :param load_depth: 是否在 RGB 之外再加载深度图
    :param load_proprio: 是否加载本体感受状态（关节角等）
    :param load_language: 是否加载语言指令
    :param action_proprio_normalization_type: 对本体感受动作采用的归一化方式

    :return: (per_dataset_kwargs, sampling_weights)
             per_dataset_kwargs: 每个子数据集的配置字典列表
             sampling_weights:   与上述配置一一对应的采样权重列表
    """

    # 第一步：去重
    # 用 set 记录已处理的数据集，防止同名的重复项导致权重叠加或配置冲突
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            # 发现重复数据集，记录警告并跳过
            overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)          # 登记已处理
        filtered_mixture_spec.append((d_name, d_weight))  # 保留去重后的列表

    # 第二步：逐个数据集生成配置并收集权重
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            # 调用工厂函数，生成单数据集配置
            kwargs = make_oxe_dataset_kwargs(
                d_name,
                data_root_dir,
                load_camera_views,
                load_depth,
                load_proprio,
                load_language,
                action_proprio_normalization_type,
            )
            per_dataset_kwargs.append(kwargs)  # 收集配置
            sampling_weights.append(d_weight)  # 收集对应权重

        except ValueError as e:
            # 如果某个数据集因不支持的动作编码等原因无法加载，记录警告并跳过
            overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")

    # 返回配置列表与权重列表，二者长度一致，可一一对应
    return per_dataset_kwargs, sampling_weights
