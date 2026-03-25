"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass,field
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Callable

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from qwen_vl_utils import process_vision_info
from mindspeed_mm.data.rlds_dataloader.datasets.rlds import make_interleaved_dataset, make_single_dataset
from mindspeed_mm.data.rlds_dataloader.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoProcessor, PreTrainedTokenizerBase
from mindspeed_mm.data.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


@dataclass
class RLDSBatchTransform:
    """
    RLDSBatchTransform
    将 RLDS 格式的单条样本，转换成模型训练/推理时需要的标准格式。
    主要工作：
    1. 提取主相机图像序列并转成 PIL.Image 列表
    2. 读取语言指令并构造多轮对话 prompt
    3. 可选地读取本体感知（proprio）数据
    4. 使用 HuggingFace AutoProcessor 对图文进行 tokenize / 预处理
    5. 组装成包含 input_ids、attention_mask、pixel_values、actions（及 proprio）的最终 dict
    """

    # 是否使用腕部相机
    use_wrist_image: bool = False

    # 是否把本体感知向量（关节角/末端位姿等）喂给模型
    use_proprio: bool = False

    # HuggingFace 多模态 processor，例如 Qwen-VL、LLaVA 等
    processor: AutoProcessor = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        入口函数：把一条 RLDS numpy 样本 -> 模型批处理字典
        返回的 dict 可直接被 PyTorch DataLoader 的 collator 二次处理
        """

        # 1. 取出数据集名与动作序列
        dataset_name = rlds_batch["dataset_name"].decode("utf-8")
        actions = rlds_batch["action"]  # shape: [window_size + future, action_dim]

        # 2. 统计时间窗长度（window_size=1 表示单帧预测，>1 表示多帧历史）
        window_size = rlds_batch["observation"]["image_primary"].shape[0]

        # 3. 把每张主相机 numpy 图转成 PIL.Image，方便后续做 augmentation / resize
        images = []
        for i in range(window_size):
            images.append(Image.fromarray(rlds_batch["observation"]["image_primary"][i]))

        # 4. 读取自然语言指令并转小写
        text = rlds_batch["task"]["language_instruction"].decode().lower()

        # 5. 如果开启 proprio，则把本体感知序列拿出来
        proprio = None
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]  # shape: [window_size, proprio_dim]

        # 6. 构造多模态对话 prompt
        #    告诉模型“你是一个使用关节控制的机器人，任务如下，请预测最多 10 个关键轨迹点”
        #    这样设计方便模型输出形如 [[x1,y1], ...] 的纯文本轨迹，后续再解析成动作
        text = (
            f'You are a robot using the joint control. The task is "{text.lower()}". '
            "Please predict up to 10 key trajectory points to complete the task. "
            "Your answer should be formatted as a list of tuples, "
            "i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
        )

        # 7. 组装成 HuggingFace 多模态消息格式：user 角色，内容 = 图片列表 + 文本
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": v} for v in images],
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 8. 使用 processor 的 chat_template 把消息序列转成纯文本（带 <im_start> <im_end> 等特殊 token）
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 9. 把消息中的图片/视频路径提取出来，供 processor 做真正的像素预处理
        image_inputs, video_inputs = process_vision_info(messages)

        # 10. 最终调用 processor 得到张量
        #     返回的 batch_input 至少包含：
        #     - input_ids: 1D int64
        #     - attention_mask: 1D int64
        #     - pixel_values: [num_images, C, H, W] 或 flatten 后的格式，取决于 processor
        batch_input = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,  # 统一长度，便于 batch
            return_tensors="pt",
        )

        # 11. 如果 window_size > 1，我们只取最后一帧对应的动作作为预测目标
        #     （历史帧仅提供视觉上下文，不重复预测动作）
        if window_size > 1:
            actions = actions[window_size - 1 :, :]

        # 12. 把动作与 proprio 塞进返回字典，后续由 collator 负责 pad / stack
        batch_input["actions"] = actions
        batch_input["proprio"] = proprio if self.use_proprio else None

        return batch_input





class RLDSDataset(IterableDataset):
    """
    基于 RLDS（TFDS）流式数据源的 PyTorch IterableDataset 封装。
    主要作用：
    1. 根据 data_mix 自动选择单数据集或多数据集混合（OXE 预设或自定义）
    2. 按需加载主相机/腕部相机图像、语言指令、本体感知（proprio）
    3. 支持图像在线增强、动态分辨率 resize、滑窗采样
    4. 每条样本经过 batch_transform（RLDSBatchTransform）后，直接产出模型可用的 dict
    注意：这是一个流式数据集，不支持随机索引 __getitem__，仅支持 for ... in ... 迭代。
    """

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
    ) -> None:
        """
        参数说明
        ----------
        data_root_dir : Path
            TFDS 本地 root 路径，所有 RLDS 数据集统一放在该目录下。
        data_mix : str
            数据集名称或 OXE 预设混合名（如 'fractal20220817_data' 或 'oxe_magic_soup').
        batch_transform : RLDSBatchTransform
            将 RLDS 原始 numpy 样本 -> 模型输入 dict 的可调用对象。
        resize_resolution : Tuple[int, int]
            图像统一 resize 后的 (H, W)，供 frame_transform 使用。
        shuffle_buffer_size : int, default 256_000
            流式 shuffle 的缓冲区大小；越大随机性越好，但内存/延迟增加。
        train : bool, default True
            True 表示训练模式（开启 shuffle、增强）；False 为验证模式。
        image_aug : bool, default False
            是否启用图像在线增强（亮度、对比度、饱和度、色相、随机裁剪）。
        window_size : int, default 1
            滑窗长度：>1 表示一次取连续多帧作为历史观测，用于时序模型。
        """
        # 保存外部传入的核心对象
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform

        # ------------------------------------------------------------------
        # 1. 解析“混合数据集”规格
        # ------------------------------------------------------------------
        # 如果 data_mix 是 OXE 官方预设的混合名，直接读取对应配置；否则视为单数据集，权重 1.0
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        # ------------------------------------------------------------------
        # 2. 根据数据集名字自动决定需要加载哪些相机视角
        # ------------------------------------------------------------------
        # 目前规则：aloha / Unitree_all_task / g1_stack_block 均使用三目（主+左右腕），其余默认主+单腕
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        elif "Unitree_all_task" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        elif "g1_stack_block" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        # ------------------------------------------------------------------
        # 3. 生成每个子数据集的独立 kwargs 及其采样权重
        # ------------------------------------------------------------------
        # get_oxe_dataset_kwargs_and_weights 会返回:
        #   per_dataset_kwargs: List[dict]，每个元素是一个子数据集的全部超参
        #   weights: List[float]，对应每个子数据集的采样权重，已做归一化
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,                       # 目前统一不加载深度
            load_proprio=True,                      # 默认加载本体感知向量
            load_language=True,                     # 只保留带语言标签的轨迹
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        # ------------------------------------------------------------------
        # 4. 组装 RLDS 配置总字典
        # ------------------------------------------------------------------
        rlds_config = dict(
            # ---------- 轨迹级 transform ----------
            traj_transform_kwargs=dict(
                window_size=window_size,                           # 单条轨迹滑窗长度
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,   # 动作 chunk 长度-1，用于预测未来多步
                skip_unlabeled=True,                               # 跳过无语言指令的轨迹
                goal_relabeling_strategy="uniform",                # 目前未使用 goal，但保留接口
            ),
            # ---------- 帧级 transform（解码、resize、增强） ----------
            frame_transform_kwargs=dict(
                resize_size=tuple(resize_resolution),
                num_parallel_calls=16,            # 并行 decode + resize 的线程池大小
            ),
            # ---------- 多数据集混合参数 ----------
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,                 # 按权重均衡采样
            traj_transform_threads=len(mixture_spec),  # 轨迹级 transform 线程数
            traj_read_threads=len(mixture_spec),       # 读数据线程数
            train=train,
        )

        # ------------------------------------------------------------------
        # 5. 如启用图像增强，往 frame_transform_kwargs 追加增强参数
        # ------------------------------------------------------------------
        if image_aug:
            # 图像在线增强配置：依次执行以下五种增强方式
            # 1. random_resized_crop：随机裁剪并 resize 回原尺寸，scale 固定 0.9（轻微裁剪），ratio 固定 1.0（保持宽高比）
            # 2. random_brightness：随机亮度偏移，范围 [-0.2, 0.2]
            # 3. random_contrast：随机对比度系数，范围 [0.8, 1.2]
            # 4. random_saturation：随机饱和度系数，范围 [0.8, 1.2]
            # 5. random_hue：随机色相偏移，范围 [-0.05, 0.05]（约 ±9°）
            rlds_config["frame_transform_kwargs"]["image_augment_kwargs"] = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                # 增强顺序：先裁剪，再依次调整亮度、对比度、饱和度、色相
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )

        # ------------------------------------------------------------------
        # 6. 真正创建底层 RLDS 流式数据集
        # ------------------------------------------------------------------
        # 返回三元组：tf.data.Dataset、总样本数、数据集统计量（均值/方差等）
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    # ------------------------------------------------------------------
    # 7. 工厂方法：允许子类（如 EpisodicRLDSDataset）覆写，切换 make_interleaved_dataset / make_single_dataset
    # ------------------------------------------------------------------
    def make_dataset(self, rlds_config):
        """默认使用多数据集混合（interleaved）接口。"""
        return make_interleaved_dataset(**rlds_config)

    # ------------------------------------------------------------------
    # 8. 迭代器：每次 yield 一个经过 batch_transform 后的模型输入 dict
    # ------------------------------------------------------------------
    def __iter__(self) -> Dict[str, Any]:
        """流式迭代，产出已转换的样本。"""
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    # ------------------------------------------------------------------
    # 9. 数据集总长度（由 RLDS 返回，已考虑权重混合）
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.dataset_length

    # ------------------------------------------------------------------
    # 10. 禁止 map-style 随机索引
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> None:
        """IterableDataset 不支持随机索引，如需随机访问请改用 map-style Dataset。"""
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


