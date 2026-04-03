from typing import Optional

import torch
from torch import nn


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)
    return default


class ActionHead(nn.Module):
    """
    ActionHead 负责将文本（或其他模态）的隐藏状态映射为机器人动作序列。
    支持可选地融合额外状态信息，并可配置输出维度、时域长度、池化方式等。
    """

    def __init__(self, config, text_hidden_size: int):
        """
        参数
        ----------
        config : dict | object
            配置对象，包含以下可选字段：
            - hidden_size      : 内部投影维度，默认与 text_hidden_size 相同
            - action_dim       : 单步动作维度，默认 7（常见机械臂 6D + 1 夹爪）
            - action_horizon   : 预测动作步数，默认 1（也可用 num_queries 字段）
            - state_dim        : 额外状态向量维度，默认 0（不使用）
            - dropout          : dropout 比例，默认 0.0
            - hidden_layout    : 输入张量维度顺序，"sbh"(seq,batch,hidden) 或 "bsh"(batch,seq,hidden)，默认 "sbh"
            - pooling          : 序列池化方式，"last" 或 "mean"，默认 "last"
        text_hidden_size : int
            输入文本（或主干网络）的隐藏维度
        """
        super().__init__()

        # 读取配置，给出默认值
        hidden_size = int(_cfg_get(config, "hidden_size", text_hidden_size))
        action_dim = int(_cfg_get(config, "action_dim", 7))
        action_horizon = int(_cfg_get(config, "action_horizon", _cfg_get(config, "num_queries", 1)))
        state_dim = int(_cfg_get(config, "state_dim", 0) or 0)
        dropout = float(_cfg_get(config, "dropout", 0.0))
        hidden_layout = str(_cfg_get(config, "hidden_layout", "sbh")).lower()
        pooling = str(_cfg_get(config, "pooling", "last")).lower()

        # 合法性校验与回退
        if hidden_layout not in {"sbh", "bsh"}:
            hidden_layout = "sbh"
        if pooling not in {"last", "mean"}:
            pooling = "last"

        # 保存关键参数
        self.hidden_layout = hidden_layout
        self.pooling = pooling
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # 输入投影：若文本维度与目标维度一致则使用恒等映射，否则线性映射
        if text_hidden_size == hidden_size:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(text_hidden_size, hidden_size, bias=False)

        # 状态投影：若提供额外状态向量则做线性映射，否则为 None
        self.state_proj = nn.Linear(state_dim, hidden_size, bias=False) if state_dim > 0 else None

        # 正则化与输出
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, action_horizon * action_dim, bias=True)

    def _to_batch_first(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        将输入的隐藏状态统一转换为 (batch, seq, hidden) 格式，方便后续池化。
        参数
        ----------
        hidden_states : Tensor, shape 为 (S,B,H) 或 (B,S,H)
        返回
        -------
        Tensor, shape 为 (B,S,H)
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states as 3D tensor, got shape {tuple(hidden_states.shape)}")
        if self.hidden_layout == "sbh":
            # (S,B,H) -> (B,S,H)
            return hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播逻辑
        -参数
        hidden_states : 主干网络输出的序列特征，通常为 (B,S,H) 或 (S,B,H)
        state : 可选的额外状态向量，若提供则与文本特征融合 , shape 为 (B,state_dim) 或 (B,T,state_dim)
        -返回
        预测的动作序列, shape 为 (B, action_horizon, action_dim)
        """
        # 1. 统一维度顺序 -> (B,S,H)
        hidden_states = self._to_batch_first(hidden_states)

        # 2. 池化：取序列最后一个时刻或全局平均
        if self.pooling == "mean":
            pooled = hidden_states.mean(dim=1)  # (B,H)
        else:
            pooled = hidden_states[:, -1, :]      # (B,H)

        # 3. 输入维度映射
        pooled = self.input_proj(pooled)        # (B,hidden_size)

        # 4. 融合额外状态（若提供）
        if state is not None and self.state_proj is not None:
            if state.ndim == 3:                 # (B,T,state_dim) -> 取最后一帧
                state = state[:, -1, :]
            state = state.to(device=pooled.device, dtype=pooled.dtype)
            pooled = pooled + self.state_proj(state)  # 残差式融合

        # 5. 正则化
        pooled = self.dropout(pooled)

        # 6. 输出映射并 reshape 成动作序列
        action_pred = self.output_proj(pooled)  # (B, action_horizon * action_dim)
        batch_size = action_pred.shape[0]
        return action_pred.view(batch_size, self.action_horizon, self.action_dim)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        repeated_diffusion_steps: int = 1,
    ) -> torch.Tensor:
        action_pred = self.forward(hidden_states=hidden_states, state=state)
        actions = actions.to(device=action_pred.device, dtype=action_pred.dtype)
        if action_pred.shape != actions.shape:
            raise ValueError(
                f"ActionHead target shape mismatch, pred={tuple(action_pred.shape)}, target={tuple(actions.shape)}"
            )
        return torch.nn.functional.mse_loss(action_pred, actions)
