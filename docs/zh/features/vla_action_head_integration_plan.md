# MindSpeed-MM 接入 UnifoLM-VLA Action Head（不含 Action Loss）

本文档给出在 MindSpeed-MM 中先接入 action head、暂不接入 action loss 的实施步骤，并说明使用 `UnifoLM-VLA-Libero` 现有资产时的注意事项。

## 1. 先回答一个关键问题：现有三件套能否“直接继续训练”

你当前的三件套：

- `UnifoLM-VLA-Libero-config.yaml`
- `UnifoLM-VLA-Libero-dataset_statistics.json`
- `checkpoints/pytorch_model.pt`

它们能构成一套完整的 **UnifoLM-VLA（Accelerate/HF 体系）** 训练/推理资产，但在 MindSpeed-MM 中 **通常不能直接无改造继续训练**，原因：

1. MindSpeed-MM 训练入口是 Megatron 体系（`pretrain_vlm.py` + `--mm-model/--mm-data`），模型结构与参数组织方式不同。  
2. `pytorch_model.pt` 是 unifolm 训练脚本保存的整模型 state_dict（见 [train_unifolm_vla.py](file:///d:/cv-big-model/%E9%A9%BB%E5%9C%BA-%E7%9F%B3%E5%8C%96/vla%E9%A1%B9%E7%9B%AE/unifolm-vla-main/src/unifolm_vla/training/train_unifolm_vla.py#L445-L449)），不是 MindSpeed-MM/Megatron 原生分布式 checkpoint 形态。  
3. 因此需要做“参数名映射 +（可能）分片转换 + 新增模块初始化策略”。

结论：**可以复用，但不能不经处理直接加载到 MindSpeed-MM 训练。**

---

## 2. 本阶段目标（你当前诉求）

- 已完成：RLDS 数据链路接入
- 本阶段：仅接入 action head，先打通模型前向输出 `action_pred`
- 暂不做：action loss（由其他同事开发）

---

## 3. 接入 action head（不接 loss）的最小实施步骤

### 步骤 1：新增 action head 模块目录

建议新增：

- `mindspeed_mm/models/action/`
  - `action_head.py`（统一入口）
  - 可选拆分：`dit_action_head.py`、`flow_matching_modules/*`

先实现最小接口：

- `forward(hidden_states, state=None) -> action_pred`
- 不做 loss，只产出动作预测张量

### 步骤 2：扩展 mm_model 配置

在 `--mm-model` 对应 JSON 中增加 `action_head` 配置段，例如：

- `enable: true`
- `type: "dit_l"`（或其他）
- `action_dim/state_dim`
- `hidden_size/num_layers`
- `num_queries/action_horizon`

目的：让 action head 结构完全由配置驱动，避免硬编码。

### 步骤 3：在 `VLMModel.__init__` 构建 action head

在 [VLMModel](file:///d:/cv-big-model/%E9%A9%BB%E5%9C%BA-%E7%9F%B3%E5%8C%96/vla%E9%A1%B9%E7%9B%AE/MindSpeed-MM-master/mindspeed_mm/models/vlm_model.py#L47-L115) 中：

- 读取 `config.action_head`
- `enable=true` 时实例化 action head
- 建议只在 text decoder 最后 stage（`post_process=True`）持有 action head，减少 PP 下重复

### 步骤 4：让前向拿到 hidden states

当前 `MMGPTModel` 在 `not post_process or reward_process` 时返回 hidden states（见 [mm_gpt_model.py](file:///d:/cv-big-model/%E9%A9%BB%E5%9C%BA-%E7%9F%B3%E5%8C%96/vla%E9%A1%B9%E7%9B%AE/MindSpeed-MM-master/mindspeed_mm/models/common/mm_gpt_model.py#L321-L332)）。

可选两种方案：

1. **VLA分支使用 hidden-state 返回模式**（推荐先做，改动小）  
2. 增加“同时返回 logits + hidden_states”的新分支（改动更大但通用性更高）

### 步骤 5：在 `VLMModel.forward` 增加 action_pred 输出

在 `forward` 返回结构里新增：

- `action_pred`
- 可选：`action_aux`（中间特征）

此阶段先不改 `loss_dict`，保持原 LM loss 路径兼容。

### 步骤 6：训练 step 先透传 action_pred

在 `pretrain_vlm.py` 的 `forward_step/loss_func` 保持现有 loss 逻辑，同时允许 `output_tensor` 包含 `action_pred`，用于后续 action loss 分支接入。

---

## 4. 参数加载策略（建议）

### 4.1 VLM 底座加载

- 继续使用 MindSpeed-MM 的 `--load` 加载 VLM 底座权重（Qwen2.5-VL/UnifoLM-VLM-0 对应路径）

### 4.2 action head 加载

- 若你有 `UnifoLM-VLA-Libero` 的 action head 参数：
  - 建议单独做脚本从 `pytorch_model.pt` 中提取 `action_model.*` 并映射到新模块 key
- 若无可用映射：
  - action head 随机初始化，先验证前向可用

### 4.3 dataset_statistics 的使用

- `dataset_statistics.json` 主要用于动作/状态归一化与反归一化，属于数据/评估辅助资产，不是模型权重。
- 训练阶段可先使用 RLDS pipeline 动态统计；推理/评估阶段建议可显式加载固定统计文件，保证一致性。

---

## 5. 除了“数据 + action head + action loss”之外，还需要改哪些框架点

为保证能稳定训练 `unifolm-vla` 风格模型，建议一并规划以下改造：

1. **训练任务分支化**  
   增加 `pretrain_vla.py` 或 `--mm-task vla`，避免把 VLA 逻辑与纯 VLM 逻辑强耦合在一个脚本里。

2. **checkpoint 兼容层**  
   增加 unifolm `pytorch_model.pt` -> MindSpeed-MM 参数映射/导入脚本（至少支持 action head 子模块导入）。

3. **优化器参数组**  
   支持 action head 与 VLM 分组学习率（如 unifolm 中 action head lr 更大）。

4. **冻结策略扩展**  
   支持“仅训 action head / 部分解冻 VLM / 全量联合训”三种模式，配置化控制。

5. **VLA 评估接口**  
   增加 `predict_action` 路径与标准化反归一化工具，便于在不接 loss 时先做动作预测 sanity check。

6. **并行策略适配**  
   明确 action head 在 PP/TP/CP 下的放置与输入形状约束（尤其 hidden states 是否需要 gather）。

7. **日志与监控**  
   提前预留 action 相关监控键（如 `action_pred_norm`、`state_norm`、后续 `action_loss`）。

---

## 6. 推荐推进顺序

1. 接入 action head 模块与配置  
2. 打通 `forward -> action_pred`（无 loss）  
3. 打通 checkpoint 部分加载（至少 action head）  
4. 接入 action loss（你同事负责）  
5. 增加 VLA 任务化训练入口与评估入口

这样可以把风险拆开，保证每一步都可独立验证。
