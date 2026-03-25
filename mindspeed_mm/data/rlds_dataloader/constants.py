"""
VLA（Vision-Language-Action）训练与评估所需的关键常量。

本模块通过解析启动训练或评估的 Python 命令行参数，自动识别应使用的机器人平台常量；
若无法识别，则默认使用 G1_EE_6D（宇树G1机器人末端执行器6D位姿控制模式）的常量。
"""

import sys
from enum import Enum


# ============================================
# 1. 分词器（Tokenizer）相关常量
# ============================================

# IGNORE_INDEX: 计算损失时忽略的标签索引
# 在PyTorch的CrossEntropyLoss中，-100是默认的忽略索引
# 用于标记填充位置（padding）或不应计算损失的token
IGNORE_INDEX = -100

# ACTION_TOKEN_BEGIN_IDX: 动作 token 在词表中的起始索引
# 这是Llama 2词表中预留给动作输出的特殊token起始位置
# 模型会输出从31743开始的连续token来表示动作值
ACTION_TOKEN_BEGIN_IDX = 31743

# STOP_INDEX: 句子结束符 '</s>' 对应的 token id
# 模型生成到这个token时停止，表示推理完成
STOP_INDEX = 2  # '</s>'


# ============================================
# 2. LISA 方法专用常量
# ============================================

# LISA（Language-conditioned Imitation with Spatial Attention）是一种VLA方法
# ACTION_TOKEN_IDX: LISA方法中专门用于表示"动作"的特殊token
# 模型通过这个特殊token来区分视觉描述和动作指令
ACTION_TOKEN_IDX = 32001


# ============================================
# 3. 数据归一化方式枚举
# ============================================

class NormalizationType(str, Enum):
    """
    动作与本体感受状态的归一化方式枚举
    
    为什么需要归一化？
    - 不同机器人的动作范围差异巨大（如关节角度-180~180度 vs 位置0~1米）
    - 神经网络对输入数值范围敏感，归一化到标准范围能稳定训练
    """
    # fmt: off  # 关闭自动格式化，保持代码整洁
    
    # NORMAL: 均值0、标准差1的正态归一化（Z-score标准化）
    # 公式: (x - mean) / std
    # 适用于: 数据分布近似高斯分布的情况
    NORMAL = "normal"
    
    # BOUNDS: 线性映射到区间 [-1, 1]
    # 公式: (x - min) / (max - min) * 2 - 1
    # 适用于: 数据范围稳定、无极端异常值的情况（如ALOHA遥操作数据）
    BOUNDS = "bounds"
    
    # BOUNDS_Q99: 用1%和99%分位数截断后再映射到 [-1, 1]
    # 公式: 先截断到[1%分位数, 99%分位数]范围，再线性映射到[-1, 1]
    # 适用于: 真实机器人数据有噪声/异常值的情况（如LIBERO仿真到真实的迁移）
    # 优点: 对极端异常值（如传感器故障导致的跳变）更鲁棒
    BOUNDS_Q99 = "bounds_q99"
    
    # fmt: on  # 恢复自动格式化


# ============================================
# 4. 各机器人平台的常量定义
# ============================================

# 常量说明：
# NUM_ACTIONS_CHUNK: 每次推理输出的未来动作步数（Action Chunking长度）
#   - 不是预测1步，而是一次预测未来N步，保证动作连续性，减少抖动
#   
# ACTION_DIM: 单步动作向量的维度
#   - 通常包括: 3维位置 + 3维旋转 + 1维夹爪开合 = 7维（单臂）
#   - 双臂就是14维，加上腰部/头部可能更多
#
# PROPRIO_DIM: 本体感受（Proprioception）向量的维度
#   - 机器人当前自身状态，如关节角度、末端位置、速度等
#   - 可以作为额外输入给模型，帮助理解自身姿态
#
# ACTION_PROPRIO_NORMALIZATION_TYPE: 数据归一化方式选择


# LIBERO: 单臂/双臂仿真机器人基准测试平台
# 特点: 基于MetaWorld的仿真环境，任务多样（抓、放、推等）
# 为什么用BOUNDS_Q99: 仿真数据可能有物理引擎导致的异常抖动
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,      # 预测未来8步（约0.4秒@20Hz）
    "ACTION_DIM": 7,             # 单臂: 3位置+3旋转+1夹爪
    "PROPRIO_DIM": 8,            # 末端位姿(7)+夹爪状态扩展
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# ALOHA: 斯坦福双臂遥操作真机平台
# 特点: 低成本双臂，真人遥控收集数据，用于精细操作（如穿线、叠衣服）
# 为什么用BOUNDS: 遥操作数据经过人工筛选，质量稳定，范围可控
ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,     # 预测未来25步（约1秒@25Hz），补偿通信延迟
    "ACTION_DIM": 14,            # 双臂: 7+7
    "PROPRIO_DIM": 14,           # 双臂关节角或末端位姿
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

# Bridge Data: 谷歌发布的机器人操作数据集
# 特点: 单臂，真实厨房环境，多样化任务
# 为什么用BOUNDS_Q99: 真实环境有传感器噪声
BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,      # 较短窗口，适合快速响应任务
    "ACTION_DIM": 7,             # 单臂标准7维
    "PROPRIO_DIM": 7,            # 单臂本体感知
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Fractal: 分形机器人（具体指代需根据上下文，可能是特定研究项目）
# 配置与LIBERO类似
FRACTAL_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# G1: 宇树科技Unitree G1人形机器人（关节空间控制模式）
# 特点: 23自由度人形，腰部、头部、双臂可动
# 为什么用BOUNDS: 关节限位明确，数据范围稳定
G1_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,     # 人形需要长程规划，25步约1秒
    "ACTION_DIM": 16,            # 16维动作（可能是精选关键关节）
    "PROPRIO_DIM": 16,           # 16维本体感知
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

# G1_EE_6D: 宇树G1末端执行器6D位姿控制模式（推荐配置）
# 特点: 不直接控制关节，而是控制双手末端的位置+旋转（6D）
# 为什么用BOUNDS_Q99: 6D位姿计算可能有数值不稳定情况
# 这是默认配置，说明项目主要针对G1的末端执行器控制
G1_EE_6D_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,     # 长窗口适合人形平滑运动
    "ACTION_DIM": 23,            # 双臂末端6D(12)+腰部/头部(11?) 或完整关节配置
    "PROPRIO_DIM": 23,           # 23维本体感知（对应23自由度）
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# G1_STACK_BLOCK: 宇树G1叠积木任务专用配置
# 特点: 针对特定精细操作任务优化
# 与G1_EE_6D相同配置，说明使用相同的控制模式
G1_STACK_BLOCK_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# ============================================
# 5. 平台自动检测逻辑
# ============================================

def detect_robot_platform():
    """
    自动检测当前训练/评估使用的是哪个机器人平台。
    
    检测原理:
    1. 把Python命令行参数（sys.argv）拼成一个大字符串
    2. 转小写后查找关键字（如"libero"、"aloha"等）
    3. 返回匹配的平台名称
    
    为什么用命令行参数检测？
    - 训练脚本通常通过 --robot_platform=libero 这类参数指定平台
    - 比修改代码更灵活，避免频繁改动constants.py
    
    示例:
    python train.py --robot_platform=libero → 检测到"libero" → 返回"LIBERO"
    python eval.py --dataset=aloha_test   → 检测到"aloha"  → 返回"ALOHA"
    """
    # 把所有命令行参数（包括脚本名和--xxx=yyy）用空格连接
    # 例如: ['train.py', '--robot_platform=libero', '--batch_size=8']
    # 变成: "train.py --robot_platform=libero --batch_size=8"
    cmd_args = " ".join(sys.argv).lower()
    
    # 打印完整命令行，方便调试时查看检测到了什么
    print(f"[constants.py] 检测到的命令行参数: {cmd_args}")
    
    # 按优先级顺序检查关键字（注意顺序，更具体的先检查）
    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "fractal" in cmd_args:
        return "FRACTAL"
    elif "ee_6d" in cmd_args:
        # ee_6d比joint更具体，先检查
        return "G1_EE_6D"
    elif "joint" in cmd_args:
        # 简单的"joint"表示G1关节控制模式
        return "G1"
    elif "stack_block" in cmd_args:
        return "G1_STACK_BLOCK"
    else:
        # 默认兜底：使用G1_EE_6D配置
        # 这说明项目主要面向宇树G1机器人的末端执行器控制
        return "G1_EE_6D"


# 执行平台检测，结果保存到全局变量ROBOT_PLATFORM
ROBOT_PLATFORM = detect_robot_platform()


# ============================================
# 6. 根据检测到的平台选择常量字典
# ============================================

# 使用if-elif链选择对应的常量字典
# 这样constants变量就指向了正确的配置字典
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "FRACTAL":
    constants = FRACTAL_CONSTANTS
elif ROBOT_PLATFORM == "G1_EE_6D":
    constants = G1_EE_6D_CONSTANTS
elif ROBOT_PLATFORM == "G1":
    constants = G1_CONSTANTS
elif ROBOT_PLATFORM == "G1_STACK_BLOCK":
    constants = G1_STACK_BLOCK_CONSTANTS


# ============================================
# 7. 导出全局常量（模块级变量）
# ============================================

# 将选中平台的常量提取为模块级全局变量
# 其他代码可以直接: from training.vla.constants import ACTION_DIM
# 而不需要知道内部的平台检测逻辑

# 预测窗口长度：模型一次生成多少步未来动作
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]

# 动作维度：单步动作向量的长度
ACTION_DIM = constants["ACTION_DIM"]

# 本体感知维度：机器人当前状态的维度
PROPRIO_DIM = constants["PROPRIO_DIM"]

# 归一化类型：数据预处理使用哪种归一化策略
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]


# ============================================
# 8. 启动日志输出
# ============================================

# 打印当前使用的平台及关键常量，便于运行日志检查
print(f"Using {ROBOT_PLATFORM} constants:")
print(f" in constants.py NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `training/vla/constants.py`!")