#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 MindSpeed-MM 训练后的权重导出为 UnifoLM 风格 pytorch_model.pt。

流程:
1) 先用 mm-convert 把 mm ckpt 转回 HF 目录 (model-xxxxx.safetensors + config/tokenizer)
2) 本脚本读取 HF 主干权重 + mm ckpt中的 action_head 权重
3) 打包为单文件 pytorch_model.pt，并生成 json/yaml 配置
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file as load_safetensors


def _load_hf_state_dict(hf_dir: Path) -> Dict[str, torch.Tensor]:
    files = sorted(hf_dir.glob("model-*.safetensors"))
    if not files:
        files = sorted(hf_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"在 {hf_dir} 下未找到 safetensors 权重")
    state: Dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_safetensors(str(f), device="cpu"))
    return state


def _to_unifolm_qwen_key(hf_key: str) -> str:
    # 逆映射 extract_unifolm_to_hf_like.py 的 normalize 规则
    if hf_key.startswith("visual."):
        return "qwen_vl_interface.model.model.visual." + hf_key[len("visual."):]
    if hf_key.startswith("model."):
        return "qwen_vl_interface.model.model.language_model." + hf_key[len("model."):]
    if hf_key.startswith("lm_head."):
        return "qwen_vl_interface.model.lm_head." + hf_key[len("lm_head."):]
    # 兼容可能出现的其它根键，仍归到 qwen_vl_interface 下
    return "qwen_vl_interface." + hf_key


def _pick_iteration(mm_ckpt_root: Path, iteration: str | None) -> Path:
    if iteration:
        p = mm_ckpt_root / iteration
        if p.exists():
            return p
        raise FileNotFoundError(f"iteration目录不存在: {p}")
    latest = mm_ckpt_root / "latest_checkpointed_iteration.txt"
    if latest.exists():
        text = latest.read_text(encoding="utf-8").strip()
        p = mm_ckpt_root / text
        if p.exists():
            return p
    p = mm_ckpt_root / "release"
    if p.exists():
        return p
    raise FileNotFoundError("无法定位 mm iteration 目录，请指定 --iteration")


def _iter_shards(iter_dir: Path) -> List[Tuple[Path, int, int]]:
    """
    返回 (dir, tp_rank, pp_rank)
    """
    out = []
    pat = re.compile(r"^mp_rank_(\d{2})(?:_(\d{3}))?(?:_(\d{3}))?$")
    for p in sorted(iter_dir.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        tp = int(m.group(1))
        pp = int(m.group(2)) if m.group(2) is not None else 0
        out.append((p, tp, pp))
    if not out:
        raise RuntimeError(f"在 {iter_dir} 下未找到 mp_rank_* 分片目录")
    return out


def _extract_action_from_mm(mm_ckpt_root: Path, iteration: str | None) -> Dict[str, torch.Tensor]:
    iter_dir = _pick_iteration(mm_ckpt_root, iteration)
    shards = _iter_shards(iter_dir)
    max_pp = max(pp for _, _, pp in shards)
    # 目前按 tp=1 导出；若多tp建议先做tp merge再导出
    target = [(d, tp, pp) for d, tp, pp in shards if pp == max_pp]
    tp_ranks = sorted({tp for _, tp, _ in target})
    if len(tp_ranks) != 1:
        raise RuntimeError(f"检测到多TP分片 {tp_ranks}，当前脚本仅支持 TP=1 导出。")

    ckpt_file = target[0][0] / "model_optim_rng.pt"
    obj = _load_mm_checkpoint(ckpt_file)
    if not isinstance(obj, dict) or "model" not in obj or not isinstance(obj["model"], dict):
        raise RuntimeError(f"无效 checkpoint 格式: {ckpt_file}")
    model_sd: Dict[str, torch.Tensor] = obj["model"]
    action = {}
    for k, v in model_sd.items():
        if k.startswith("action_head.") and torch.is_tensor(v):
            action["action_model." + k[len("action_head."):]] = v.detach().cpu()
    if not action:
        raise RuntimeError("在最终PP分片中未发现 action_head.* 权重")
    return action


def _load_mm_checkpoint(ckpt_file: Path):
    """
    兼容 PyTorch 2.6+/torch_npu 的安全反序列化限制。
    """
    try:
        # 在新版本里显式关闭 weights_only，允许完整对象反序列化
        return torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版本 torch.load 不支持 weights_only 参数
        return torch.load(ckpt_file, map_location="cpu")
    except Exception:
        # 回退：允许 argparse.Namespace 反序列化
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([argparse.Namespace]):
                try:
                    return torch.load(ckpt_file, map_location="cpu", weights_only=False)
                except TypeError:
                    return torch.load(ckpt_file, map_location="cpu")
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([argparse.Namespace])
            try:
                return torch.load(ckpt_file, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(ckpt_file, map_location="cpu")
        raise


def _infer_action_cfg(action_sd: Dict[str, torch.Tensor]) -> Dict[str, int | bool]:
    def shp(name: str, default=None):
        t = action_sd.get(name)
        return list(t.shape) if isinstance(t, torch.Tensor) else default

    block_ids = []
    to_k_dims = {}
    for k in action_sd.keys():
        m = re.match(r"action_model\.model\.transformer_blocks\.(\d+)\.attn1\.to_k\.weight$", k)
        if m:
            idx = int(m.group(1))
            block_ids.append(idx)
            to_k_dims[idx] = action_sd[k].shape[1]
    num_layers = max(block_ids) + 1 if block_ids else 0
    interleave = False
    if to_k_dims:
        vals = sorted(set(to_k_dims.values()))
        if len(vals) == 2:
            bigger, smaller = max(vals), min(vals)
            interleave = all((to_k_dims[i] == (bigger if i % 2 == 0 else smaller)) for i in sorted(to_k_dims))

    inner_dim = shp("action_model.future_tokens.weight", [0, 0])[1]
    horizon_tokens = shp("action_model.future_tokens.weight", [0, 0])[0]
    action_dim = shp("action_model.action_decoder.layer2.weight", [0, 0])[0]
    state_dim = shp("action_model.state_encoder.layer1.weight", [0, 0])[1]
    hidden_size = shp("action_model.action_decoder.layer1.weight", [0, 0])[0]
    cross_dim = max(to_k_dims.values()) if to_k_dims else 0
    num_heads = 16 if inner_dim % 16 == 0 else 1
    head_dim = inner_dim // num_heads if num_heads else inner_dim
    return {
        "input_embedding_dim": int(inner_dim),
        "hidden_size": int(hidden_size),
        "action_dim": int(action_dim),
        "state_dim": int(state_dim),
        "num_target_vision_tokens": int(horizon_tokens),
        "num_layers": int(num_layers),
        "num_attention_heads": int(num_heads),
        "attention_head_dim": int(head_dim),
        "cross_attention_dim": int(cross_dim),
        "interleave_self_attention": bool(interleave),
    }


def _write_json_config(hf_dir: Path, out_json: Path):
    cfg_src = hf_dir / "config.json"
    if not cfg_src.exists():
        raise FileNotFoundError(f"未找到 {cfg_src}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_yaml_config(
    out_yaml: Path,
    action_cfg: Dict[str, int | bool],
    base_vlm_path: str,
):
    txt = f"""seed: 42
framework:
  name: QwenFM
  qwenvl:
    base_vlm: {base_vlm_path}
    model_type: qwen2_5_vl
  action_model:
    input_embedding_dim: {action_cfg["input_embedding_dim"]}
    hidden_size: {action_cfg["hidden_size"]}
    add_pos_embed: true
    max_seq_len: 1024
    action_dim: {action_cfg["action_dim"]}
    state_dim: {action_cfg["state_dim"]}
    action_horizon: 8
    num_target_vision_tokens: {action_cfg["num_target_vision_tokens"]}
    diffusion_model_cfg:
      cross_attention_dim: {action_cfg["cross_attention_dim"]}
      attention_head_dim: {action_cfg["attention_head_dim"]}
      num_attention_heads: {action_cfg["num_attention_heads"]}
      interleave_self_attention: {"true" if action_cfg["interleave_self_attention"] else "false"}
      num_layers: {action_cfg["num_layers"]}
      output_dim: {action_cfg["hidden_size"]}
      positional_embeddings: null
"""
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(txt, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export MindSpeed-MM checkpoint to UnifoLM-style pytorch_model.pt")
    parser.add_argument("--hf-dir", type=Path, required=True, help="mm_to_hf 导出的HF目录")
    parser.add_argument("--mm-ckpt-root", type=Path, required=True, help="注入了action_head的MM checkpoint根目录")
    parser.add_argument("--output-pt", type=Path, required=True, help="输出 pytorch_model.pt 路径")
    parser.add_argument("--iteration", type=str, default=None, help="release 或 iter_xxx")
    parser.add_argument("--output-vlm-json", type=Path, default=None, help="输出VLM config json路径")
    parser.add_argument("--output-vla-yaml", type=Path, default=None, help="输出VLA config yaml路径")
    parser.add_argument("--base-vlm-path", type=str, default="./weights/checkpoints/pytorch_model.pt", help="写入yaml的base_vlm字段")
    args = parser.parse_args()

    hf_sd = _load_hf_state_dict(args.hf_dir)
    qwen_sd = {_to_unifolm_qwen_key(k): v.detach().cpu() for k, v in hf_sd.items()}
    action_sd = _extract_action_from_mm(args.mm_ckpt_root, args.iteration)
    merged = {**qwen_sd, **action_sd}

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_pt)

    summary = {
        "hf_keys": len(hf_sd),
        "qwen_unifolm_keys": len(qwen_sd),
        "action_keys": len(action_sd),
        "total_keys": len(merged),
        "output_pt": str(args.output_pt),
    }

    if args.output_vlm_json:
        _write_json_config(args.hf_dir, args.output_vlm_json)
        summary["output_vlm_json"] = str(args.output_vlm_json)

    if args.output_vla_yaml:
        action_cfg = _infer_action_cfg(action_sd)
        _write_yaml_config(args.output_vla_yaml, action_cfg, args.base_vlm_path)
        summary["output_vla_yaml"] = str(args.output_vla_yaml)
        summary["inferred_action_cfg"] = action_cfg

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
