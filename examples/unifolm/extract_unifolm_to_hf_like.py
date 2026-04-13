#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 UnifoLM-VLA 的 pytorch_model.pt 中拆分:
1) qwen_vl_interface -> HF-like 权重 (供 MindSpeed-MM qwen2.5vl 转换)
2) action_model      -> 独立权重 (后续注入 action_head)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    from safetensors.torch import save_file as save_safetensors
except Exception:  # pragma: no cover
    save_safetensors = None


def _ensure_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        for candidate in ("state_dict", "model", "module"):
            maybe = raw_obj.get(candidate)
            if isinstance(maybe, dict):
                return maybe
        if all(torch.is_tensor(v) for v in raw_obj.values()):
            return raw_obj
    raise ValueError("无法从输入文件解析 state_dict，请检查输入文件格式。")


def _normalize_qwen_key(src_key: str) -> str:
    """
    将 UnifoLM 的 qwen_vl_interface key 规范化到 MindSpeed-MM qwen2.5vl converter 可识别的形式。
    核心目标:
    - qwen_vl_interface.model.model.visual.*           -> visual.*
    - qwen_vl_interface.model.model.language_model.*   -> model.*
    - qwen_vl_interface.model.lm_head.*                -> lm_head.*
    """
    if not src_key.startswith("qwen_vl_interface."):
        raise ValueError(f"非法 qwen key: {src_key}")

    key = src_key[len("qwen_vl_interface."):]

    rules: Tuple[Tuple[str, str], ...] = (
        ("model.model.visual.", "visual."),
        ("model.visual.", "visual."),
        ("model.model.language_model.", "model."),
        ("model.language_model.", "model."),
        ("model.lm_head.", "lm_head."),
    )
    for old, new in rules:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def _normalize_action_key(src_key: str) -> str:
    if src_key.startswith("action_model."):
        return src_key[len("action_model."):]
    return src_key


def extract_weights(input_pt: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    raw = torch.load(input_pt, map_location="cpu")
    state_dict = _ensure_state_dict(raw)

    qwen_weights: Dict[str, torch.Tensor] = {}
    action_weights: Dict[str, torch.Tensor] = {}
    skipped = 0

    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            skipped += 1
            continue
        if k.startswith("qwen_vl_interface."):
            qwen_weights[_normalize_qwen_key(k)] = v.detach().cpu()
        elif k.startswith("action_model."):
            action_weights[_normalize_action_key(k)] = v.detach().cpu()
        else:
            skipped += 1

    stats = {
        "total_in_input": len(state_dict),
        "qwen_keys": len(qwen_weights),
        "action_keys": len(action_weights),
        "skipped_or_non_tensor": skipped,
    }
    return qwen_weights, action_weights, stats


def main():
    parser = argparse.ArgumentParser(description="Extract UnifoLM-VLA weights into qwen/action two parts.")
    parser.add_argument("--input-pt", type=Path, required=True, help="UnifoLM pytorch_model.pt 路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--save-safetensors", action="store_true", help="额外保存 qwen 权重为 safetensors")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    qwen_weights, action_weights, stats = extract_weights(args.input_pt)
    if len(qwen_weights) == 0:
        raise RuntimeError("未提取到 qwen_vl_interface 权重，请检查输入文件。")
    if len(action_weights) == 0:
        raise RuntimeError("未提取到 action_model 权重，请检查输入文件。")

    qwen_pt = args.output_dir / "qwen_vl_interface_hf_like.pt"
    action_pt = args.output_dir / "action_model_state_dict.pt"
    torch.save(qwen_weights, qwen_pt)
    torch.save(action_weights, action_pt)

    qwen_st_path = None
    if args.save_safetensors:
        if save_safetensors is None:
            raise RuntimeError("当前环境未安装 safetensors，无法使用 --save-safetensors")
        qwen_st_path = args.output_dir / "qwen_vl_interface_hf_like.safetensors"
        save_safetensors(qwen_weights, str(qwen_st_path))

    summary = {
        **stats,
        "input_pt": str(args.input_pt),
        "qwen_pt": str(qwen_pt),
        "action_pt": str(action_pt),
        "qwen_safetensors": str(qwen_st_path) if qwen_st_path else None,
        "qwen_sample_keys": list(qwen_weights.keys())[:20],
        "action_sample_keys": list(action_weights.keys())[:20],
    }
    summary_path = args.output_dir / "extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
