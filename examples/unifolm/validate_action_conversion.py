#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校验 UnifoLM action 权重与 MindSpeed-MM action_head 配置/注入报告的一致性。

支持三类输入（可单独或组合）：
1) --action-state: action_model_state_dict.pt
2) --mm-model-json: mm-model.json
3) --inject-report: inject_action_head_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


FLOWMATCHING_TYPES = {"flowmatching", "flow_matching", "flowmatching_dit", "dit", "dit_l"}


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法读取 --action-state。仅分析 --inject-report 可不依赖 torch。")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if all(torch.is_tensor(v) for v in raw.values()):
            return raw
        for candidate in ("state_dict", "model", "module"):
            maybe = raw.get(candidate)
            if isinstance(maybe, dict) and all(torch.is_tensor(v) for v in maybe.values()):
                return maybe
    raise ValueError(f"无法从 {path} 解析 state_dict")


def _pick_num_heads(inner_dim: int, preferred: int = 16) -> int:
    for heads in [preferred, 12, 10, 8, 6, 5, 4, 3, 2, 1]:
        if inner_dim % heads == 0:
            return heads
    return 1


def _norm_action_key(k: str) -> str:
    return k[len("action_model."):] if k.startswith("action_model.") else k


def infer_action_profile(action_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = [_norm_action_key(k) for k in action_sd.keys()]
    shape_map = {k: list(action_sd[k2].shape) for k2, k in zip(action_sd.keys(), keys)}

    block_ids = []
    to_q_shapes: Dict[int, List[int]] = {}
    to_k_shapes: Dict[int, List[int]] = {}
    for k, shape in shape_map.items():
        m = re.match(r"model\.transformer_blocks\.(\d+)\.attn1\.to_q\.weight$", k)
        if m:
            idx = int(m.group(1))
            block_ids.append(idx)
            to_q_shapes[idx] = shape
        m = re.match(r"model\.transformer_blocks\.(\d+)\.attn1\.to_k\.weight$", k)
        if m:
            idx = int(m.group(1))
            to_k_shapes[idx] = shape

    num_layers = max(block_ids) + 1 if block_ids else 0

    inner_dim = None
    if "future_tokens.weight" in shape_map:
        inner_dim = shape_map["future_tokens.weight"][1]
    elif "position_embedding.weight" in shape_map:
        inner_dim = shape_map["position_embedding.weight"][1]
    elif to_q_shapes:
        inner_dim = next(iter(to_q_shapes.values()))[0]

    to_k_in_dims = {idx: shp[1] for idx, shp in to_k_shapes.items() if len(shp) == 2}
    unique_to_k_in_dims = sorted(set(to_k_in_dims.values()))

    inferred_interleave = False
    inferred_cross_dim = None
    if inner_dim is not None and len(unique_to_k_in_dims) >= 2:
        bigger = max(unique_to_k_in_dims)
        smaller = min(unique_to_k_in_dims)
        # 常见模式：偶数层cross-attn(大维度)，奇数层self-attn(inner_dim)
        parity_ok = True
        for idx in sorted(to_k_in_dims):
            expected = bigger if idx % 2 == 0 else smaller
            if to_k_in_dims[idx] != expected:
                parity_ok = False
                break
        inferred_interleave = parity_ok and (smaller == inner_dim)
        inferred_cross_dim = bigger
    elif len(unique_to_k_in_dims) == 1:
        inferred_cross_dim = unique_to_k_in_dims[0]

    profile = {
        "num_keys": len(keys),
        "num_layers": num_layers,
        "inner_dim": inner_dim,
        "to_q_weight_shape_example": to_q_shapes.get(0),
        "to_k_in_dims_unique": unique_to_k_in_dims,
        "interleave_inferred": inferred_interleave,
        "cross_attention_dim_inferred": inferred_cross_dim,
        "action_dim": shape_map.get("action_decoder.layer2.weight", [0, 0])[0] if "action_decoder.layer2.weight" in shape_map else None,
        "state_dim": shape_map.get("state_encoder.layer1.weight", [0, 0])[1] if "state_encoder.layer1.weight" in shape_map else None,
        # future_tokens 通常对应视觉目标token数量，而非 action_horizon
        "num_target_vision_tokens": shape_map.get("future_tokens.weight", [0, 0])[0] if "future_tokens.weight" in shape_map else None,
    }
    return profile


def infer_effective_mm_action_cfg(mm_model: Dict[str, Any]) -> Dict[str, Any]:
    text_hidden_size = int(_cfg_get(_cfg_get(mm_model, "text_decoder", {}), "hidden_size", 0) or 0)
    action_cfg = _cfg_get(mm_model, "action_head", {}) or {}
    action_type = str(_cfg_get(action_cfg, "type", "mlp")).lower()
    diffusion_cfg = _cfg_get(action_cfg, "diffusion_model_cfg", {}) or {}

    input_embedding_dim = int(
        _cfg_get(action_cfg, "input_embedding_dim",
                 _cfg_get(action_cfg, "hidden_size", text_hidden_size)) or 0
    )
    num_heads = int(_cfg_get(diffusion_cfg, "num_attention_heads", _pick_num_heads(input_embedding_dim, 16)) or 1)
    head_dim = int(_cfg_get(diffusion_cfg, "attention_head_dim", input_embedding_dim // max(1, num_heads)) or 1)
    inner_dim = num_heads * head_dim

    effective = {
        "enabled": bool(_cfg_get(action_cfg, "enable", False)),
        "type": action_type,
        "flowmatching_mode": action_type in FLOWMATCHING_TYPES,
        "num_layers": int(_cfg_get(diffusion_cfg, "num_layers", 8)),
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "inner_dim": inner_dim,
        "interleave_self_attention": bool(_cfg_get(diffusion_cfg, "interleave_self_attention", False)),
        "cross_attention_dim": int(_cfg_get(diffusion_cfg, "cross_attention_dim", text_hidden_size) or 0),
        "action_dim": int(_cfg_get(action_cfg, "action_dim", 0) or 0),
        "state_dim": int(_cfg_get(action_cfg, "state_dim", 0) or 0),
        "num_target_vision_tokens": int(_cfg_get(action_cfg, "num_target_vision_tokens", 0) or 0),
        "action_horizon": int(_cfg_get(action_cfg, "action_horizon", _cfg_get(action_cfg, "num_queries", 0)) or 0),
        "text_hidden_size": text_hidden_size,
    }
    return effective


def compare_profile_and_cfg(profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    checks = []

    def add_check(name: str, src, tar, strict: bool = True):
        ok = (src == tar)
        level = "ok" if ok else ("error" if strict else "warn")
        checks.append({"name": name, "source": src, "target": tar, "status": level})

    add_check("num_layers", profile.get("num_layers"), cfg.get("num_layers"))
    add_check("inner_dim", profile.get("inner_dim"), cfg.get("inner_dim"))
    add_check("interleave_self_attention", profile.get("interleave_inferred"), cfg.get("interleave_self_attention"))
    add_check("cross_attention_dim", profile.get("cross_attention_dim_inferred"), cfg.get("cross_attention_dim"), strict=False)
    add_check("action_dim", profile.get("action_dim"), cfg.get("action_dim"))
    add_check("state_dim", profile.get("state_dim"), cfg.get("state_dim"))
    add_check("num_target_vision_tokens", profile.get("num_target_vision_tokens"), cfg.get("num_target_vision_tokens"))

    errors = [c for c in checks if c["status"] == "error"]
    warns = [c for c in checks if c["status"] == "warn"]
    oks = [c for c in checks if c["status"] == "ok"]
    return {"checks": checks, "error_count": len(errors), "warn_count": len(warns), "ok_count": len(oks)}


def analyze_inject_report(report: Dict[str, Any]) -> Dict[str, Any]:
    shard_count = 0
    model_count = 0
    total_loaded = 0
    total_missing = 0
    total_unexpected = 0
    total_shape_mismatch = 0
    examples = {"missing": [], "unexpected": [], "shape_mismatch": []}

    shards = report.get("shards", {})
    for _, shard_info in shards.items():
        if not isinstance(shard_info, dict):
            continue
        shard_count += 1
        for model_name, model_info in shard_info.items():
            if not isinstance(model_info, dict):
                continue
            if "loaded_count" not in model_info:
                continue
            model_count += 1
            total_loaded += int(model_info.get("loaded_count", 0))
            total_missing += int(model_info.get("missing_count", 0))
            total_unexpected += int(model_info.get("unexpected_count", 0))
            total_shape_mismatch += int(model_info.get("shape_mismatch_count", 0))
            for k, sample_key in [("missing", "missing_sample"), ("unexpected", "unexpected_sample"), ("shape_mismatch", "shape_mismatch_sample")]:
                if not examples[k]:
                    sample = model_info.get(sample_key, [])
                    if isinstance(sample, list):
                        examples[k] = sample[:10]

    conclusion = "ok"
    if total_shape_mismatch > 0:
        conclusion = "error"
    elif total_missing > 0:
        conclusion = "warn"

    return {
        "iteration_dir": report.get("iteration_dir"),
        "max_pp_rank": report.get("max_pp_rank"),
        "scanned_shards": shard_count,
        "scanned_models": model_count,
        "total_loaded": total_loaded,
        "total_missing": total_missing,
        "total_unexpected": total_unexpected,
        "total_shape_mismatch": total_shape_mismatch,
        "examples": examples,
        "conclusion": conclusion,
    }


def build_recommended_patch(profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    inner = profile.get("inner_dim") or cfg.get("inner_dim") or 0
    heads = cfg.get("num_attention_heads") or _pick_num_heads(int(inner), 16)
    head_dim = int(inner) // int(heads) if heads else 0
    return {
        "action_head": {
            "type": "flowmatching",
            "action_dim": profile.get("action_dim"),
            # action_horizon 无法从权重直接可靠推断，保留目标配置值
            "action_horizon": cfg.get("action_horizon"),
            "state_dim": profile.get("state_dim"),
            "num_target_vision_tokens": profile.get("num_target_vision_tokens"),
            "input_embedding_dim": inner,
            "diffusion_model_cfg": {
                "num_layers": profile.get("num_layers"),
                "num_attention_heads": heads,
                "attention_head_dim": head_dim,
                "interleave_self_attention": profile.get("interleave_inferred"),
                "cross_attention_dim": profile.get("cross_attention_dim_inferred"),
            }
        }
    }


def print_human_readable(result: Dict[str, Any]):
    print("\n=== Action Weight Compatibility Check ===")
    if result.get("action_profile"):
        print("[source] action_profile:")
        print(json.dumps(result["action_profile"], ensure_ascii=False, indent=2))
    if result.get("effective_mm_action_cfg"):
        print("\n[target] effective_mm_action_cfg:")
        print(json.dumps(result["effective_mm_action_cfg"], ensure_ascii=False, indent=2))
    if result.get("config_compare"):
        cmp_res = result["config_compare"]
        print(f"\n[compare] ok={cmp_res['ok_count']} warn={cmp_res['warn_count']} error={cmp_res['error_count']}")
        for c in cmp_res["checks"]:
            print(f"  - {c['status']:>5} | {c['name']}: source={c['source']} target={c['target']}")
    if result.get("inject_report_analysis"):
        ana = result["inject_report_analysis"]
        print("\n[inject_report]")
        print(json.dumps(ana, ensure_ascii=False, indent=2))
    if result.get("recommended_action_head_patch"):
        print("\n[recommended_action_head_patch]")
        print(json.dumps(result["recommended_action_head_patch"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Validate action config vs source weights and injection report.")
    parser.add_argument("--action-state", type=Path, default=None, help="action_model_state_dict.pt")
    parser.add_argument("--mm-model-json", type=Path, default=None, help="mm-model.json")
    parser.add_argument("--inject-report", type=Path, default=None, help="inject_action_head_report.json")
    parser.add_argument("--save-json", type=Path, default=None, help="可选：将分析结果保存到json")
    args = parser.parse_args()

    if not args.action_state and not args.mm_model_json and not args.inject_report:
        raise ValueError("至少传入一个输入: --action-state / --mm-model-json / --inject-report")

    result: Dict[str, Any] = {}
    profile = None
    cfg = None

    if args.action_state:
        action_sd = _load_state_dict(args.action_state)
        profile = infer_action_profile(action_sd)
        result["action_profile"] = profile
        result["action_state_path"] = str(args.action_state)

    if args.mm_model_json:
        mm_model = _load_json(args.mm_model_json)
        cfg = infer_effective_mm_action_cfg(mm_model)
        result["effective_mm_action_cfg"] = cfg
        result["mm_model_json_path"] = str(args.mm_model_json)

    if profile and cfg:
        result["config_compare"] = compare_profile_and_cfg(profile, cfg)
        result["recommended_action_head_patch"] = build_recommended_patch(profile, cfg)

    if args.inject_report:
        report = _load_json(args.inject_report)
        result["inject_report_analysis"] = analyze_inject_report(report)
        result["inject_report_path"] = str(args.inject_report)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_human_readable(result)


if __name__ == "__main__":
    main()
