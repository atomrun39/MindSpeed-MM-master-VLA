#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 UnifoLM action_model 权重映射并注入到 MindSpeed-MM checkpoint.

默认行为:
- 仅注入最后 PP stage 对应的 shard (post_process stage)
- 支持 model/model0/model1... 多个 state_dict 容器
- 输出 loaded/missing/unexpected/shape_mismatch 覆盖率统计
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if all(torch.is_tensor(v) for v in raw.values()):
            return raw
        for candidate in ("state_dict", "model", "module"):
            maybe = raw.get(candidate)
            if isinstance(maybe, dict) and all(torch.is_tensor(v) for v in maybe.values()):
                return maybe
    raise ValueError(f"无法从 {path} 解析 action state_dict")


def _build_action_mapping(action_sd: Dict[str, torch.Tensor], target_prefix: str) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    for src_k, v in action_sd.items():
        key = src_k
        if key.startswith("action_model."):
            key = key[len("action_model."):]
        mapped[f"{target_prefix}{key}"] = v
    return mapped


def _iter_shard_dirs(iter_dir: Path) -> List[Tuple[Path, int]]:
    """
    返回 [(mp_rank_dir, pp_rank)].
    目录命名兼容:
    - mp_rank_00
    - mp_rank_00_003
    - mp_rank_00_003_000
    """
    out: List[Tuple[Path, int]] = []
    pattern = re.compile(r"^mp_rank_(\d{2})(?:_(\d{3}))?(?:_(\d{3}))?$")
    for p in sorted(iter_dir.iterdir()):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        pp_rank = int(m.group(2)) if m.group(2) is not None else 0
        out.append((p, pp_rank))
    if not out:
        raise RuntimeError(f"在 {iter_dir} 下未找到 mp_rank_* 目录")
    return out


def _pick_iteration(mm_ckpt_root: Path, iteration: str | None) -> Path:
    if iteration:
        return mm_ckpt_root / iteration
    latest_file = mm_ckpt_root / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        latest = latest_file.read_text(encoding="utf-8").strip()
        if latest:
            return mm_ckpt_root / latest
    # 兜底: 优先 release
    if (mm_ckpt_root / "release").exists():
        return mm_ckpt_root / "release"
    raise RuntimeError("无法自动确定 iteration，请显式传 --iteration")


def _find_model_containers(ckpt_obj: Dict) -> List[str]:
    containers = []
    for k, v in ckpt_obj.items():
        if isinstance(v, dict) and k.startswith("model"):
            # 只接收 tensor state_dict
            if all(torch.is_tensor(x) for x in v.values()):
                containers.append(k)
    return containers


def _inject_into_state_dict(
    target_sd: Dict[str, torch.Tensor],
    mapped_action: Dict[str, torch.Tensor],
    strict_shape: bool,
) -> Dict[str, List[str]]:
    report = {
        "loaded": [],
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
    }

    target_action_keys = {k for k in target_sd.keys() if k.startswith("action_head.")}
    mapped_keys = set(mapped_action.keys())

    report["missing"] = sorted(target_action_keys - mapped_keys)
    report["unexpected"] = sorted(mapped_keys - target_action_keys)

    for k, src_v in mapped_action.items():
        if k in target_sd:
            tar_v = target_sd[k]
            if tuple(tar_v.shape) != tuple(src_v.shape):
                report["shape_mismatch"].append(
                    f"{k}: target={tuple(tar_v.shape)} src={tuple(src_v.shape)}"
                )
                if strict_shape:
                    continue
            target_sd[k] = src_v.to(device=tar_v.device, dtype=tar_v.dtype)
        else:
            target_sd[k] = src_v
        report["loaded"].append(k)

    return report


def _truncate(items: Iterable[str], max_items: int = 20) -> List[str]:
    items = list(items)
    if len(items) <= max_items:
        return items
    return items[:max_items] + [f"... ({len(items) - max_items} more)"]


def main():
    parser = argparse.ArgumentParser(description="Inject action_model weights into MindSpeed-MM checkpoint shards.")
    parser.add_argument("--action-state", type=Path, required=True, help="action_model_state_dict.pt 路径")
    parser.add_argument("--mm-ckpt-root", type=Path, required=True, help="MindSpeed-MM checkpoint 根目录")
    parser.add_argument("--iteration", type=str, default=None, help="例如 release / iter_0001000")
    parser.add_argument("--target-prefix", type=str, default="action_head.", help="目标 key 前缀")
    parser.add_argument("--strict-shape", action="store_true", help="shape 不一致时跳过该 key")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不写回")
    args = parser.parse_args()

    action_sd = _load_state_dict(args.action_state)
    mapped_action = _build_action_mapping(action_sd, args.target_prefix)

    iter_dir = _pick_iteration(args.mm_ckpt_root, args.iteration)
    shard_infos = _iter_shard_dirs(iter_dir)
    max_pp_rank = max(pp for _, pp in shard_infos)
    target_shards = [d for d, pp in shard_infos if pp == max_pp_rank]

    full_report = {
        "iteration_dir": str(iter_dir),
        "max_pp_rank": max_pp_rank,
        "num_target_shards": len(target_shards),
        "shards": {},
    }

    for shard_dir in target_shards:
        ckpt_file = shard_dir / "model_optim_rng.pt"
        if not ckpt_file.exists():
            full_report["shards"][str(shard_dir)] = {"error": "model_optim_rng.pt not found"}
            continue

        ckpt_obj = torch.load(ckpt_file, map_location="cpu")
        if not isinstance(ckpt_obj, dict):
            full_report["shards"][str(shard_dir)] = {"error": "checkpoint root is not dict"}
            continue

        containers = _find_model_containers(ckpt_obj)
        if not containers:
            full_report["shards"][str(shard_dir)] = {"error": "no model/modelX tensor containers found"}
            continue

        shard_report = {}
        for container_name in containers:
            rep = _inject_into_state_dict(
                ckpt_obj[container_name],
                mapped_action,
                strict_shape=args.strict_shape,
            )
            shard_report[container_name] = {
                "loaded_count": len(rep["loaded"]),
                "missing_count": len(rep["missing"]),
                "unexpected_count": len(rep["unexpected"]),
                "shape_mismatch_count": len(rep["shape_mismatch"]),
                "loaded_sample": _truncate(rep["loaded"]),
                "missing_sample": _truncate(rep["missing"]),
                "unexpected_sample": _truncate(rep["unexpected"]),
                "shape_mismatch_sample": _truncate(rep["shape_mismatch"]),
            }

        full_report["shards"][str(shard_dir)] = shard_report
        if not args.dry_run:
            torch.save(ckpt_obj, ckpt_file)

    summary_path = iter_dir / "inject_action_head_report.json"
    summary_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(full_report, indent=2, ensure_ascii=False))
    print(f"\nreport saved to: {summary_path}")


if __name__ == "__main__":
    main()
