from __future__ import annotations

from omegaconf import DictConfig


def get_expert_models(cfg: DictConfig) -> list:
    if hasattr(cfg, "swe"):
        return list(cfg.swe.get("expert_models") or [])
    return []


def get_performance_value_dim(cfg: DictConfig) -> int:
    # Policy value plus one per configured expert model.
    return 1 + len(get_expert_models(cfg))
