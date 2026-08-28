"""Feature-fusion policy head for full-execution MDP routing."""

from __future__ import annotations

import torch

from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
)


class StructuredStatePolicy(QwenEmbeddingRouter):
    """Fuse an LM state embedding with normalized routing-state scalars.

    The base encoder and LoRA configuration are identical to the text-only
    policy. The only architectural change is a small MLP over the explicit
    routing state, concatenated with the normalized encoder representation
    immediately before the existing prediction head.
    """

    def __init__(
        self,
        *,
        state_feature_dim: int,
        state_feature_hidden_size: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if state_feature_dim < 1 or state_feature_hidden_size < 1:
            raise ValueError("Structured state dimensions must be positive")
        self.state_feature_dim = int(state_feature_dim)
        self.state_feature_hidden_size = int(state_feature_hidden_size)
        self.state_feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.state_feature_dim, self.state_feature_hidden_size),
            torch.nn.GELU(),
        )
        encoder_dim = int(self.encoder.config.hidden_size) * self.segment_count
        self.reward_head = self._make_head(
            encoder_dim + self.state_feature_hidden_size,
            self.target_dim,
            float(kwargs["dropout"]),
            int(kwargs["mlp_hidden_size"]),
        )
        self.head = self.reward_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        segment_input_ids: list[torch.Tensor] | None = None,
        segment_attention_mask: list[torch.Tensor] | None = None,
        state_features: torch.Tensor | None = None,
        cost_only: bool = False,
    ) -> tuple[torch.Tensor, None, None]:
        if cost_only:
            raise ValueError("Structured full-execution policy has no cost-only head")
        if state_features is None:
            raise ValueError("Structured full-execution policy requires state_features")
        if state_features.ndim != 2 or state_features.shape[1] != self.state_feature_dim:
            raise ValueError(
                f"Expected state_features [batch, {self.state_feature_dim}], got "
                f"{tuple(state_features.shape)}"
            )
        embeddings = self._encode_batch(
            input_ids,
            attention_mask,
            segment_input_ids,
            segment_attention_mask,
            adapter_name=self.reward_adapter_name if hasattr(self.encoder, "set_adapter") else None,
        )
        numeric = self.state_feature_encoder(
            state_features.to(device=embeddings.device, dtype=embeddings.dtype)
        )
        return self.reward_head(torch.cat([embeddings, numeric], dim=1)), None, None
