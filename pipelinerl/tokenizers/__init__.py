"""Helpers for working with non-standard tokenizer setups."""

from __future__ import annotations

from .devstral import (
    configure_devstral_tokenizer,
    is_devstral_model_name,
)

__all__ = [
    "configure_devstral_tokenizer",
    "is_devstral_model_name",
]
