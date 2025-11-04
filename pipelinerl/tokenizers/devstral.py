"""Adapters for loading DevStral/Mistral tokenizers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

logger = logging.getLogger(__name__)

TEKKEN_FILENAME = "tekken.json"
SYSTEM_PROMPT_FILENAME = "SYSTEM_PROMPT.txt"


def is_devstral_model_name(model_name: str | None) -> bool:
    """Return True when the provided model name points to a Mistral DevStral model."""
    if not isinstance(model_name, str):
        return False
    return model_name.startswith("mistralai/")


def configure_devstral_tokenizer(llm: Any) -> Any:
    """
    Attach a custom tokenizer loader to the provided TrainableLLM instance.

    The default Tapeagents loader assumes a HuggingFace tokenizer. DevStral
    models require the `mistral-common` tokenizer instead, so we override the
    loader with a custom implementation that downloads the `tekken.json`
    definition and wraps it with a chat-friendly adapter.
    """

    def _load_tokenizer(self: Any):
        existing = getattr(self, "_tokenizer", None) or getattr(self, "tokenizer", None)
        if existing is not None:
            return existing

        model_name = getattr(self, "tokenizer_name", None) or getattr(self, "model_name", None)
        if not is_devstral_model_name(model_name):
            raise ValueError(f"configure_devstral_tokenizer applied to non-DevStral model {model_name}")

        try:
            from huggingface_hub import hf_hub_download
            from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "mistral-common >= 1.5.5 and huggingface_hub are required for DevStral tokenizers"
            ) from exc

        logger.info("Loading DevStral tokenizer for %s", model_name)
        tekken_path = Path(hf_hub_download(repo_id=model_name, filename=TEKKEN_FILENAME))

        system_prompt = ""
        try:
            system_prompt_path = Path(hf_hub_download(repo_id=model_name, filename=SYSTEM_PROMPT_FILENAME))
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
        except Exception:
            logger.debug("SYSTEM_PROMPT.txt not found for %s, proceeding without default prompt", model_name)

        mistral_tokenizer = MistralTokenizer.from_file(str(tekken_path))
        adapter = _DevStralChatTokenizerAdapter(
            tokenizer=mistral_tokenizer,
            system_prompt=system_prompt,
            assistant_cls=AssistantMessage,
            system_cls=SystemMessage,
            user_cls=UserMessage,
            request_cls=ChatCompletionRequest,
        )

        object.__setattr__(self, "_tokenizer", adapter)
        object.__setattr__(self, "tokenizer", adapter)
        object.__setattr__(self, "_tokenizer_loaded", True)
        return adapter

    try:
        object.__setattr__(llm, "load_tokenizer", MethodType(_load_tokenizer, llm))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"Failed to install DevStral tokenizer loader on {llm}: {exc}"
        ) from exc
    return llm


@dataclass
class _DevStralChatTokenizerAdapter:
    """
    Lightweight adapter that mimics the subset of the HuggingFace tokenizer API
    used across the PipelineRL SWE agents while delegating tokenization to the
    DevStral `MistralTokenizer`.
    """

    tokenizer: Any
    system_prompt: str
    assistant_cls: Any
    system_cls: Any
    user_cls: Any
    request_cls: Any

    def __post_init__(self) -> None:
        special_tokens = getattr(self.tokenizer, "special_tokens", None)
        self.eos_token_id = getattr(special_tokens, "eos_id", None)
        self.bos_token_id = getattr(special_tokens, "bos_id", None)
        self.pad_token_id = getattr(special_tokens, "pad_id", self.eos_token_id)
        self.eos_token = getattr(special_tokens, "eos", None)
        self.bos_token = getattr(special_tokens, "bos", None)
        self.pad_token = getattr(special_tokens, "pad", None)

        self._assistant_generation_kwargs: dict[str, Any] | None = None
        for flag in ("prefix", "continue_final_message"):
            try:
                self.assistant_cls(content="", **{flag: True})
            except TypeError:
                continue
            self._assistant_generation_kwargs = {"content": "", flag: True}
            break
        if self._assistant_generation_kwargs is None:
            logger.debug(
                "DevStral assistant messages do not accept 'prefix' or 'continue_final_message'; "
                "generation prompts will be skipped."
            )

    # Delegate unknown attributes to the wrapped tokenizer.
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - defensive programming
        return getattr(self.tokenizer, item)

    def apply_chat_template(
        self,
        conversation: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        add_special_tokens: bool = True,  # noqa: ARG002 - kept for API parity
    ):
        """
        Convert a chat-style conversation into either token ids or a formatted string.
        """
        messages = self._build_messages(conversation, add_generation_prompt=add_generation_prompt)
        if tokenize:
            request = self.request_cls(messages=messages)
            encoded = self.tokenizer.encode_chat_completion(request)
            return list(encoded.tokens)
        return self._render_chat(conversation, add_generation_prompt=add_generation_prompt)

    def decode(self, tokens: int | Iterable[int]) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(list(tokens))

    # Helper utilities -----------------------------------------------------------------
    def _build_messages(self, conversation: list[dict], add_generation_prompt: bool):
        messages: list[Any] = []

        has_system = any(msg.get("role") == "system" for msg in conversation)
        if not has_system and self.system_prompt:
            messages.append(self.system_cls(content=self.system_prompt))

        for msg in conversation:
            role = msg.get("role")
            content = self._normalize_content(msg.get("content"))
            if role == "system":
                messages.append(self.system_cls(content=content))
            elif role == "user":
                messages.append(self.user_cls(content=content))
            elif role == "assistant":
                messages.append(self.assistant_cls(content=content))
            else:
                raise ValueError(f"Unsupported role '{role}' in conversation: {msg}")

        if add_generation_prompt:
            if self._assistant_generation_kwargs:
                messages.append(self.assistant_cls(**self._assistant_generation_kwargs))
            else:
                logger.debug(
                    "Skipping DevStral assistant generation prompt because required flags are unsupported."
                )

        return messages

    def _normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type in {"image", "image_url"}:
                        text_parts.append("[image]")
                else:
                    text_parts.append(str(item))
            return "\n".join(filter(None, text_parts))
        return str(content)

    def _render_chat(self, conversation: list[dict], add_generation_prompt: bool) -> str:
        rendered = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = self._normalize_content(msg.get("content"))
            rendered.append(f"<|{role}|>\n{content}")
        if add_generation_prompt and self._assistant_generation_kwargs:
            rendered.append("<|assistant|>\n")
        return "\n\n".join(rendered)
