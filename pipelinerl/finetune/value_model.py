import os
import torch
import torch.nn as nn
import json
from dataclasses import dataclass
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple, Union
from transformers import AutoModelForCausalLM
from .context import get_accelerator, logger


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    """
    Output type for causal language models with an additional value head.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modeling head.
        value (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Value predictions from the value head.
        performance_value (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_solvers)`):
            Performance predictions for policy + experts.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Contains cached key/value states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states of the model at the output of each layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after the attention softmax.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    value: torch.FloatTensor = None
    performance_value: torch.FloatTensor | None = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ValueHead(nn.Module):
    """Scalar value head for PPO training."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.output = nn.Linear(hidden_size, 1)
        torch.manual_seed(42)  # For reproducibility
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = self.output(hidden_states).squeeze(-1)  # (batch_size, sequence_length)
        return values


class PerformanceValueHead(nn.Module):
    """Vector performance head for policy + expert rewards."""

    def __init__(
        self,
        hidden_size: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dims = [int(dim) for dim in (hidden_dims or []) if int(dim) > 0]
        self.hidden_dims = hidden_dims
        self.output_dim = int(output_dim)
        self.activation_name = activation

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._make_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        torch.manual_seed(42)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1e-3)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        normalized = str(name).lower()
        if normalized == "gelu":
            return nn.GELU()
        if normalized == "relu":
            return nn.ReLU()
        if normalized == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported performance head activation: {name}")

    def config_dict(self) -> dict:
        return {
            "hidden_dims": list(self.hidden_dims),
            "output_dim": int(self.output_dim),
            "activation": self.activation_name,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)  # (batch_size, sequence_length, output_dim)


class AutoModelForCausalLMWithValueHead(nn.Module):
    """
    A wrapper around a causal language model that adds a value head for PPO training.
    """

    def __init__(
        self,
        pretrained_model,
        performance_value_dim: int = 1,
        performance_value_hidden_dims: Optional[list[int]] = None,
        performance_value_activation: str = "gelu",
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = self.config.hidden_size

        # Initialize value head
        self.value_head = ValueHead(hidden_size)
        # Initialize performance head (policy + experts)
        self.performance_value_head = PerformanceValueHead(
            hidden_size,
            performance_value_dim,
            hidden_dims=performance_value_hidden_dims,
            activation=performance_value_activation,
        )
        self.performance_value_dim = performance_value_dim
        self.performance_value_hidden_dims = list(performance_value_hidden_dims or [])
        self.performance_value_activation = performance_value_activation

        # Copy relevant attributes from the pretrained model
        self.main_input_name = pretrained_model.main_input_name

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """
        Forward pass that computes both language modeling outputs and value predictions.
        """

        # Get outputs from the base model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]

        # Compute values
        values = self.value_head(hidden_states)
        performance_values = self.performance_value_head(hidden_states)

        return CausalLMOutputWithValue(
            loss=outputs.loss,
            logits=outputs.logits,
            value=values,
            performance_value=performance_values,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.pretrained_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        safe_serialization: bool = False,
        **kwargs,
    ):
        """Save model and value head separately."""
        import os
        
        if state_dict is None:
            state_dict = self.state_dict()
        
        # Extract pretrained model and value head state dicts
        pretrained_model_state_dict = {}
        value_head_state_dict = {}
        performance_head_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("value_head."):
                # Remove the "value_head." prefix
                new_key = key[len("value_head."):]
                value_head_state_dict[new_key] = value
            elif key.startswith("performance_value_head."):
                new_key = key[len("performance_value_head."):]
                performance_head_state_dict[new_key] = value
            elif key.startswith("pretrained_model."):
                # Remove the "pretrained_model." prefix
                new_key = key[len("pretrained_model."):]
                pretrained_model_state_dict[new_key] = value
            else:
                raise ValueError(
                    f"Unexpected key in state dict: {key}. "
                    "Expected keys should start with 'value_head.', 'performance_value_head.' or 'pretrained_model.'."
                )
        
        # Save the pretrained model which can be easily loaded by vllm, etc.
        self.pretrained_model.save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=pretrained_model_state_dict,
            save_function=save_function,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        
        # Save value head separately
        if is_main_process:
            value_head_path = os.path.join(save_directory, "value_head.pt")
            save_function(value_head_state_dict, value_head_path)
            logger.info(f"Saved value head to {value_head_path}")
            performance_head_path = os.path.join(save_directory, "performance_value_head.pt")
            save_function(performance_head_state_dict, performance_head_path)
            logger.info(f"Saved performance value head to {performance_head_path}")
            performance_head_config_path = os.path.join(save_directory, "performance_value_head_config.json")
            with open(performance_head_config_path, "w") as handle:
                json.dump(self.performance_value_head.config_dict(), handle, indent=2)
            logger.info(f"Saved performance value head config to {performance_head_config_path}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a model with value head from pretrained weights."""

        logger.info(f"Loading pretrained model from {pretrained_model_name_or_path}...")

        performance_value_dim = kwargs.pop("performance_value_dim", None)
        performance_value_hidden_dims = kwargs.pop("performance_value_hidden_dims", None)
        performance_value_activation = kwargs.pop("performance_value_activation", None)
        performance_head_path = os.path.join(pretrained_model_name_or_path, "performance_value_head.pt")
        performance_head_config_path = os.path.join(pretrained_model_name_or_path, "performance_value_head_config.json")
        if os.path.exists(performance_head_config_path):
            with open(performance_head_config_path) as handle:
                performance_head_config = json.load(handle)
        else:
            performance_head_config = {}

        if performance_value_hidden_dims is None:
            performance_value_hidden_dims = performance_head_config.get("hidden_dims")
        if performance_value_activation is None:
            performance_value_activation = performance_head_config.get("activation", "gelu")
        if performance_value_dim is None and os.path.exists(performance_head_path):
            state = torch.load(performance_head_path, map_location="cpu")
            linear_weights = [
                value for key, value in state.items() if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2
            ]
            if linear_weights:
                performance_value_dim = linear_weights[-1].shape[0]
        if performance_value_dim is None:
            performance_value_dim = 1
        if performance_value_hidden_dims is None:
            performance_value_hidden_dims = []

        # Load the base model
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Create the model with value head
        model = cls(
            pretrained_model,
            performance_value_dim=performance_value_dim,
            performance_value_hidden_dims=performance_value_hidden_dims,
            performance_value_activation=performance_value_activation,
        )

        # Try to load value head weights if they exist
        value_head_path = os.path.join(pretrained_model_name_or_path, "value_head.pt")
        if os.path.exists(value_head_path):
            value_head_state_dict = torch.load(value_head_path, map_location="cpu")
            model.value_head.load_state_dict(value_head_state_dict)
        if os.path.exists(performance_head_path):
            performance_head_state_dict = torch.load(performance_head_path, map_location="cpu")
            try:
                model.performance_value_head.load_state_dict(performance_head_state_dict)
            except RuntimeError as exc:
                logger.warning(
                    "Skipping performance_value_head load from %s due to shape/config mismatch: %s",
                    performance_head_path,
                    exc,
                )

        return model

    @property
    def device(self):
        """Get the device of the model."""
        return self.pretrained_model.device

    @property
    def dtype(self):
        """Get the dtype of the model."""
        return self.pretrained_model.dtype

    def __getattr__(self, name):
        """Forward attribute access to the pretrained model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pretrained_model, name)
