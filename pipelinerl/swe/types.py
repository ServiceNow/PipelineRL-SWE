"""Shared type definitions for the SWE pipeline."""
from typing import Literal
from pydantic import Field
from tapeagents.core import Observation


class ExpertModelAdvice(Observation):
    """Expert advice packaged as an observation for other agents."""
    kind: Literal["expert_model_advice"] = "expert_model_advice"
    advice: str = Field(description="Advice from expert model")
    original_query: str = Field(description="Query sent to expert")
    stage_name: str = Field(description="Stage this advice is for")
    
    def llm_view(self, indent: int | None = 2) -> str:
        return (
            f"=== EXPERT GUIDANCE FOR {self.stage_name.upper()} ===\n"
            f"You previously asked for guidance: {self.original_query}\n\n"
            f"Expert advice received:\n"
            f"{self.advice}\n\n"
            f"Please incorporate this expert guidance to improve your {self.stage_name} output.\n"
            f"=== END EXPERT GUIDANCE ==="
        )