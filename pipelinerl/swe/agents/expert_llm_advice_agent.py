import logging
from typing import Annotated, Any, Generator, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    Prompt,
    Step,
    Tape,
    Action,
)
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

from pipelinerl.tokenizers import configure_devstral_tokenizer, is_devstral_model_name

logger = logging.getLogger(__name__)


class ExpertAdviceTask(Observation):
    """Task for getting advice from a stronger expert model."""
    kind: Literal["expert_advice_task"] = "expert_advice_task"
    query: str = Field(description="Query formulated by the smaller model")
    
    def llm_view(self, indent: int | None = 2) -> str:
        return (
            f"You are an expert AI assistant. A smaller model is seeking your guidance "
            f"on improving their work. Please provide helpful, specific, and actionable advice.\n\n"
            f"=== QUERY FROM SMALLER MODEL ===\n"
            f"{self.query}\n\n"
            f"Please provide clear, specific guidance that will help improve their output. "
            f"Focus on actionable recommendations rather than just evaluation."
        )


class ExpertAdviceResponse(Action):
    """Response containing advice from the expert model."""
    kind: Literal["expert_advice_response"] = "expert_advice_response"
    advice: str = Field(description="Advice provided by the expert model")


ExpertAdviceStep: TypeAlias = Annotated[
    ExpertAdviceResponse,
    Field(discriminator="kind"),
]

ExpertAdviceTape = Tape[
    None,
    Union[
        ExpertAdviceTask,
        ExpertAdviceResponse,
        LLMOutputParsingFailureAction,
    ],
]


class ExpertAdviceNode(StandardNode):
    """Node that provides expert advice."""
    
    max_prompt_length: int = 20000  # Larger for comprehensive advice

    # ----------------------------
    # Few-shot advice pairs
    # ----------------------------
    def _fewshot_pairs(self) -> list[dict]:
        """
        Advice demos for repair: the user sends the tagged, self-contained query;
        assistant replies with targeted nudges (no full patches), rationale, and gotchas.
        """
        pairs: list[dict] = []

        # A — Pagination
        user1 = (
            "<context>\n"
            "Full pages (n % page_size == 0) currently return 49 instead of 50.\n"
            "</context>\n"
            "<code path=\"paginator.py\">\n"
            "end = start + page_size - 1\n"
            "return items[start:end]\n"
            "</code>\n"
            "<question>\n"
            "Confirm fix: compute end = start + page_size and keep half-open slice?\n"
            "</question>"
        )
        asst1 = (
            "Yes. Guidance:\n"
            "• Compute `end = start + page_size`; keep `items[start:end]` (end exclusive).\n"
            "• Edge cases to sanity-check (conceptually): start=0; final full page; empty items; start ≥ len(items); page_size<=0.\n"
            "• Consider a brief comment explaining the half-open slice to prevent regressions."
        )
        pairs += [{"role": "user", "content": user1}, {"role": "assistant", "content": asst1}]

        # B — 'strict' allowlist + normalization
        user2 = (
            "<context>\n"
            "Allowlist rejects 'strict' while downstream reads it as a boolean.\n"
            "</context>\n"
            "<code path=\"config_schema.py\">\n"
            "ALLOWED = {\n"
            "    'timeout', 'retries', 'endpoint'\n"
            "}\n"
            "</code>\n"
            "<code path=\"parser.py\">\n"
            "if k not in ALLOWED:\n"
            "    raise ValueError(f\"unknown key '{k}'\")\n"
            "if d.get('strict'):\n"
            "    validate_strict(d)\n"
            "</code>\n"
            "<question>\n"
            "Add 'strict' and normalize to bool at load time? Accepted forms?\n"
            "</question>"
        )
        asst2 = (
            "Do this:\n"
            "• Add 'strict' to ALLOWED.\n"
            "• Normalize in load_config so all downstream code sees a boolean.\n"
            "• Accept common forms: 'true'/'false', '1'/'0', 'yes'/'no', 'on'/'off' (case-insensitive), numbers→bool.\n"
            "• Default when absent: False (unless product requires tri-state)."
        )
        pairs += [{"role": "user", "content": user2}, {"role": "assistant", "content": asst2}]

        # C — Backoff ms vs seconds
        user3 = (
            "<context>\n"
            "backoff() returns milliseconds; time.sleep expects seconds; delays 1000× too long.\n"
            "</context>\n"
            "<code path=\"client.py\">\n"
            "time.sleep(backoff(attempt))\n"
            "</code>\n"
            "<question>\n"
            "Convert at call site or change backoff() globally?\n"
            "</question>"
        )
        asst3 = (
            "Prefer minimal, safe change:\n"
            "• Convert at the call site: `time.sleep(backoff(attempt) / 1000.0)`.\n"
            "• Only change backoff() to seconds if you audit all callers and update them consistently.\n"
            "• Consider adding a docstring note near backoff() about units to prevent future misuse."
        )
        pairs += [{"role": "user", "content": user3}, {"role": "assistant", "content": asst3}]

        return pairs
    
    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        """Parse the LLM completion to extract the advice."""
        try:
            # The entire completion is the advice
            advice = completion.strip()
            
            if not advice:
                yield LLMOutputParsingFailureAction(
                    error="Empty advice generated", 
                    llm_output=completion
                )
                return
            
            yield ExpertAdviceResponse(advice=advice)
            
        except Exception as e:
            logger.info(f"Failed to parse expert advice: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse expert advice: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        task = tape.steps[0]
        assert isinstance(task, ExpertAdviceTask), f"Expected ExpertAdviceTask, got {task.__class__.__name__}"
        
        system_message = {
            "role": "system",
            "content": (
                "You are an expert software engineer. Provide concise, actionable REPAIR guidance. "
                "Nudge; do not write full patches."
            )
        }
        
        # The query now contains both context and question
        user_message = {
            "role": "user",
            "content": task.query  # Just use the query directly
        }
        
        messages = [system_message, *self._fewshot_pairs(), user_message]
        
        # Apply token limit
        prompt_token_ids = None
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'tokenizer') and agent.llm.tokenizer:
            prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
                messages, add_special_tokens=True, add_generation_prompt=True
            )
            prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            
        return Prompt(messages=messages, token_ids=prompt_token_ids)


class ExpertLLMAdviceAgent(Agent):
    """Agent that provides expert advice using a stronger model."""
    
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 20000):
        """Create an ExpertLLMAdviceAgent."""
        # Handle the llm parameter correctly for the Agent base class
        llm_map: dict[str, LLM] = {}
        llms = llm

        if isinstance(llm, dict):
            llm_map = {key: value for key, value in llm.items() if value is not None}
        elif llm is not None:
            llm_map = {"default": llm}
            llms = llm_map

        for value in llm_map.values():
            model_name = getattr(value, "tokenizer_name", None) or getattr(value, "model_name", None)
            if is_devstral_model_name(model_name):
                configure_devstral_tokenizer(value)
            
        agent = super().create(
            llms=llms,
            nodes=[
                ExpertAdviceNode(
                    name="expert_advice",
                    agent_step_cls=ExpertAdviceStep,
                    system_prompt=system_prompt if system_prompt else "",
                    max_prompt_length=max_prompt_length,
                ),
            ],
            max_iterations=1,  # Single step agent
        )
        agent.store_llm_calls = True
        def _init_tokenizer(candidate: LLM | None):
            if candidate is None:
                return
            model_name = getattr(candidate, "tokenizer_name", None) or getattr(candidate, "model_name", None)
            if is_devstral_model_name(model_name):
                configure_devstral_tokenizer(candidate)
            if hasattr(candidate, "load_tokenizer"):
                candidate.load_tokenizer()

        if hasattr(agent, "llms") and isinstance(agent.llms, dict):
            for candidate in agent.llms.values():
                _init_tokenizer(candidate)
        else:
            _init_tokenizer(getattr(agent, "llm", None))
        return agent
