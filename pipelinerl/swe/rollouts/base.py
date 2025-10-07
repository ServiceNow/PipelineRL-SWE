import logging
from tapeagents.orchestrator import async_execute_agent
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from tapeagents.core import Action
from tapeagents.environment import AsyncEnvironment
from tapeagents.agent import TapeType

logger = logging.getLogger(__name__)

async def execute_agent_with_retry(agent, tape, session):
    """Execute agent with retry logic to handle cases where llm_call is None."""
    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(5)):
            with attempt:
                new_tape = await async_execute_agent(agent, tape, EmptyAsyncEnvironment(), session, max_loops=1)
                
                # Extract LLM call and validate it exists
                llm_call = None
                for step in new_tape.steps:
                    if (
                        hasattr(step, 'metadata') and 
                        step.metadata and 
                        hasattr(step.metadata, 'other') and
                        "llm_call" in step.metadata.other and
                        step.metadata.other["llm_call"] is not None
                    ):
                        llm_call = step.metadata.other["llm_call"]
                        break
                
                if llm_call is None:
                    raise ValueError("No LLM call found in the generated tape - retrying")
                
                return new_tape, llm_call
                
    except RetryError:
        raise ValueError("No LLM call found in the generated tape after 5 retry attempts")

class EmptyAsyncEnvironment(AsyncEnvironment):
    async def ainitialize(self):
        pass

    async def areact(self, tape: TapeType) -> TapeType:
        return tape # no op

    async def astep(self, action: Action):
        pass

    async def areset(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def react(self, tape: TapeType) -> TapeType:
        return tape