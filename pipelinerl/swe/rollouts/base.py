from tapeagents.orchestrator import async_execute_agent
from tenacity import retry, stop_after_attempt, AsyncRetrying, RetryError

async def execute_agent_with_retry(agent, tape, session):
    """
    Execute agent with retry logic to handle cases where llm_call is None.
    """
    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(5)):
            with attempt:
                # Run the agent to get the localization query
                new_tape = await async_execute_agent(
                    agent,
                    tape,
                    None,  # no environment needed
                    session,
                )
                
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
                
                # If llm_call is None, raise exception to trigger retry
                if llm_call is None:
                    raise ValueError("No LLM call found in the generated tape - retrying")
                
                return new_tape, llm_call
                
    except RetryError:
        # All retries exhausted, raise the final error
        raise ValueError("No LLM call found in the generated tape after 5 retry attempts")