import logging
from typing import Any

from accelerate import Accelerator

logger = logging.getLogger(__name__)

# step_scheduler_with_optimizer=False prevents the scheduler
# from being stepped multiple times in the multi-gpu setting.
# (The default behavior in AcceleratedScheduler when split_batches=False is to
#   step() "num_processes" times, because they expect the lr schedule to
#   depend on processed samples/epochs, not completed_steps)

_accelerator = None
_accelerator_init_kwargs: dict[str, Any] | None = None


def configure_accelerator(**kwargs):
    global _accelerator, _accelerator_init_kwargs
    init_kwargs = {"step_scheduler_with_optimizer": False, **kwargs}
    if _accelerator is None:
        _accelerator = Accelerator(**init_kwargs)
        _accelerator_init_kwargs = init_kwargs
    elif kwargs and _accelerator_init_kwargs != init_kwargs:
        raise RuntimeError(
            f"Accelerator was already initialized with {_accelerator_init_kwargs}, cannot reconfigure with {init_kwargs}"
        )
    return _accelerator


def get_accelerator():
    global _accelerator
    if _accelerator is None:
        return configure_accelerator()
    return _accelerator


def accelerator_is_initialized() -> bool:
    return _accelerator is not None
