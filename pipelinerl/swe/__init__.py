# pipelinerl/swe/__init__.py
"""
SWE (Software Engineering) pipeline components for PipelineRL.
"""

# Explicitly import submodules to make them available as attributes
from . import rollouts
from . import agents  
from . import utils

__all__ = ['rollouts', 'agents', 'utils']