# pipelinerl/swe/__init__.py
"""
SWE (Software Engineering) pipeline components for PipelineRL.
"""

# Lazy imports: submodules are only loaded when first accessed.
# Eager imports caused GPU/triton failures on CPU-only nodes (scripts jobs).
__all__ = ['rollouts', 'agents', 'utils']


def __getattr__(name: str):
    if name in __all__:
        import importlib
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
