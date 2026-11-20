"""Optional frozen-policy root search. Configuration remains Torch-free."""
from .config import SearchConfig

__all__ = ['SearchConfig', 'SearchAgent', 'load_search_agent', 'register_search_checkpoint']


def __getattr__(name):
    if name in __all__[1:]:
        from . import agent
        return getattr(agent, name)
    raise AttributeError(name)
