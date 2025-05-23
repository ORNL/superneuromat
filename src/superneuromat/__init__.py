from .neuromorphicmodel import SNN
from .accessor_classes import Neuron, Synapse
from .util import is_intlike, pretty_spike_train, print_spike_train, getenvbool

__all__ = [
    "SNN",
    "Neuron",
    "Synapse",
    "is_intlike",
    "pretty_spike_train",
    "print_spike_train",
    "getenvbool",
]

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("superneuromat")
except ImportError:
    __version__ = "unknown"
