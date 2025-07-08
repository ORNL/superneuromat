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
except (ImportError, StopIteration):
    __version__ = "unknown"


def print_debugversions():
    """Prints the versions of the operating system and Python."""
    import platform
    import numpy
    import scipy
    print(f"SuperNeuroMAT: {__version__}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Numpy: {numpy.__version__}")
    print(f"Scipy: {scipy.__version__}")
    try:
        import numba
        print(f"Numba: {numba.__version__}")
    except ImportError:
        print("Numba: not installed or not importable")
    try:
        import importlib.metadata
        print(f"numba-cuda: {importlib.metadata.version('numba-cuda')}")
    except StopIteration:
        print("numba-cuda: no metadata")
    try:
        from numba import cuda
        print(f"cuda.is_available(): {cuda.is_available()}")
    except (ImportError, ModuleNotFoundError):
        print("CUDA: not installed or not importable")
