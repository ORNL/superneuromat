import sys
from .src.superneuromat.neuromorphicmodel import NeuromorphicModel

__all__ = ["NeuromorphicModel"]

if "SUPPRESS_SUPERNEUROMAT_WARNINGS" not in sys.path:
    print(
        "WARNING: Importing superneuromat as relative module is deprecated and will be removed in the next version.",
        file=sys.stderr
    )
