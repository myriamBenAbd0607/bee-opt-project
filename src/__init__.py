"""
BEE-OPT: Artificial Bee Colony Algorithm Package
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[Your Email]"

from .abc_algorithm import ArtificialBeeColony
from .benchmark_functions import (
    sphere,
    rosenbrock,
    rastrigin,
    ackley,
    get_function_bounds,
    get_function_by_name
)

__all__ = [
    "ArtificialBeeColony",
    "sphere",
    "rosenbrock",
    "rastrigin",
    "ackley",
    "get_function_bounds",
    "get_function_by_name"
]