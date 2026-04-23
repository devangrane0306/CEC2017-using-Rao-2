"""
Benchmark functions module.

Contains implementations of benchmark functions,
including the CEC2017 test suite.
"""

from functions import cec2017
from functions.core import evaluate, reset_fes, get_fes, get_optimal_value

__all__ = ["cec2017", "evaluate", "reset_fes", "get_fes", "get_optimal_value"]