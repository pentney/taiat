"""
Global Optimized Prolog Path Planner package for Taiat.

This package provides global optimized Prolog-based path planning functionality for determining
optimal execution sequences in Taiat queries.
"""

from .optimized_prolog_interface import (
    OptimizedPrologPathPlanner,
    plan_taiat_path_optimized,
    plan_taiat_path_global_optimized,
    get_global_optimized_planner,
)

__all__ = [
    "OptimizedPrologPathPlanner",
    "plan_taiat_path_optimized",
    "plan_taiat_path_global_optimized",
    "get_global_optimized_planner",
]

__version__ = "2.0.0"
