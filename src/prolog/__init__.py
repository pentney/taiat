"""
Taiat Path Planner package.

This package provides Prolog-based path planning functionality for determining
optimal execution sequences in Taiat queries.
"""

from .taiat_path_planner import (
    TaiatPathPlanner,
    plan_taiat_path,
    plan_taiat_path_global,
    get_global_planner,
)

__all__ = [
    "TaiatPathPlanner",
    "plan_taiat_path",
    "plan_taiat_path_global",
    "get_global_planner",
]

__version__ = "2.0.0"
