"""
TaiatManager with Haskell Path Planner Integration.

This module provides TaiatManager that uses the Haskell
path planner to determine optimal execution sequences for Taiat queries.
"""

from functools import partial
from typing import Callable, Optional, List, Dict, Any
from taiat.base import TAIAT_TERMINAL_NODE, AgentData, TaiatQuery, START_NODE
from taiat.builder import AgentGraphNode, AgentGraphNodeSet, State
from taiat.metrics import TaiatMetrics
from threading import RLock
import time
from langgraph.graph import START

# Import the Haskell path planner
from haskell.path_planner_interface import plan_path


class TaiatManager:
    """
    TaiatManager that uses Haskell path planning for optimal agent selection.

    This manager combines the existing TaiatManager functionality with the Haskell
    path planner to provide better execution planning and agent selection.
    """

    def __init__(
        self,
        node_set: AgentGraphNodeSet,
        reverse_plan_edges: dict[str, list[str]],
        wait_interval: int = 5,
        verbose: bool = False,
        metrics: Optional[TaiatMetrics] = None,
        use_haskell_planning: bool = True,
        fallback_to_original: bool = True,
    ):
        """
        Initialize the TaiatManager.

        Args:
            node_set: Set of available agent nodes
            reverse_plan_edges: Reverse dependency edges
            wait_interval: Interval between status checks
            verbose: Enable verbose output
            metrics: Metrics collection object
            use_haskell_planning: Whether to use Haskell path planning
            fallback_to_original: Whether to fall back to original planning if Haskell fails
        """
        self.interval = wait_interval
        self.node_set = node_set
        self.reverse_plan_edges = reverse_plan_edges
        self.use_haskell_planning = use_haskell_planning
        self.fallback_to_original = fallback_to_original

        # Initialize Haskell path planner if enabled
        if self.use_haskell_planning:
            try:
                # Test the Haskell planner
                test_result = plan_path(node_set, [])
                self.haskell_available = True
                if self.verbose:
                    print("Haskell path planner initialized successfully")
            except Exception as e:
                print(f"Warning: Haskell path planner not available: {e}")
                self.haskell_available = False
        else:
            self.haskell_available = False

        # Build reverse graph to find next node to run (original logic)
        self.plan_edges = {k.name: [] for k in self.node_set.nodes + [START_NODE]}
        for node, neighbors in reverse_plan_edges.items():
            for neighbor in neighbors:
                self.plan_edges[neighbor].append(node)

        # Initialize status tracking
        self.output_status = {k.name: "pending" for k in self.node_set.nodes}
        self.output_status[START] = "running"
        self.output_status[TAIAT_TERMINAL_NODE] = "pending"

        # Execution path from Haskell planner
        self.planned_execution_path = []
        self.current_path_index = 0

        self.status_lock = RLock()
        self.verbose = verbose
        self.metrics = metrics

    def plan_execution_with_haskell(self, desired_outputs: List[AgentData]) -> bool:
        """
        Plan execution path using Haskell path planner.

        Args:
            desired_outputs: List of desired outputs to produce

        Returns:
            True if planning was successful, False otherwise
        """
        if not self.haskell_available:
            return False

        try:
            execution_path = plan_taiat_path(self.node_set, desired_outputs)
            if execution_path is not None:
                self.planned_execution_path = execution_path
                self.current_path_index = 0
                if self.verbose:
                    print(f"Haskell planned execution path: {execution_path}")
                return True
            else:
                if self.verbose:
                    print(
                        "Haskell path planning failed, falling back to original method"
                    )
                return False
        except Exception as e:
            if self.verbose:
                print(f"Error in Haskell path planning: {e}")
            return False

    def validate_outputs_with_haskell(self, desired_outputs: List[AgentData]) -> bool:
        """
        Validate that desired outputs can be produced using Haskell planner.

        Args:
            desired_outputs: List of desired outputs to validate

        Returns:
            True if all outputs can be produced, False otherwise
        """
        if not self.haskell_available:
            return False

        try:
            # Try to plan a path - if it succeeds, the outputs are valid
            execution_path = plan_path(self.node_set, desired_outputs)
            return execution_path is not None
        except Exception as e:
            if self.verbose:
                print(f"Error in Haskell output validation: {e}")
            return False

    def get_available_outputs_with_haskell(self) -> List[str]:
        """
        Get all available outputs using Haskell planner.

        Returns:
            List of available output names
        """
        if not self.haskell_available:
            return []

        try:
            # Collect all outputs from all nodes
            available_outputs = []
            for node in self.node_set.nodes:
                for output in node.outputs:
                    available_outputs.append(output.name)
            return list(set(available_outputs))  # Remove duplicates
        except Exception as e:
            if self.verbose:
                print(f"Error getting available outputs with Haskell: {e}")
            return []

    def make_router_function(self, node) -> Callable:
        """Create a router function for the given node."""
        return partial(
            self.enhanced_router_function,
            node=node,
        )

    def enhanced_router_function(
        self,
        state: State,
        node: str,
    ) -> str | None:
        """
        Enhanced router function that uses Haskell planning when available.

        This function combines Haskell path planning with the original dependency-based
        routing logic for optimal agent selection.
        """
        with self.status_lock:
            self.output_status[node] = "done"

        if self.verbose:
            print(f"Node {node} complete, looking for next node")

        # If we have a Haskell-planned path, try to follow it
        if self.planned_execution_path and self.current_path_index < len(
            self.planned_execution_path
        ):
            next_planned_node = self.planned_execution_path[self.current_path_index]

            # Check if the planned node is ready to run
            if self._is_node_ready_to_run(next_planned_node):
                self.current_path_index += 1
                with self.status_lock:
                    if self.verbose:
                        print(f"Running planned node: {next_planned_node}")
                    self.output_status[next_planned_node] = "running"
                if self.metrics is not None:
                    self.metrics[next_planned_node]["calls"] += 1
                return next_planned_node

        # Fall back to original routing logic
        if self.fallback_to_original:
            return self._original_router_logic(node)

        # If no fallback and no more planned nodes, we're done
        return TAIAT_TERMINAL_NODE

    def _is_node_ready_to_run(self, node_name: str) -> bool:
        """
        Check if a node is ready to run based on its dependencies.

        Args:
            node_name: Name of the node to check

        Returns:
            True if the node is ready to run, False otherwise
        """
        # Find the node
        node = next((n for n in self.node_set.nodes if n.name == node_name), None)
        if node is None:
            return False

        # Check if all dependencies are satisfied
        for input_data in node.inputs:
            dependency_satisfied = False

            # Look for a node that produces this input
            for other_node in self.node_set.nodes:
                for output in other_node.outputs:
                    if (
                        output.name == input_data.name
                        and self.output_status[other_node.name] == "done"
                    ):
                        # Check if parameters match (input parameters should be a subset of output parameters)
                        if (
                            not input_data.parameters
                            or input_data.parameters.items()
                            <= output.parameters.items()
                        ):
                            dependency_satisfied = True
                            break
                if dependency_satisfied:
                    break

            if not dependency_satisfied:
                return False

        return True

    def _original_router_logic(self, node: str) -> str | None:
        """
        Original router logic as fallback when Haskell planning is not available.

        Args:
            node: Name of the completed node

        Returns:
            Name of the next node to run, or None if no nodes are ready
        """
        # Find nodes that are ready to run
        ready_nodes = []
        for node_name in self.plan_edges[node]:
            if self.output_status[node_name] == "pending":
                # Check if all dependencies are satisfied
                if self._is_node_ready_to_run(node_name):
                    ready_nodes.append(node_name)

        if ready_nodes:
            # Choose the first ready node (could be enhanced with priority logic)
            next_node = ready_nodes[0]
            with self.status_lock:
                if self.verbose:
                    print(f"Running node: {next_node}")
                self.output_status[next_node] = "running"
            if self.metrics is not None:
                self.metrics[next_node]["calls"] += 1
            return next_node

        # Check if we're done (all nodes that depend on the current node are done)
        all_done = True
        for node_name in self.plan_edges[node]:
            if self.output_status[node_name] != "done":
                all_done = False
                break

        if all_done:
            return TAIAT_TERMINAL_NODE

        return None

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics including Haskell planning information.

        Returns:
            Dictionary with execution statistics
        """
        stats = {
            "total_nodes": len(self.node_set.nodes),
            "completed_nodes": sum(
                1 for status in self.output_status.values() if status == "done"
            ),
            "pending_nodes": sum(
                1 for status in self.output_status.values() if status == "pending"
            ),
            "running_nodes": sum(
                1 for status in self.output_status.values() if status == "running"
            ),
            "haskell_available": self.haskell_available,
            "planned_execution_path": self.planned_execution_path,
            "current_path_index": self.current_path_index,
        }

        if self.metrics is not None:
            stats["metrics"] = self.metrics.get_statistics()

        return stats


def create_manager(
    node_set: AgentGraphNodeSet,
    reverse_plan_edges: dict[str, list[str]],
    desired_outputs: List[AgentData] = None,
    **kwargs,
) -> TaiatManager:
    """
    Create a TaiatManager with Haskell path planning.

    Args:
        node_set: Set of available agent nodes
        reverse_plan_edges: Reverse dependency edges
        desired_outputs: Desired outputs to plan for (optional)
        **kwargs: Additional arguments to pass to TaiatManager

    Returns:
        Configured TaiatManager instance
    """
    manager = TaiatManager(
        node_set=node_set, reverse_plan_edges=reverse_plan_edges, **kwargs
    )

    # Plan execution with Haskell if desired outputs are provided
    if desired_outputs and manager.use_haskell_planning:
        manager.plan_execution_with_haskell(desired_outputs)

    return manager


def validate_query_with_haskell(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> bool:
    """
    Validate a query using Haskell path planner.

    Args:
        node_set: Set of available agent nodes
        desired_outputs: List of desired outputs to validate

    Returns:
        True if the query can be satisfied, False otherwise
    """
    try:
        execution_path = plan_path(node_set, desired_outputs)
        return execution_path is not None
    except Exception:
        return False


def plan_query_with_haskell(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> Optional[List[str]]:
    """
    Plan a query using Haskell path planner.

    Args:
        node_set: Set of available agent nodes
        desired_outputs: List of desired outputs to produce

    Returns:
        List of node names in execution order, or None if planning fails
    """
    try:
        return plan_path(node_set, desired_outputs)
    except Exception:
        return None
