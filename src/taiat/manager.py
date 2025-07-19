"""
TaiatManager with Haskell Path Planner Integration.

This module provides TaiatManager that uses the Haskell
path planner to determine optimal execution sequences for Taiat queries.
"""

from functools import partial
from typing import Callable, Optional, List, Dict, Any, Set
from taiat.base import TAIAT_TERMINAL_NODE, AgentData, TaiatQuery, START_NODE
from taiat.builder import AgentGraphNode, AgentGraphNodeSet, State
from taiat.metrics import TaiatMetrics
from threading import RLock
import time


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
        max_retries: int = 3,
        enable_alternative_paths: bool = True,
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
            max_retries: Maximum number of retries for failed nodes
            enable_alternative_paths: Whether to enable alternative path selection
        """
        self.interval = wait_interval
        self.node_set = node_set
        self.reverse_plan_edges = reverse_plan_edges
        self.use_haskell_planning = use_haskell_planning
        self.fallback_to_original = fallback_to_original
        self.max_retries = max_retries
        self.enable_alternative_paths = enable_alternative_paths

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
        self.output_status[START_NODE.name] = "running"
        self.output_status[TAIAT_TERMINAL_NODE] = "pending"

        # Failure tracking
        self.failed_nodes: Set[str] = set()
        self.node_retry_counts: Dict[str, int] = {}
        self.failed_outputs: Set[str] = set()  # Track outputs that couldn't be produced

        # Execution path from Haskell planner
        self.planned_execution_path = []
        self.current_path_index = 0
        self.alternative_paths: List[
            List[str]
        ] = []  # Store alternative execution paths

        self.status_lock = RLock()
        self.verbose = verbose
        self.metrics = metrics

    def mark_node_failed(self, node_name: str, error_message: str = "") -> None:
        """
        Mark a node as failed and update failure tracking.

        Args:
            node_name: Name of the failed node
            error_message: Optional error message describing the failure
        """
        with self.status_lock:
            self.failed_nodes.add(node_name)
            self.output_status[node_name] = "failed"

            # Increment retry count
            self.node_retry_counts[node_name] = (
                self.node_retry_counts.get(node_name, 0) + 1
            )

            # Mark outputs as failed if this was the only producer
            node = next((n for n in self.node_set.nodes if n.name == node_name), None)
            if node:
                for output in node.outputs:
                    # Check if this is the only producer of this output
                    other_producers = [
                        n
                        for n in self.node_set.nodes
                        if n.name != node_name
                        and any(o.name == output.name for o in n.outputs)
                    ]
                    if not other_producers:
                        self.failed_outputs.add(output.name)

            if self.verbose:
                print(
                    f"Node {node_name} marked as failed. Retry count: {self.node_retry_counts[node_name]}"
                )
                if error_message:
                    print(f"Error: {error_message}")

    def can_retry_node(self, node_name: str) -> bool:
        """
        Check if a node can be retried.

        Args:
            node_name: Name of the node to check

        Returns:
            True if the node can be retried, False otherwise
        """
        retry_count = self.node_retry_counts.get(node_name, 0)
        return retry_count < self.max_retries

    def reset_node_failure(self, node_name: str) -> None:
        """
        Reset failure status for a node (e.g., after successful retry).

        Args:
            node_name: Name of the node to reset
        """
        with self.status_lock:
            if node_name in self.failed_nodes:
                self.failed_nodes.remove(node_name)
            self.output_status[node_name] = "pending"
            if self.verbose:
                print(f"Node {node_name} failure status reset")

    def find_alternative_paths(
        self, desired_outputs: List[AgentData]
    ) -> List[List[str]]:
        """
        Find alternative execution paths that avoid failed nodes.

        Args:
            desired_outputs: List of desired outputs to produce

        Returns:
            List of alternative execution paths
        """
        if not self.enable_alternative_paths or not self.haskell_available:
            return []

        try:
            # Use the new Haskell function for multiple alternative paths
            from haskell.path_planner_interface import plan_multiple_alternative_paths

            failed_node_names = list(self.failed_nodes)
            alternative_paths = plan_multiple_alternative_paths(
                self.node_set, desired_outputs, failed_node_names
            )

            if alternative_paths:
                if self.verbose:
                    print(f"Found {len(alternative_paths)} alternative paths")
                    for i, path in enumerate(alternative_paths):
                        print(f"  Alternative path {i + 1}: {path}")
                return alternative_paths
            else:
                if self.verbose:
                    print("No alternative paths found")
                return []

        except Exception as e:
            if self.verbose:
                print(f"Error finding alternative paths: {e}")
            return []

    def get_next_alternative_path(self) -> Optional[List[str]]:
        """
        Get the next alternative path to try.

        Returns:
            Next alternative path, or None if no more alternatives
        """
        if self.alternative_paths:
            return self.alternative_paths.pop(0)
        return None

    def switch_to_alternative_path(self, alternative_path: List[str]) -> bool:
        """
        Switch to an alternative execution path.

        Args:
            alternative_path: The alternative path to switch to

        Returns:
            True if successfully switched, False otherwise
        """
        if not alternative_path:
            return False

        with self.status_lock:
            self.planned_execution_path = alternative_path
            self.current_path_index = 0

            # Reset status for nodes in the new path that aren't failed
            for node_name in alternative_path:
                if node_name not in self.failed_nodes:
                    self.output_status[node_name] = "pending"

            if self.verbose:
                print(f"Switched to alternative path: {alternative_path}")
            return True

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
            execution_path = plan_path(self.node_set, desired_outputs)
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
        routing logic for optimal agent selection, and includes failure handling.
        """
        with self.status_lock:
            # Check if the node failed
            if node in self.failed_nodes:
                if self.can_retry_node(node):
                    # Reset failure status for retry
                    self.reset_node_failure(node)
                    if self.verbose:
                        print(f"Retrying failed node: {node}")
                else:
                    if self.verbose:
                        print(f"Node {node} has exceeded maximum retries")
                    # Try to find alternative paths if we haven't already
                    if not self.alternative_paths and self.enable_alternative_paths:
                        # Get desired outputs from state if available
                        desired_outputs = []
                        if "query" in state and hasattr(
                            state["query"], "inferred_goal_output"
                        ):
                            desired_outputs = state["query"].inferred_goal_output

                        if desired_outputs:
                            self.alternative_paths = self.find_alternative_paths(
                                desired_outputs
                            )
                            if self.alternative_paths:
                                alternative_path = self.get_next_alternative_path()
                                if alternative_path:
                                    self.switch_to_alternative_path(alternative_path)
                                    if self.verbose:
                                        print(
                                            f"Switched to alternative path after node {node} failure"
                                        )

            # Mark node as done if it's not failed
            if node not in self.failed_nodes:
                self.output_status[node] = "done"

        if self.verbose:
            print(f"Node {node} complete, looking for next node")

        # If we have a Haskell-planned path, try to follow it
        if self.planned_execution_path and self.current_path_index < len(
            self.planned_execution_path
        ):
            next_planned_node = self.planned_execution_path[self.current_path_index]

            # Check if the planned node is ready to run and not failed
            if (
                self._is_node_ready_to_run(next_planned_node)
                and next_planned_node not in self.failed_nodes
            ):
                self.current_path_index += 1
                with self.status_lock:
                    if self.verbose:
                        print(f"Running planned node: {next_planned_node}")
                    self.output_status[next_planned_node] = "running"
                if self.metrics is not None:
                    self.metrics[next_planned_node]["calls"] += 1
                return next_planned_node
            elif next_planned_node in self.failed_nodes:
                # Skip failed nodes in the planned path
                self.current_path_index += 1
                if self.verbose:
                    print(f"Skipping failed node in planned path: {next_planned_node}")
                # Recursively call to get the next node
                return self.enhanced_router_function(state, node)

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
            if (
                self.output_status[node_name] == "pending"
                and node_name not in self.failed_nodes
            ):
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
            if (
                self.output_status[node_name] != "done"
                and node_name not in self.failed_nodes
            ):
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
            "failed_nodes": len(self.failed_nodes),
            "haskell_available": self.haskell_available,
            "planned_execution_path": self.planned_execution_path,
            "current_path_index": self.current_path_index,
            "alternative_paths_count": len(self.alternative_paths),
            "failed_outputs": list(self.failed_outputs),
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
