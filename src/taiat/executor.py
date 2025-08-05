"""
TaiatExecutor - Custom execution engine to replace langgraph dependency.

This module provides a manual execution engine that handles running tasks
represented by AgentGraphNode objects in dependency order, replacing the
langgraph StateGraph functionality.
"""

from typing import List, Dict, Any, Optional, Callable, Set
from collections import defaultdict, deque
import functools
import time

from taiat.base import (
    State,
    AgentGraphNode,
    AgentGraphNodeSet,
    AgentData,
    TaiatQuery,
    START_NODE,
    TAIAT_TERMINAL_NODE,
)


class TaiatExecutor:
    """
    Custom execution engine that replaces langgraph functionality.

    This executor manually manages the execution of AgentGraphNode objects
    in dependency order, handling state transitions, error cases, and
    execution path tracking.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.execution_path = []
        self.execution_stats = {
            "nodes_executed": 0,
            "execution_time": 0,
            "errors": [],
            "dependencies_resolved": 0,
        }

    def _build_dependency_graph(
        self, nodes: List[AgentGraphNode]
    ) -> Dict[str, Set[str]]:
        """
        Build a dependency graph from AgentGraphNode objects.

        Args:
            nodes: List of AgentGraphNode objects

        Returns:
            Dictionary mapping node names to sets of their dependencies
        """
        dependency_graph = defaultdict(set)

        # Create a mapping of output names to node names with parameters
        output_to_node = {}
        for node in nodes:
            for output in node.outputs:
                # Store output with its parameters for subset matching
                output_to_node[(output.name, frozenset(output.parameters.items()))] = (
                    node.name
                )

        # Build dependency graph using subset matching
        for node in nodes:
            for input_data in node.inputs:
                input_params = frozenset(input_data.parameters.items())

                # Find outputs that satisfy this input (subset matching)
                for (
                    output_name,
                    output_params,
                ), producer_node in output_to_node.items():
                    if input_data.name == output_name and input_params <= output_params:
                        dependency_graph[node.name].add(producer_node)

        return dict(dependency_graph)

    def _topological_sort(
        self, nodes: List[AgentGraphNode], dependency_graph: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Perform topological sort to determine execution order.

        Args:
            nodes: List of AgentGraphNode objects
            dependency_graph: Dependency graph mapping node names to dependencies

        Returns:
            List of node names in execution order
        """
        # Create adjacency list and in-degree count
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)

        node_names = {node.name for node in nodes}

        for node_name in node_names:
            in_degree[node_name] = 0

        for node_name, dependencies in dependency_graph.items():
            for dep in dependencies:
                if dep in node_names:
                    adjacency[dep].append(node_name)
                    in_degree[node_name] += 1

        # Kahn's algorithm for topological sort
        queue = deque(
            [node_name for node_name in node_names if in_degree[node_name] == 0]
        )
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(node_names):
            raise ValueError("Circular dependency detected in agent graph")

        return result

    def _is_node_ready(
        self,
        node_name: str,
        dependency_graph: Dict[str, Set[str]],
        executed_nodes: Set[str],
    ) -> bool:
        """
        Check if a node is ready to execute (all dependencies satisfied).

        Args:
            node_name: Name of the node to check
            dependency_graph: Dependency graph
            executed_nodes: Set of already executed node names

        Returns:
            True if node is ready to execute
        """
        if node_name not in dependency_graph:
            return True

        dependencies = dependency_graph[node_name]
        return all(dep in executed_nodes for dep in dependencies)

    def _find_node_by_name(
        self, node_name: str, nodes: List[AgentGraphNode]
    ) -> Optional[AgentGraphNode]:
        """
        Find an AgentGraphNode by name.

        Args:
            node_name: Name of the node to find
            nodes: List of AgentGraphNode objects

        Returns:
            AgentGraphNode if found, None otherwise
        """
        for node in nodes:
            if node.name == node_name:
                return node
        return None

    def _execute_node(self, node: AgentGraphNode, state: State) -> State:
        """
        Execute a single AgentGraphNode.

        Args:
            node: AgentGraphNode to execute
            state: Current execution state

        Returns:
            Updated state after execution
        """
        if not node.function:
            if self.verbose:
                print(f"Warning: Node {node.name} has no function to execute")
            return state

        try:
            if self.verbose:
                print(f"Executing node: {node.name}")

            start_time = time.time()
            result_state = node.function(state)
            execution_time = time.time() - start_time

            # Store outputs in state for dependency resolution
            for output in node.outputs:
                # Create a key that includes parameters for proper matching
                if output.parameters:
                    # For outputs with parameters, store with parameter key
                    param_key = (
                        f"{output.name}_{hash(frozenset(output.parameters.items()))}"
                    )
                    result_state["data"][param_key] = result_state["data"].get(
                        output.name
                    )
                else:
                    # For outputs without parameters, store directly
                    result_state["data"][output.name] = result_state["data"].get(
                        output.name
                    )

            self.execution_stats["nodes_executed"] += 1
            self.execution_stats["execution_time"] += execution_time

            if self.verbose:
                print(
                    f"Node {node.name} executed successfully in {execution_time:.3f}s"
                )

            return result_state

        except Exception as e:
            error_msg = f"Error executing node {node.name}: {str(e)}"
            self.execution_stats["errors"].append(error_msg)

            if self.verbose:
                print(f"Error: {error_msg}")

            # Update query status
            if "query" in state:
                state["query"].status = "error"
                state["query"].error = error_msg

            raise e

    def execute(
        self,
        nodes: List[AgentGraphNode],
        state: State,
        goal_outputs: List[AgentData] = None,
    ) -> State:
        """
        Execute a list of AgentGraphNode objects in dependency order.

        Args:
            nodes: List of AgentGraphNode objects to execute
            state: Initial execution state
            goal_outputs: Optional list of goal outputs to track

        Returns:
            Final execution state
        """
        if not nodes:
            return state

        start_time = time.time()
        self.execution_path = []
        self.execution_stats = {
            "nodes_executed": 0,
            "execution_time": 0,
            "errors": [],
            "dependencies_resolved": 0,
        }

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(nodes)

        # Get execution order
        try:
            execution_order = self._topological_sort(nodes, dependency_graph)
        except ValueError as e:
            if self.verbose:
                print(f"Error in dependency resolution: {e}")
            if "query" in state:
                state["query"].status = "error"
                state["query"].error = str(e)
            return state

        # Execute nodes in order
        executed_nodes = set()

        for node_name in execution_order:
            if not self._is_node_ready(node_name, dependency_graph, executed_nodes):
                error_msg = f"Node {node_name} has unsatisfied dependencies"
                if self.verbose:
                    print(f"Error: {error_msg}")
                if "query" in state:
                    state["query"].status = "error"
                    state["query"].error = error_msg
                return state

            node = self._find_node_by_name(node_name, nodes)
            if not node:
                if self.verbose:
                    print(f"Warning: Node {node_name} not found in node list")
                continue

            try:
                state = self._execute_node(node, state)
                executed_nodes.add(node_name)
                self.execution_path.append(node)
                self.execution_stats["dependencies_resolved"] += 1

            except Exception as e:
                # Error already handled in _execute_node
                return state

        # Update query with execution path
        if "query" in state:
            state["query"].path = self.execution_path
            state["query"].status = "success"

        total_time = time.time() - start_time
        self.execution_stats["execution_time"] = total_time

        if self.verbose:
            print(f"Execution completed in {total_time:.3f}s")
            print(f"Nodes executed: {self.execution_stats['nodes_executed']}")
            print(
                f"Dependencies resolved: {self.execution_stats['dependencies_resolved']}"
            )

        return state

    def get_execution_path(self) -> List[AgentGraphNode]:
        """
        Get the execution path from the last run.

        Returns:
            List of AgentGraphNode objects in execution order
        """
        return self.execution_path.copy()

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics from the last run.

        Returns:
            Dictionary containing execution statistics
        """
        return self.execution_stats.copy()
