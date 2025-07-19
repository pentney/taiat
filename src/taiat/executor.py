"""
TaiatExecutor - Manual Agent Execution Engine

This module provides a manual execution engine that replaces langgraph
functionality while maintaining the same agent execution semantics.
"""

from typing import Dict, List, Set, Optional, Callable, Any
from collections import defaultdict, deque
import copy
from taiat.base import State, AgentData, AgentGraphNode, AgentGraphNodeSet, TaiatQuery


class TaiatExecutor:
    """
    Manual execution engine that replaces langgraph functionality.

    This executor manages the execution of agents based on their dependencies,
    ensuring that agents are executed in the correct order to satisfy all
    input requirements.
    """

    def __init__(self, node_set: AgentGraphNodeSet, verbose: bool = False):
        self.node_set = node_set
        self.verbose = verbose
        self.execution_path: List[AgentGraphNode] = []
        self.executed_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()

        # Build dependency graph
        self._build_dependency_graph()

    def _build_dependency_graph(self):
        """Build dependency graph for execution planning."""
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.node_map: Dict[str, AgentGraphNode] = {}

        for node in self.node_set.nodes:
            self.node_map[node.name] = node

            # Add dependencies for each input
            for input_data in node.inputs:
                # Find nodes that produce this input
                for other_node in self.node_set.nodes:
                    for output in other_node.outputs:
                        if (
                            output.name == input_data.name
                            and output.parameters == input_data.parameters
                        ):
                            self.dependencies[node.name].add(other_node.name)
                            self.reverse_dependencies[other_node.name].add(node.name)

    def _is_node_ready(self, node_name: str, available_data: Set[str]) -> bool:
        """
        Check if a node is ready to execute based on available data.

        Args:
            node_name: Name of the node to check
            available_data: Set of available data names

        Returns:
            True if the node can be executed, False otherwise
        """
        if node_name in self.executed_nodes or node_name in self.failed_nodes:
            return False

        node = self.node_map[node_name]

        # Check if all inputs are available
        for input_data in node.inputs:
            # Check for exact match first
            input_key = f"{input_data.name}_{hash(str(input_data.parameters))}"
            if input_key in available_data:
                continue

            # Check if the input name is available as a key
            if input_data.name in available_data:
                continue

            # Check if any available data has the same name and compatible parameters
            found = False
            for data_key in available_data:
                if data_key == input_data.name:
                    found = True
                    break
            if not found:
                return False

        return True

    def _get_available_data(self, state: State) -> Set[str]:
        """
        Get set of available data keys from current state.

        Args:
            state: Current execution state

        Returns:
            Set of available data keys
        """
        available = set()
        for key, agent_data in state["data"].items():
            # Add both the key and the name for flexibility
            available.add(key)
            data_key = f"{agent_data.name}_{hash(str(agent_data.parameters))}"
            available.add(data_key)
        return available

    def _execute_node(self, node: AgentGraphNode, state: State) -> State:
        """
        Execute a single node and update state.

        Args:
            node: Node to execute
            state: Current execution state

        Returns:
            Updated state after node execution
        """
        if self.verbose:
            print(f"Executing node: {node.name}")

        try:
            if node.function:
                new_state = node.function(state)
                self.executed_nodes.add(node.name)
                self.execution_path.append(node)
                return new_state
            else:
                # Node without function - just mark as executed
                self.executed_nodes.add(node.name)
                self.execution_path.append(node)
                return state
        except Exception as e:
            if self.verbose:
                print(f"Error executing node {node.name}: {e}")
            self.failed_nodes.add(node.name)
            raise

    def execute(self, state: State, goal_outputs: List[AgentData]) -> State:
        """
        Execute the agent graph to produce the desired outputs.

        Args:
            state: Initial execution state
            goal_outputs: List of desired outputs to produce

        Returns:
            Final state after execution
        """
        if self.verbose:
            print(f"Starting execution for {len(goal_outputs)} goal outputs")

        # Reset execution tracking
        self.execution_path = []
        self.executed_nodes = set()
        self.failed_nodes = set()

        # Find all nodes needed to produce goal outputs
        required_nodes = self._find_required_nodes(goal_outputs)

        if self.verbose:
            print(f"Required nodes: {list(required_nodes)}")

        # Execute nodes in dependency order
        while required_nodes:
            # Find nodes that are ready to execute
            available_data = self._get_available_data(state)
            ready_nodes = [
                node_name
                for node_name in required_nodes
                if self._is_node_ready(node_name, available_data)
            ]

            if not ready_nodes:
                # Check if we have any failed nodes
                failed_required = required_nodes & self.failed_nodes
                if failed_required:
                    raise RuntimeError(f"Required nodes failed: {failed_required}")

                # Check for circular dependencies or missing inputs
                remaining_nodes = (
                    required_nodes - self.executed_nodes - self.failed_nodes
                )
                if remaining_nodes:
                    raise RuntimeError(
                        f"Cannot execute remaining nodes: {remaining_nodes}"
                    )
                break

            # Execute ready nodes
            for node_name in ready_nodes:
                node = self.node_map[node_name]
                state = self._execute_node(node, state)
                required_nodes.remove(node_name)

        return state

    def _find_required_nodes(self, goal_outputs: List[AgentData]) -> Set[str]:
        """
        Find all nodes required to produce the goal outputs.

        Args:
            goal_outputs: List of desired outputs

        Returns:
            Set of node names required for execution
        """
        required_nodes = set()
        nodes_to_check = deque()

        # Add nodes that produce goal outputs
        for goal_output in goal_outputs:
            for node in self.node_set.nodes:
                for output in node.outputs:
                    if (
                        output.name == goal_output.name
                        and output.parameters == goal_output.parameters
                    ):
                        nodes_to_check.append(node.name)
                        required_nodes.add(node.name)

        # Traverse dependencies
        while nodes_to_check:
            node_name = nodes_to_check.popleft()
            node = self.node_map[node_name]

            # Add dependencies for this node's inputs
            for input_data in node.inputs:
                for other_node in self.node_set.nodes:
                    for output in other_node.outputs:
                        if (
                            output.name == input_data.name
                            and output.parameters == input_data.parameters
                        ):
                            if other_node.name not in required_nodes:
                                required_nodes.add(other_node.name)
                                nodes_to_check.append(other_node.name)

        return required_nodes

    def get_execution_path(self) -> List[AgentGraphNode]:
        """Get the execution path of successfully executed nodes."""
        return self.execution_path.copy()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "executed_nodes": len(self.executed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "total_nodes": len(self.node_set.nodes),
            "execution_path": [node.name for node in self.execution_path],
        }
