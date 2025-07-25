"""
TaiatBuilder with Global Optimized Prolog Path Planner Integration.

This module provides TaiatBuilder that uses the global optimized
Prolog path planner to improve query planning and execution path determination.
"""

from copy import copy
from collections import defaultdict
import json
import os
import operator
import getpass
from typing_extensions import TypedDict
from typing import Any, Callable, Optional, Annotated, List

from IPython.display import Image, display
from pydantic import BaseModel, field_validator, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from taiat.base import (
    AgentData,
    AgentGraphNode,
    AgentGraphNodeSet,
    FrozenAgentData,
    State,
    TaiatQuery,
    TAIAT_TERMINAL_NODE,
    taiat_terminal_node,
    START_NODE,
)
from taiat.metrics import TaiatMetrics
from taiat.manager import TaiatManager, create_manager


class TaiatBuilder:
    """
    TaiatBuilder that uses global optimized Prolog path planning for better query execution.

    This builder extends the original TaiatBuilder with global optimized Prolog-based path planning
    capabilities while maintaining backward compatibility.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        verbose: bool = False,
        add_metrics: bool = True,
        use_prolog_planning: bool = True,
        fallback_to_original: bool = True,
    ):
        self.llm = llm
        self.graph = None
        self.data_source = defaultdict(dict)
        self.data_dependence = defaultdict(dict)
        self.verbose = verbose
        self.add_metrics = add_metrics
        self.use_prolog_planning = use_prolog_planning
        self.fallback_to_original = fallback_to_original

    def source_match(self, name, parameters):
        """Find a node that produces a specific output."""
        if name in self.data_source:
            for source_parameter_json, source in self.data_source[name].items():
                source_parameters = json.loads(source_parameter_json)
                if parameters.items() <= source_parameters.items():
                    return source
        return None

    def get_dependencies(self, name, parameters):
        """Get dependencies for a specific output."""
        deps = []
        if name in self.data_dependence:
            for dep_parameter_json, dep_list in self.data_dependence[name].items():
                dep_parameters = json.loads(dep_parameter_json)
                for dep in dep_list:
                    if dep is START_NODE:
                        continue
                    else:
                        deps.append(AgentData(name=dep.name, parameters=dep.parameters))
        else:
            return None
        return deps

    def build(
        self,
        node_set: AgentGraphNodeSet | list[dict],
        inputs: list[AgentData],
        terminal_nodes: list[str],
        add_metrics: bool = True,
    ) -> StateGraph:
        """
        Builds dependency and source maps for a set of nodes for path generation.
        This method is identical to the original TaiatBuilder.build().
        """
        self.node_set = node_set
        nodes = node_set
        if isinstance(node_set, list):
            nodes = AgentGraphNodeSet(
                nodes=[AgentGraphNode(**node) for node in node_set]
            )
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node(TAIAT_TERMINAL_NODE, taiat_terminal_node)
        self.graph_builder.add_edge(TAIAT_TERMINAL_NODE, END)
        for node in nodes.nodes:
            for output in node.outputs:
                output_json = json.dumps(output.parameters)
                if (
                    output.name in self.data_source
                    and output_json in self.data_source[output.name]
                ):
                    raise ValueError(f"output {output} defined twice")
                self.data_source.setdefault(output.name, {})[
                    json.dumps(output.parameters)
                ] = node
                self.data_dependence.setdefault(output.name, {}).setdefault(
                    json.dumps(output.parameters), []
                ).extend(node.inputs)
            self.graph_builder.add_node(node.name, node.function)
            if node.name in terminal_nodes:
                self.graph_builder.add_edge(node.name, TAIAT_TERMINAL_NODE)
        for input in inputs:
            self.data_dependence.setdefault(input.name, {})[
                json.dumps(input.parameters)
            ] = [START_NODE]
            self.data_source.setdefault(input.name, {})[
                json.dumps(input.parameters)
            ] = START_NODE
        for dest_output_name, dependence_map in self.data_dependence.items():
            for dest_output_parameter_json, data_dependencies in dependence_map.items():
                dest_output_parameters = json.loads(dest_output_parameter_json)
                dep_dest = self.source_match(dest_output_name, dest_output_parameters)
                if dep_dest is None:
                    raise ValueError(
                        f"dependency {AgentData(name=dest_output_name, parameters=dest_output_parameters)} not defined"
                    )
                else:
                    for dependency in data_dependencies:
                        if dependency is START_NODE:
                            continue  # Reachable from START
                        dep_src = self.source_match(
                            dependency.name, dependency.parameters
                        )
                        if dep_src is None:
                            raise ValueError(
                                f"dependencies for {dependency} not defined"
                            )
                        else:
                            if dep_src is START_NODE:
                                self.graph_builder.add_edge(START, dep_dest.name)
                            else:
                                self.graph_builder.add_edge(dep_src.name, dep_dest.name)

        # Add metrics, if appropriate.
        if self.add_metrics:
            self.metrics = TaiatMetrics()
            for node in self.graph_builder.nodes:
                self.metrics.add_node_counter(node)

        self.graph = self.graph_builder.compile()
        return self.graph

    def get_plan_with_prolog(
        self, query: TaiatQuery, goal_outputs: list[AgentData]
    ) -> tuple[StateGraph | None, str, str]:
        """
        Get execution plan using global optimized Prolog path planner.

        This method uses the global optimized Prolog path planner to determine the optimal execution
        sequence, then builds the appropriate StateGraph for execution.

        Args:
            query: The TaiatQuery to plan for
            goal_outputs: List of desired outputs to produce

        Returns:
            Tuple of (StateGraph, status, error_message)
        """
        if not self.use_prolog_planning:
            return self.get_plan_original(query, goal_outputs)

        try:
            # Import global optimized Prolog planner
            from prolog.taiat_path_planner import (
                plan_taiat_path_global,
            )

            # Plan execution path using global optimized Prolog
            execution_path = plan_taiat_path_global(self.node_set, goal_outputs)
            if execution_path is None:
                if self.fallback_to_original:
                    if self.verbose:
                        print(
                            "Global optimized Prolog planning failed, falling back to original method"
                        )
                    return self.get_plan_original(query, goal_outputs)
                else:
                    return (
                        None,
                        "error",
                        "Global optimized Prolog planning failed and fallback is disabled",
                    )

            if self.verbose:
                print(
                    f"Global optimized Prolog planned execution path: {execution_path}"
                )

            # Build a simplified StateGraph based on the Prolog execution path
            # This creates a linear execution path as determined by Prolog
            simplified_graph = StateGraph(State)
            simplified_graph.add_node(TAIAT_TERMINAL_NODE, taiat_terminal_node)
            simplified_graph.add_edge(TAIAT_TERMINAL_NODE, END)

            # Add nodes in the execution order
            for i, node_name in enumerate(execution_path):
                # Find the actual node
                node = next(
                    (n for n in self.node_set.nodes if n.name == node_name), None
                )
                if node is None:
                    return (
                        None,
                        "error",
                        f"Node {node_name} from Prolog plan not found in node set",
                    )

                simplified_graph.add_node(node_name, node.function)

                # Add edge from previous node or START
                if i == 0:
                    simplified_graph.add_edge(START, node_name)
                else:
                    simplified_graph.add_edge(execution_path[i - 1], node_name)

                # Add edge to terminal if this is the last node
                if i == len(execution_path) - 1:
                    simplified_graph.add_edge(node_name, TAIAT_TERMINAL_NODE)

            # Add metrics if enabled
            if self.add_metrics:
                self.metrics = TaiatMetrics()
                for node in simplified_graph.nodes:
                    self.metrics.add_node_counter(node)

            compiled_graph = simplified_graph.compile()
            return (compiled_graph, "success", "")

        except Exception as e:
            if self.fallback_to_original:
                if self.verbose:
                    print(
                        f"Error in global optimized Prolog planning: {e}, falling back to original method"
                    )
                return self.get_plan_original(query, goal_outputs)
            else:
                return (None, "error", f"Global optimized Prolog planning error: {e}")

    def get_plan_original(
        self, query: TaiatQuery, goal_outputs: list[AgentData]
    ) -> tuple[StateGraph | None, str, str]:
        """
        Get execution plan using the original TaiatBuilder logic.

        This method implements the original planning logic as a fallback when
        Prolog planning is not available or fails.

        Args:
            query: The TaiatQuery to plan for
            goal_outputs: List of desired outputs to produce

        Returns:
            Tuple of (StateGraph, status, error_message)
        """
        try:
            # Use the original TaiatBuilder logic
            # This is a simplified version that focuses on the core planning logic

            # Find nodes that produce the desired outputs
            required_nodes = set()
            for goal_output in goal_outputs:
                source_node = self.source_match(
                    goal_output.name, goal_output.parameters
                )
                if source_node is None:
                    return (None, "error", f"Cannot produce output: {goal_output.name}")
                required_nodes.add(source_node.name)

            # Add dependencies of required nodes
            to_process = list(required_nodes)
            while to_process:
                current = to_process.pop(0)
                dependencies = self.get_dependencies(current, {})
                if dependencies:
                    for dep in dependencies:
                        dep_source = self.source_match(dep.name, dep.parameters)
                        if dep_source and dep_source.name not in required_nodes:
                            required_nodes.add(dep_source.name)
                            to_process.append(dep_source.name)

            # Build execution graph
            execution_graph = StateGraph(State)
            execution_graph.add_node(TAIAT_TERMINAL_NODE, taiat_terminal_node)
            execution_graph.add_edge(TAIAT_TERMINAL_NODE, END)

            # Add required nodes
            for node_name in required_nodes:
                node = next(
                    (n for n in self.node_set.nodes if n.name == node_name), None
                )
                if node:
                    execution_graph.add_node(node_name, node.function)

            # Add edges based on dependencies
            for node_name in required_nodes:
                dependencies = self.get_dependencies(node_name, {})
                if dependencies:
                    for dep in dependencies:
                        dep_source = self.source_match(dep.name, dep.parameters)
                        if dep_source and dep_source.name in required_nodes:
                            execution_graph.add_edge(dep_source.name, node_name)
                else:
                    # No dependencies, connect to START
                    execution_graph.add_edge(START, node_name)

            # Connect final nodes to terminal
            for node_name in required_nodes:
                # Check if this node produces any goal outputs
                node = next(
                    (n for n in self.node_set.nodes if n.name == node_name), None
                )
                if node:
                    for output in node.outputs:
                        for goal_output in goal_outputs:
                            if output.name == goal_output.name:
                                execution_graph.add_edge(node_name, TAIAT_TERMINAL_NODE)
                                break

            # Add metrics if enabled
            if self.add_metrics:
                self.metrics = TaiatMetrics()
                for node in execution_graph.nodes:
                    self.metrics.add_node_counter(node)

            compiled_graph = execution_graph.compile()
            return (compiled_graph, "success", "")

        except Exception as e:
            return (None, "error", f"Original planning error: {e}")

    def get_plan(
        self, query: TaiatQuery, goal_outputs: list[AgentData]
    ) -> tuple[StateGraph | None, str, str]:
        """
        Get execution plan using the appropriate planning method.

        This method automatically chooses between Prolog planning and original planning
        based on configuration and availability.

        Args:
            query: The TaiatQuery to plan for
            goal_outputs: List of desired outputs to produce

        Returns:
            Tuple of (StateGraph, status, error_message)
        """
        if self.use_prolog_planning:
            return self.get_plan_with_prolog(query, goal_outputs)
        else:
            return self.get_plan_original(query, goal_outputs)

    def _format_output_with_params(self, output: AgentData) -> str:
        """
        Format an output name with its parameters for display.

        Args:
            output: The AgentData output to format

        Returns:
            Formatted string with name and parameters
        """
        if not output.parameters:
            return output.name
        else:
            # Format parameters as key=value pairs
            param_str = ", ".join([f"{k}={v}" for k, v in output.parameters.items()])
            return f"{output.name}({param_str})"

    def create_graph_visualization(
        self, query: TaiatQuery, goal_outputs: list[AgentData]
    ) -> Optional[str]:
        """
        Creates a graphviz visualization of the execution path for the specific query.
        Returns the DOT source code as a string, or None if visualization cannot be created.
        """
        try:
            import graphviz
        except ImportError:
            return None

        # Use the query's path to show the actual execution sequence
        if not query.path:
            return None

        dot = graphviz.Digraph(comment="Taiat Query Execution Path")
        dot.attr(rankdir="TB")

        # Add nodes for the execution path
        for i, node in enumerate(query.path):
            label = node.description or node.name
            dot.node(node.name, f"{i + 1}. {label}", shape="box")

        # Add edges to show the execution sequence with input/output data
        for i in range(len(query.path) - 1):
            current_node = query.path[i]
            next_node = query.path[i + 1]

            # Find outputs from current node that are inputs to next node
            edge_outputs = []
            for output in current_node.outputs:
                for input_data in next_node.inputs:
                    if output.name == input_data.name:
                        # Check if parameters match - input parameters should be a subset of output parameters
                        # If input has no parameters, it should match any output with the same name
                        if (
                            not input_data.parameters
                            or input_data.parameters.items()
                            <= output.parameters.items()
                        ):
                            edge_outputs.append(self._format_output_with_params(output))

            # Create edge label with the data flow
            if edge_outputs:
                edge_label = f"→ {', '.join(edge_outputs)}"
                # Use green color for successful data flow
                dot.edge(
                    current_node.name,
                    next_node.name,
                    label=edge_label,
                    color="green",
                    fontcolor="green",
                )
            else:
                # Use red if no clear data flow (potential issue)
                dot.edge(
                    current_node.name, next_node.name, color="red", fontcolor="red"
                )

        # Show final outputs from the last node(s) in the execution path
        if query.path:
            # Find all requested outputs and which nodes produce them
            requested_outputs = {}
            for goal_output in goal_outputs:
                # Find which node in the execution path produces this output
                found_match = False
                for node in query.path:
                    if found_match:
                        break
                    for output in node.outputs:
                        if output.name == goal_output.name:
                            # Check if parameters match
                            if (
                                not goal_output.parameters
                                or goal_output.parameters.items()
                                <= output.parameters.items()
                            ):
                                formatted_output = self._format_output_with_params(
                                    output
                                )
                                requested_outputs[formatted_output] = node.name
                                found_match = True
                                break

            if requested_outputs:
                # Add a special node to show final outputs
                final_outputs_str = (
                    f"Final Outputs: {', '.join(requested_outputs.keys())}"
                )
                dot.node(
                    "final_outputs",
                    final_outputs_str,
                    shape="ellipse",
                    style="filled",
                    fillcolor="lightgreen",
                )

                # Add edges from the nodes that produce the final outputs
                for output_name, producer_node in requested_outputs.items():
                    dot.edge(
                        producer_node,
                        "final_outputs",
                        label=f"→ {output_name}",
                        color="green",
                        fontcolor="green",
                    )

        return dot.source


def create_builder(
    llm: BaseChatModel,
    use_prolog_planning: bool = True,
    fallback_to_original: bool = True,
    **kwargs,
) -> TaiatBuilder:
    """
    Create a TaiatBuilder with global optimized Prolog path planning.

    Args:
        llm: The language model to use
        use_prolog_planning: Whether to use global optimized Prolog path planning
        fallback_to_original: Whether to fall back to original planning if Prolog fails
        **kwargs: Additional arguments to pass to TaiatBuilder

    Returns:
        Configured TaiatBuilder instance
    """
    return TaiatBuilder(
        llm=llm,
        use_prolog_planning=use_prolog_planning,
        fallback_to_original=fallback_to_original,
        **kwargs,
    )
