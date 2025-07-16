from copy import copy
from collections import defaultdict
import json
import os
import operator
import getpass
from typing_extensions import TypedDict
from typing import Any, Callable, Optional, Annotated

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
)
from taiat.metrics import TaiatMetrics

START_NODE = AgentGraphNode(name=START, function=None, inputs=[], outputs=[])


from taiat.manager import TaiatManager

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


class TaiatBuilder:
    def __init__(
        self, llm: BaseChatModel, verbose: bool = False, add_metrics: bool = True
    ):
        self.llm = llm
        self.graph = None
        self.data_source = defaultdict(dict)
        self.data_dependence = defaultdict(dict)
        self.verbose = verbose
        self.add_metrics = add_metrics

    def source_match(self, name, parameters):
        if name in self.data_source:
            for source_parameter_json, source in self.data_source[name].items():
                source_parameters = json.loads(source_parameter_json)
                if parameters.items() <= source_parameters.items():
                    return source
        return None

    def get_dependencies(self, name, parameters):
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
                            edge_outputs.append(output.name)

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
                for node in query.path:
                    for output in node.outputs:
                        if output.name == goal_output.name:
                            # Check if parameters match
                            if (
                                not goal_output.parameters
                                or goal_output.parameters.items()
                                <= output.parameters.items()
                            ):
                                requested_outputs[goal_output.name] = node.name
                                break
                    if goal_output.name in requested_outputs:
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

    def build(
        self,
        node_set: AgentGraphNodeSet | list[dict],
        inputs: list[AgentData],
        terminal_nodes: list[str],
        add_metrics: bool = True,
    ) -> StateGraph:
        """
        Builds dependency and source maps for a set of nodes for path generation.
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

    def get_plan(
        self, query: TaiatQuery, goal_outputs: list[str]
    ) -> tuple[StateGraph | None, str, str]:
        start_nodes = []
        all_needed_outputs = set(
            FrozenAgentData(**goal_output.model_dump()) for goal_output in goal_outputs
        )
        for goal_output in goal_outputs:
            dep_src = self.source_match(goal_output.name, goal_output.parameters)
            if dep_src is None:
                status = "error"
                error = f"Goal output {goal_output} unknown"
                return (None, status, error)
            else:
                output_set = set()
                output_set.add(FrozenAgentData(**goal_output.model_dump()))
                while len(output_set) > 0:
                    current_output = output_set.pop()
                    deps = self.get_dependencies(
                        current_output.name, current_output.parameters
                    )
                    if deps is None:
                        status = "error"
                        error = (
                            f"Graph error: bad intermediate data {current_output} found"
                        )
                        return (None, status, error)
                    needed_outputs = set(
                        [FrozenAgentData(**dep.model_dump()) for dep in deps]
                    )
                    if needed_outputs:
                        while len(needed_outputs) > 0:
                            needed_output = needed_outputs.pop()
                            if needed_output.name not in self.data_dependence:
                                status = "error"
                                error = (
                                    f"Intermediate data {needed_output.name} unknown"
                                )
                                return (None, status, error)
                            # TODO: fix to check parameter match
                            if needed_output.name in needed_outputs:
                                status = "error"
                                error = f"Graph error: circular dependency found: {needed_output}"
                                return (None, status, error)
                            all_needed_outputs.add(
                                FrozenAgentData(**needed_output.model_dump())
                            )
                            deps = self.get_dependencies(
                                needed_output.name, needed_output.parameters
                            )
                            if deps is not None:
                                for dep in deps:
                                    needed_outputs.add(
                                        FrozenAgentData(**dep.model_dump())
                                    )
        # Now get nodes and edges for graph
        graph_builder = StateGraph(State)
        graph_builder.add_node(TAIAT_TERMINAL_NODE, taiat_terminal_node)
        graph_builder.add_edge(TAIAT_TERMINAL_NODE, END)
        plan_nodes = {}
        reverse_plan_edges = defaultdict(list)
        plan_edges = defaultdict(list)
        for output in all_needed_outputs:
            src = self.source_match(output.name, output.parameters)
            if src is None:
                status = "error"
                error = f"Graph error: bad intermediate data {output} found"
                return (None, status, error)
            elif src.name not in plan_nodes:
                # graph_builder.add_node(src.name, src.function)
                plan_nodes[src.name] = src
        for src, dest in self.graph_builder.edges:
            if ((src in plan_nodes or src is START) and dest in plan_nodes) or (
                src in plan_nodes and dest is TAIAT_TERMINAL_NODE
            ):
                reverse_plan_edges[dest].append(src)
                plan_edges[src].append(dest)

        self.manager = TaiatManager(
            self.node_set, reverse_plan_edges, verbose=self.verbose
        )
        # Add a conditional edge for every node backwards from the goal.
        for node_name, node in plan_nodes.items():
            router = self.manager.make_router_function(node_name)
            if node_name != START:
                graph_builder.add_node(node.name, node.function)
            graph_builder.add_conditional_edges(node.name, router)
        query.all_outputs = [output.name for output in all_needed_outputs]
        graph_builder.add_edge(TAIAT_TERMINAL_NODE, END)
        graph = graph_builder.compile()
        return (graph, "success", "")
