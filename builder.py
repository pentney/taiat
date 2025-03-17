from collections import defaultdict
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
    AgentGraphNode,
    AgentGraphNodeSet,
    State,
    TaiatQuery,
    TAIAT_TERMINAL_NODE,
    taiat_terminal_node,
)

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


class TaiatBuilder:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.graph = None
        self.data_source = defaultdict(list)
        self.data_dependence = defaultdict(list)

    def build(
            self,
            state: State,
            node_set: AgentGraphNodeSet,
            inputs: list[str],
            terminal_nodes: list[str],
        ) -> StateGraph:
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node(TAIAT_TERMINAL_NODE, taiat_terminal_node)
        self.graph_builder.add_edge(TAIAT_TERMINAL_NODE, END)
        for node in node_set.nodes:
            no_deps = True
            for output in node.outputs:
                if output.name in self.data_source:
                    raise ValueError(f"output {output.name} defined twice")
                self.data_source[output.name] = node.name
                self.data_dependence[output.name].extend(node.inputs)
                for input in node.inputs:
                    if input.name not in state["data"]:
                        no_deps = False
            self.graph_builder.add_node(node.name, node.function)
            if node.name in terminal_nodes:
                self.graph_builder.add_edge(node.name, TAIAT_TERMINAL_NODE)
            if no_deps:
                self.graph_builder.add_edge(START, node.name)
        for input in inputs:
            self.data_dependence[input] = None
            self.data_source[input] = None
        for dest_output, dependence in self.data_dependence.items():
            dest = self.data_source[dest_output]
            if dest is None:
                if dest_output not in state["data"]:
                    raise ValueError(f"output {dest_output} not defined")
            if dependence is not None:
                for dep in dependence:
                    src = self.data_source[dep.name]
                    if src is not None:
                        self.graph_builder.add_edge(src, dest)
        self.graph = self.graph_builder.compile()
        return self.graph

    def get_plan(self, needed_outputs: list[str]) -> list[AgentGraphNode]:
        pass

    def visualize(self) -> Image:
        return Image(self.graph.get_graph().draw_mermaid_png())


