import os
import getpass
from typing_extensions import TypedDict
from typing import Annotated, Any, Callable

from IPython.display import Image, display
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_models import BaseChatModel
from langgraph.graph import StateGraph, GraphNode, START, END
from langgraph.graph.message import add_messages


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("ANTHROPIC_API_KEY")

# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

class AgentData(BaseModel):
    name: str
    data: Any

class AgentGraphNode(BaseModel):
    name: str
    function: Callable
    inputs: list[AgentData]
    outputs: list[AgentData]

class AgentGraph(BaseModel):
    nodes: dict[str, "AgentGraph"]

    def get_plan(self, needed_outputs: list[str]) -> list[AgentGraphNode]:
        pass

    def execute_plan(self, plan: list[AgentGraphNode], query: TaiatQuery) -> TaiatQuery:
        pass


class TaiatBuilder:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.graph = None
        self.data_source = None
        self.data_dependence = None

    def build(self, graph_builder: StateGraph,
              subgraph: AgentGraph,
              top_node: GraphNode=START):
        for dest, subgraph in subgraph.nodes.items():
            for output in dest.outputs:
                if output.name not in self.data_source:
                    self.data_source[output.name] = []
                self.data_source[output.name].append(dest)
                if output.name not in self.data_dependence:
                    self.data_dependence[output.name] = []
                self.data_dependence[output.name].extend(dest.inputs)
            graph_builder.add_node(dest)
            graph_builder.add_edge(top_node, dest)
            for node in subgraph:
                self.build(graph_builder, node, dest)
        self.graph = graph_builder.compile()

    def visualize(self) -> Image:
        return Image(self.graph.get_graph().draw_mermaid_png())


