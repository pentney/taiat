from typing_extensions import TypedDict
from typing import Annotated, Any, Callable

from pydantic import BaseModel
from langgraph.graph import StateGraph, GraphNode, START, END
from langgraph.graph.message import add_messages

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

class Agentinator:
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.graph = None

    def build(self, graph_builder: StateGraph,
              subgraph: AgentGraph,
              top_node: GraphNode=START):
        for dest, subgraph in subgraph.nodes.items():
            graph_builder.add_node(dest)
            graph_builder.add_edge(top_node, dest)
            for node in subgraph:
                self.build(graph_builder, node, dest)
        self.graph = graph_builder.compile()


