from typing_extensions import TypedDict
from typing import Annotated, Any, Callable

from IPython.display import Image, display

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
                for input in dest.inputs:
                    self.data_dependence[output.name].append(input)

            graph_builder.add_node(dest)
            graph_builder.add_edge(top_node, dest)
            for node in subgraph:
                self.build(graph_builder, node, dest)
        self.graph = graph_builder.compile()

    def visualize(self) -> Image:
        return Image(self.graph.get_graph().draw_mermaid_png())


