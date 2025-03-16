import os
import getpass
from typing_extensions import TypedDict
from typing import Any, Callable, Optional
from collections import defaultdict
from IPython.display import Image, display
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

class AgentData(BaseModel):
    name: str
    data: Optional[Any] = None

class AgentGraphNode(BaseModel):
    name: str
    function: Callable
    inputs: list[AgentData]
    outputs: list[AgentData]


class AgentGraphNodeSet(BaseModel):
    nodes: list[AgentGraphNode]

    def get_plan(self, needed_outputs: list[str]) -> list[AgentGraphNode]:
        pass

class TaiatQuery(BaseModel):
    query: str
    inferred_goal_output: Optional[str] = None
    intermediate_data: Optional[list[str]] = None
    status: Optional[str] = None
    error: str = ""
    path: Optional[list[AgentGraphNode]] = None

    @classmethod
    def from_db_dict(db_dict: dict) -> "TaiatQuery":
        return TaiatQuery(
            query=db_dict["query"],
            status=db_dict["status"],
            path=db_dict["path"],
        )
    
    def as_db_dict(self) -> dict:
        return {
            "query": self.query,
            "status": self.status,
            "path": self.path,
        }

class State(TypedDict):
    query: TaiatQuery
    data: dict[str, AgentData]

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
            print(f"adding node {node}")
            self.graph_builder.add_node(node.name, node.function)
            if node.name in terminal_nodes:
                print(f"adding edge {node.name} -> END")
                self.graph_builder.add_edge(node.name, END)
            if no_deps:
                print(f"adding edge START -> {node.name}")
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
                        print(f"adding edge {src} -> {dest}")
                        self.graph_builder.add_edge(src, dest)
        print(f"graph_builder: {self.graph_builder}")
        self.graph = self.graph_builder.compile()
        return self.graph
    
    def visualize(self) -> Image:
        return Image(self.graph.get_graph().draw_mermaid_png())


