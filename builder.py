import os
import operator
import getpass
from typing_extensions import TypedDict
from typing import Any, Callable, Optional, Annotated
from collections import defaultdict
from IPython.display import Image, display
from pydantic import BaseModel, field_validator, Field

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

    @classmethod
    @field_validator("data",mode="after")
    def validate_data(cls):
        return cls

class AgentGraphNode(BaseModel):
    name: str
    function: Callable
    inputs: list[AgentData]
    outputs: list[AgentData]


class AgentGraphNodeSet(BaseModel):
    nodes: list[AgentGraphNode]

class TaiatQuery(BaseModel):
    query: Annotated[str, operator.add]
    inferred_goal_output: Optional[str] = None
    intermediate_data: Annotated[list[str], operator.add] = []
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
    query: Annotated[TaiatQuery, lambda x,_: x]
    data: Annotated[dict[str, Any], operator.or_] = {}


TAIAT_TERMINAL_NODE = "__terminal__"
def taiat_terminal_node(state: State) -> State:
    return state

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


