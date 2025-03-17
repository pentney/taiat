import operator
from typing import Any, Callable, Optional
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, field_validator


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
