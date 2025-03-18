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
    description: str
    function: Optional[Callable] = None
    inputs: list[AgentData]
    outputs: list[AgentData]


class AgentGraphNodeSet(BaseModel):
    nodes: list[AgentGraphNode]

class TaiatQuery(BaseModel):
    id: Optional[int] = None
    query: Annotated[str, operator.add]
    inferred_goal_output: Optional[str] = None
    intermediate_data: Annotated[list[str], operator.add] = []
    status: Optional[str] = None
    error: str = ""
    path: Optional[list[AgentGraphNode]] = None

    @classmethod
    def from_db_dict(db_dict: dict) -> "TaiatQuery":
        return TaiatQuery(
            id=db_dict.get("id"),
            query=db_dict["query"],
            inferred_goal_output=db_dict["inferred_goal_output"],
            status=db_dict["status"],
            error=db_dict["error"],
            path=[AgentGraphNode(**node) for node in db_dict["path"]],
        )
    
    def as_db_dict(self) -> dict:
        clean_path = [node.model_dump(exclude={"function"}) for node in self.path]
        for node in clean_path:
            for input in node["inputs"]:
                input["data"] = None
            for output in node["outputs"]:
                output["data"] = None
        return {
            "id": self.id,
            "query": self.query,
            "inferred_goal_output": self.inferred_goal_output,
            "status": self.status,
            "error": self.error,
            "path": clean_path,
        }


class State(TypedDict):
    query: Annotated[TaiatQuery, lambda x,_: x]
    data: Annotated[dict[str, Any], operator.or_] = {}


TAIAT_TERMINAL_NODE = "__terminal__"
def taiat_terminal_node(state: State) -> State:
    return state
