import operator
from typing import Any, Callable, Optional
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, field_validator, ConfigDict


class AgentData(BaseModel):
    name: str
    parameters: dict[str, Any] = {}
    description: str = ""
    data: Optional[Any] = None

    @classmethod
    @field_validator("data",mode="after")
    def validate_data(cls):
        return cls

# Used for hashability in sets.
# Should NOT be used in actual workflow runs, as the data
# cannot be updated.
class FrozenAgentData(AgentData):
    model_config = {"frozen": True}

    def __hash__(self):
        # Convert dictionary to an immutable format (sorted tuple of key-value pairs)
        param_tuple = tuple(sorted(self.parameters.items()))
        return hash((self.name, param_tuple))


class AgentGraphNode(BaseModel):
    name: str
    description: Optional[str] = None
    function: Optional[Callable] = None
    inputs: list[AgentData]
    outputs: list[AgentData]

# base class for output matchers
class OutputMatcher:
    """
    Base class for output matchers, which take a query and return a list of outputs.
    """
    def get_outputs(self, query: str) -> list[str]:
        return []

class AgentGraphNodeSet(BaseModel):
    """
    A set of agent graph nodes.
    """
    nodes: list[AgentGraphNode]

class TaiatQuery(BaseModel):
    """
    A query to be run by the Taiat engine.
    """
    id: Optional[int] = None
    query: Annotated[str, operator.add]
    inferred_goal_output: Optional[str] = None
    all_outputs: Annotated[list[str], operator.add] = []
    status: Optional[str] = None
    error: str = ""
    path: Optional[list[AgentGraphNode]] = None

    @classmethod
    def from_db_dict(db_dict: dict) -> "TaiatQuery":
        """
        Create a TaiatQuery from a dictionary, like those representing a database row.
        """
        return TaiatQuery(
            id=db_dict.get("id"),
            query=db_dict["query"],
            inferred_goal_output=db_dict["inferred_goal_output"],
            status=db_dict["status"],
            error=db_dict["error"],
            path=[AgentGraphNode(**node) for node in db_dict["path"]],
        )
    
    def as_db_dict(self) -> dict:
        """
        Convert a TaiatQuery to a dictionary, like those representing a database row.
        """
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
    """
    The state of the Taiat engine.
    Contains all relevant data to a run of the engine at any point.
    """
    query: Annotated[TaiatQuery, lambda x,_: x]
    data: Annotated[dict[str, AgentData], operator.or_] = {}


TAIAT_TERMINAL_NODE = "__terminal__"
def taiat_terminal_node(state: State) -> State:
    """
    The terminal node of the Taiat engine.
    """
    return state
