from pydantic import BaseModel

from taiat_builder import AgentGraphNode

class TaiatQuery(BaseModel):
    query: str
    status: bool
    path: list[AgentGraphNode]

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

