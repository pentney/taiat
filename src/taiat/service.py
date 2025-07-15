from pydantic import BaseModel
from taiat.taiat.engine import TaiatEngine
from taiat.taiat.db import Database


class TaiatService(BaseModel):
    engine: TaiatEngine = Field(default_factory=TaiatEngine)
    db: Database = Field(default_factory=Database)

    def handle_query(self, query: TaiatQuery) -> TaiatQuery:
        self.db.add_row(query)
        self.engine.run(query)
        self.db.update_row(query)
        return query
