from pydantic import BaseModel
from sqlalchemy import sessionmaker

class Database(BaseModel):
    type: str

class PostgresDatabase(Database):
    type: str = "postgres"
    host: str
    port: int
    user: str
    password: str
    session_maker: sessionmaker

    def add_row(self, query: TaiatQuery) -> None:
        pass

    def update_row(self, query: TaiatQuery) -> None:
        pass


    