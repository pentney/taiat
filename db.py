from typing import Any
from pydantic import BaseModel
from sqlalchemy import (
    Table,
    Column,
    DateTime,
    BigInteger,
    String,
    MetaData,
    ForeignKey,
    insert,
    func,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from taiat.base import TaiatQuery

class Database(BaseModel):
    type: str
    def add_run(
            self,
            query: TaiatQuery,
            data: dict[str, Any],
        ) -> None:
            pass

    def update_run(
            self,
            query: TaiatQuery,
            data: dict[str, Any],
        ) -> None:
            pass

metadata = MetaData()

taiat_query_table = Table(
    'taiat_query',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('query', String),
    Column('inferred_goal_output', String),
    Column('intermediate_data', ARRAY(String)),
    Column('status', String),
    Column('error', String),
    Column('path', String),
    Column('created_at', DateTime, server_default=func.now())
)

taiat_output_table = Table(
    'taiat_query_data',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('query_id', BigInteger, ForeignKey('taiat_query.id')),
    Column('data', JSONB),
    Column('created_at', DateTime, server_default=func.now())
)

class PostgresDatabase(Database):
    type: str = "postgres"
    session_maker: sessionmaker

    model_config = {
        "arbitrary_types_allowed": True
    }

    def add_run(
        self,
        query: TaiatQuery,
        data: dict[str, Any],
    ) -> None:
        try:
            session = self.session_maker()
            qstmt = insert(taiat_query_table).values(
                query.to_dict()
            ),
            id = session.execute(qstmt).fetchone()[0]
            for name, value in data.items():
                dstmt = insert(taiat_output_table).values(
                    {
                        'query_id': id,
                        'name': name,
                        'data': value,
                    }
                ),
                session.execute(dstmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()




    