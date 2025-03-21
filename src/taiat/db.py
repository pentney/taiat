from typing import Any
from pydantic import BaseModel
from sqlalchemy import (
    Table,
    Column,
    DateTime,
    Integer,
    BigInteger,
    String,
    MetaData,
    ForeignKey,
    insert,
    func,
    update,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from taiat.base import TaiatQuery

class Database(BaseModel):
    """
    Base class for a database.
    """
    type: str
    def add_run(
            self,
            query: TaiatQuery,
            data: dict[str, Any],
        ) -> None:
            """
            Add a run to the database.
            """
            pass

    def update_run(
            self,
            query: TaiatQuery,
            data: dict[str, Any],
        ) -> None:
            """
            Update a run in the database.
            """
            pass

metadata = MetaData()

taiat_query_table = Table(
    'taiat_query',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('query', String),
    Column('inferred_goal_output', String),
    Column('intermediate_data', ARRAY(String)),
    Column('status', String),
    Column('error', String),
    Column('path', ARRAY(JSONB)),
    Column('created_at', DateTime, server_default=func.now())
)

taiat_output_table = Table(
    'taiat_query_data',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('query_id', BigInteger, ForeignKey('taiat_query.id')),
    Column('name', String),
    Column('data', ARRAY(JSONB)),
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
        """
        Add a run to the database.
        """
        try:
            session = self.session_maker()
            qd = query.as_db_dict()
            qstmt = insert(taiat_query_table).values(qd).returning(
                 taiat_query_table.c.id)
            id = session.execute(qstmt).first()[0]
            for name, value in data.items():
                dstmt = insert(taiat_output_table).values(
                    {
                        'query_id': id,
                        'name': name,
                        'data': value,
                    }
                )
                session.execute(dstmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_run(
        self,
        query: TaiatQuery,
        data: dict[str, Any],
    ) -> None:
        """
        Update a run in the database.
        """
        try:
            session = self.session_maker()
            qd = query.as_db_dict()
            qstmt = update(taiat_query_table).values(qd)
            session.execute(qstmt)
            for name, value in data.items():
                dstmt = insert(taiat_output_table).values(
                    {
                        'query_id': query.id,
                        'name': name,
                        'data': value,
                    }
                )
                session.execute(dstmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()



    