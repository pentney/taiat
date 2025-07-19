from typing import Any
from pydantic import BaseModel
from sqlalchemy import (
    Table,
    Column,
    DateTime,
    Integer,
    BigInteger,
    String,
    Boolean,
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

# Agent outputs table - stores actual data with unique IDs
agent_output_table = Table(
    "agent_output",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent_name", String, nullable=False),
    Column("output_name", String, nullable=False),
    Column("data", ARRAY(JSONB), nullable=False),
    Column("data_hash", String, nullable=False),  # For deduplication
    Column("created_at", DateTime, server_default=func.now()),
)

# Query table - stores query metadata
taiat_query_table = Table(
    "taiat_query",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query", String),
    Column("inferred_goal_output", String),
    Column("intermediate_data", ARRAY(String)),
    Column("status", String),
    Column("error", String),
    Column("path", ARRAY(JSONB)),
    Column("visualize_graph", Boolean),
    Column("parameters", JSONB),  # Query-specific parameters
    Column("state_info", JSONB),  # Additional state/context information
    Column("created_at", DateTime, server_default=func.now()),
)

# Junction table linking queries to agent outputs
query_output_table = Table(
    "query_output",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_id", BigInteger, ForeignKey("taiat_query.id"), nullable=False),
    Column(
        "agent_output_id", BigInteger, ForeignKey("agent_output.id"), nullable=False
    ),
    Column(
        "output_order", Integer, nullable=False
    ),  # Order of outputs in query execution
    Column("created_at", DateTime, server_default=func.now()),
)


class PostgresDatabase(Database):
    type: str = "postgres"
    session_maker: sessionmaker

    model_config = {"arbitrary_types_allowed": True}

    def _get_or_create_agent_output(
        self, session, agent_name: str, output_name: str, data: list, data_hash: str
    ) -> int:
        """
        Get existing agent output or create new one, returns the ID.
        """
        # Check if output already exists
        existing = session.execute(
            "SELECT id FROM agent_output WHERE data_hash = :hash", {"hash": data_hash}
        ).first()

        if existing:
            return existing[0]

        # Create new agent output
        stmt = (
            insert(agent_output_table)
            .values(
                agent_name=agent_name,
                output_name=output_name,
                data=data,
                data_hash=data_hash,
            )
            .returning(agent_output_table.c.id)
        )

        return session.execute(stmt).first()[0]

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

            # Insert query
            qd = query.as_db_dict()
            qstmt = (
                insert(taiat_query_table).values(qd).returning(taiat_query_table.c.id)
            )
            query_id = session.execute(qstmt).first()[0]

            # Process each output
            for order, (name, value) in enumerate(data.items()):
                # Extract agent name from output name (assuming format like "agent_name.output_name")
                if "." in name:
                    agent_name, output_name = name.split(".", 1)
                else:
                    agent_name = "unknown"
                    output_name = name

                # Create hash for deduplication
                import hashlib
                import json

                data_str = json.dumps(value, sort_keys=True)
                data_hash = hashlib.md5(data_str.encode()).hexdigest()

                # Get or create agent output
                agent_output_id = self._get_or_create_agent_output(
                    session, agent_name, output_name, value, data_hash
                )

                # Link query to agent output
                link_stmt = insert(query_output_table).values(
                    query_id=query_id,
                    agent_output_id=agent_output_id,
                    output_order=order,
                )
                session.execute(link_stmt)

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

            # Update query
            qd = query.as_db_dict()
            qstmt = update(taiat_query_table).values(qd)
            session.execute(qstmt)

            # Remove existing output links
            session.execute(
                "DELETE FROM query_output WHERE query_id = :query_id",
                {"query_id": query.id},
            )

            # Add new output links (reusing existing agent outputs where possible)
            for order, (name, value) in enumerate(data.items()):
                if "." in name:
                    agent_name, output_name = name.split(".", 1)
                else:
                    agent_name = "unknown"
                    output_name = name

                import hashlib
                import json

                data_str = json.dumps(value, sort_keys=True)
                data_hash = hashlib.md5(data_str.encode()).hexdigest()

                agent_output_id = self._get_or_create_agent_output(
                    session, agent_name, output_name, value, data_hash
                )

                link_stmt = insert(query_output_table).values(
                    query_id=query.id,
                    agent_output_id=agent_output_id,
                    output_order=order,
                )
                session.execute(link_stmt)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
