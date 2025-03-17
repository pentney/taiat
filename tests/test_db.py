import json

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import unittest

from taiat.base import TaiatQuery, AgentGraphNode, AgentData
from taiat.db import PostgresDatabase, taiat_query_table, taiat_output_table

class TestDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database
        cls.engine = create_engine('sqlite:///:memory:')

        # deal with discrepancy between sqlite test and postgres model
        idata_column = taiat_query_table.c.intermediate_data
        idata_column.type = JSON()
        output_column = taiat_output_table.c.data
        output_column.type = JSON()
        path_column = taiat_query_table.c.path
        path_column.type = JSON()
        # Create all tables from your existing models
        taiat_query_table.metadata.create_all(cls.engine)
        taiat_output_table.metadata.create_all(cls.engine)
        
        # Create a session factory
        cls.session_maker = sessionmaker(bind=cls.engine)

    def test_add_run(self):
        db = PostgresDatabase(
            session_maker=self.session_maker
        )
        db.add_run(
            query=TaiatQuery(
                query='Give me a TDE summary',
                inferred_goal_output='td_summary',
                intermediate_data=['tde_data', 'ppi_data', 'cex_data', 'dea_data'],
                status='success',
                path=[
                    AgentGraphNode(
                        name="dea_analysis",
                        inputs=[AgentData(name="dataset", data="this should be clobbered")],
                        outputs=[AgentData(name="dea_data")],
                    ),
                    AgentGraphNode(
                        name="ppi_analysis",
                        inputs=[AgentData(name="dataset")],
                        outputs=[AgentData(name="ppi_data")],
                    ),
                    AgentGraphNode(
                        name="cex_analysis",
                        inputs=[AgentData(name="dataset")],
                        outputs=[AgentData(name="cex_data", data="this too")],
                    ),
                    AgentGraphNode(
                        name="tde_analysis",
                        inputs=[AgentData(name="ppi_data"), AgentData(name="cex_data"), AgentData(name="dea_data")],
                        outputs=[AgentData(name="tde_data")],
                    ),
                    AgentGraphNode(
                        name="td_summary",
                        inputs=[AgentData(name="tde_data")],
                        outputs=[AgentData(name="td_summary")],
                    ),
                ],
            ),
            data={
                'td_summary': 'TDE summary',
                'tde_data': 'TDE data',
                'ppi_data': 'PPI data',
                'cex_data': 'CEX data',
                'dea_data': 'DEA data',
            }
        )
        session = self.session_maker()
        self.assertEqual(session.query(taiat_query_table).count(), 1)
        self.assertEqual(session.query(taiat_output_table).count(), 5)

if __name__ == '__main__':
    unittest.main()