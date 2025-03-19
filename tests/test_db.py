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
                query='Give me a four summary',
                inferred_goal_output='four_summary',
                intermediate_data=['four_data', 'three_data', 'two_data', 'dea_data'],
                status='success',
                path=[
                    AgentGraphNode(
                        name="one_analysis",
                        inputs=[AgentData(name="dataset", data="this should be clobbered")],
                        outputs=[AgentData(name="one_data")],
                    ),
                    AgentGraphNode(
                        name="three_analysis",
                        inputs=[AgentData(name="dataset")],
                        outputs=[AgentData(name="three_data")],
                    ),
                    AgentGraphNode(
                        name="two_analysis",
                        inputs=[AgentData(name="dataset")],
                        outputs=[AgentData(name="two_data", data="this too")],
                    ),
                    AgentGraphNode(
                        name="four_analysis",
                        inputs=[AgentData(name="three_data"), AgentData(name="two_data"), AgentData(name="dea_data")],
                        outputs=[AgentData(name="four_data")],
                    ),
                    AgentGraphNode(
                        name="four_summary",
                        inputs=[AgentData(name="four_data")],
                        outputs=[AgentData(name="four_summary")],
                    ),
                ],
            ),
            data={
                'four_summary': 'FOUR summary',
                'four_data': 'FOUR data',
                'three_data': 'THREE data',
                'two_data': 'TWO data',
                'dea_data': 'ONE data',
            }
        )
        session = self.session_maker()
        self.assertEqual(session.query(taiat_query_table).count(), 1)
        self.assertEqual(session.query(taiat_output_table).count(), 5)

if __name__ == '__main__':
    unittest.main()