from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import unittest

from taiat.base import TaiatQuery
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
                error=None,
                path=["dea_analysis", "ppi_analysis", "cex_analysis",
                      "tde_analysis", "td_summary"],
            ),
            data={
                'td_summary': 'TDE summary',
                'tde_data': 'TDE data',
                'ppi_data': 'PPI data',
                'cex_data': 'CEX data',
                'dea_data': 'DEA data',
            }
        )
        self.assertEqual(db.session_maker.query(taiat_query_table).count(), 1)
        self.assertEqual(db.session_maker.query(taiat_output_table).count(), 5)

if __name__ == '__main__':
    unittest.main()