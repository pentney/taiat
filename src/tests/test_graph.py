import unittest

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeChatModel

from taiat.engine import TaiatEngine, OutputMatcher
from taiat.builder import (
    AgentData,
    TaiatBuilder,
    TaiatQuery,
    TAIAT_TERMINAL_NODE,
    State,
)
from .test_agents import TestNodeSet, TestNodeSetWithParams


class DummyModel(FakeChatModel):
    pass


class SimpleOutputMatcher(OutputMatcher):
    def get_outputs(self, query: str) -> list[AgentData]:
        return [
            AgentData(
                name="four_summary",
                parameters={},
            )
        ]


class TestGraph(unittest.TestCase):
    def _build_graph(self, node_set):
        llm = DummyModel()
        builder = TaiatBuilder(llm)
        builder.build(
            node_set=node_set,
            inputs=[AgentData(name="dataset", parameters={})],
            terminal_nodes=["four_summary"],
        )
        return builder

    def test_build_graph(self):
        builder = self._build_graph(TestNodeSet)
        # Test that the builder has the expected data structures
        assert builder.node_set is not None
        assert len(builder.node_set.nodes) > 0

        # Test that the data_source and data_dependence are populated
        assert len(builder.data_source) > 0
        assert len(builder.data_dependence) > 0

        # Test that we can find the expected nodes
        expected_nodes = [
            "one_analysis",
            "two_analysis",
            "three_analysis",
            "four_analysis",
            "four_summary",
        ]

        actual_nodes = [node.name for node in builder.node_set.nodes]
        for expected_node in expected_nodes:
            assert expected_node in actual_nodes, (
                f"Expected node {expected_node} not found"
            )

    def test_run_graph(self):
        builder = self._build_graph(TestNodeSet)
        engine = TaiatEngine(
            llm=DummyModel(),
            builder=builder,
            output_matcher=SimpleOutputMatcher(),
        )
        query = TaiatQuery(
            query="Give me a FOUR summary",
        )
        state = State(
            query=query,
            data={
                "dataset": AgentData(
                    name="dataset",
                    data=pd.DataFrame(
                        {
                            "id": [1, 2, 3],
                        }
                    ),
                ),
            },
        )
        state = engine.run(state)
        assert query.status == "success", "Error: " + query.error
        assert state["data"]["four_summary"] == "summary of FOUR. sum: 9.0"

    def test_run_graph_with_params(self):
        builder = self._build_graph(TestNodeSetWithParams)
        engine = TaiatEngine(
            llm=DummyModel(),
            builder=builder,
            output_matcher=SimpleOutputMatcher(),
        )
        query = TaiatQuery(
            query="Give me a FOUR summary",
        )
        state = State(
            query=query,
            data={
                "dataset": AgentData(
                    name="dataset",
                    data=pd.DataFrame(
                        {
                            "id": [1, 2, 3],
                        }
                    ),
                ),
            },
        )
        state = engine.run(state)
        assert query.status == "success", "Error: " + query.error
        assert state["data"]["four_summary"] == "summary of FOUR. sum: 6.0"


if __name__ == "__main__":
    unittest.main()
