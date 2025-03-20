import unittest

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeChatModel

from taiat.engine import TaiatEngine, OutputMatcher
from taiat.builder import AgentData, TaiatBuilder, TaiatQuery, TAIAT_TERMINAL_NODE, State
from test_agents import TestNodeSet, TestNodeSetWithParams

class DummyModel(FakeChatModel):
    pass

class SimpleOutputMatcher(OutputMatcher):
    def get_outputs(self, query: str) -> list[AgentData]:
        return [AgentData(
            name="four_summary",
            parameters={},
        )]

class TestGraph(unittest.TestCase):
    def _build_graph(self, node_set):
        llm = DummyModel()
        builder = TaiatBuilder(llm)
        graph = builder.build(
            node_set=node_set,
            inputs=[AgentData(name="dataset", parameters={})],
            terminal_nodes=["four_summary"],
        )
        return builder, graph

    def test_build_graph(self):
        _, graph = self._build_graph(TestNodeSet)
        assert graph is not None
        nodes = graph.get_graph().nodes
        expected_nodes = ["__start__", "one_analysis", "two_analysis", "three_analysis",
                          "four_analysis", "four_summary", TAIAT_TERMINAL_NODE, "__end__"]
        assert len(nodes) == len(expected_nodes)
        for node in nodes.keys():
            assert node in expected_nodes
            expected_nodes.remove(node)
        assert len(expected_nodes) == 0
        edges = graph.get_graph().edges
        expected_edges = [
            ("__start__", "one_analysis"),
            ("one_analysis", "two_analysis"),
            ("one_analysis", "three_analysis"),
            ("three_analysis", "four_analysis"),
            ("two_analysis", "four_analysis"),
            ("four_analysis", "four_summary"),
            ("four_summary", TAIAT_TERMINAL_NODE),
            (TAIAT_TERMINAL_NODE, "__end__"),
        ]
        for edge in edges:
            assert (edge.source, edge.target) in expected_edges
            expected_edges.remove((edge.source, edge.target))
        assert len(expected_edges) == 0

    def test_run_graph(self):
        builder, _ = self._build_graph(TestNodeSet)
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
                    data=pd.DataFrame({
                    "id": [1, 2, 3],
                })),
            },
        )
        state = engine.run(state)
        assert query.status == "success", "Error: " + query.error
        assert state["data"]["four_summary"] == "summary of FOUR. sum: 9.0"

    def test_run_graph_with_params(self):
        builder, _ = self._build_graph(TestNodeSetWithParams)
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
                    data=pd.DataFrame({
                    "id": [1, 2, 3],
                })),
            },
        )
        state = engine.run(state)
        assert query.status == "success", "Error: " + query.error
        assert state["data"]["four_summary"] == "summary of FOUR. sum: 6.0"


if __name__ == "__main__":
    unittest.main()