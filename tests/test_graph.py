import unittest

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeChatModel

from taiat.engine import TaiatEngine
from taiat.builder import AgentData, TaiatBuilder, TaiatQuery, TAIAT_TERMINAL_NODE
from test_agents import TestNodeSet, TestState

class DummyModel(FakeChatModel):
    pass

class TestGraph(unittest.TestCase):
    def _build_graph(self):
        llm = DummyModel()
        builder = TaiatBuilder(llm)
        graph = builder.build(
            state=TestState(
                data={
                    "dataset": AgentData(name="dataset", data=pd.DataFrame()),
                },
            ),
            node_set=TestNodeSet,
            inputs=["dataset"],
            terminal_nodes=["td_summary"],
        )
        return builder, graph

    def test_build_graph(self):
        _, graph = self._build_graph()
        assert graph is not None
        nodes = graph.get_graph().nodes
        print("nodes", nodes)
        expected_nodes = ["__start__", "dea_analysis", "cex_analysis", "ppi_analysis",
                          "tde_analysis", "td_summary", TAIAT_TERMINAL_NODE, "__end__"]
        assert len(nodes) == len(expected_nodes)
        print(nodes)
        for node in nodes.keys():
            assert node in expected_nodes
            expected_nodes.remove(node)
        assert len(expected_nodes) == 0
        edges = graph.get_graph().edges
        print(edges)
        expected_edges = [
            ("__start__", "dea_analysis"),
            ("dea_analysis", "cex_analysis"),
            ("dea_analysis", "ppi_analysis"),
            ("ppi_analysis", "tde_analysis"),
            ("cex_analysis", "tde_analysis"),
            ("tde_analysis", "td_summary"),
            ("td_summary", TAIAT_TERMINAL_NODE),
            (TAIAT_TERMINAL_NODE, "__end__"),
        ]
        for edge in edges:
            print(edge.source, edge.target)
            assert (edge.source, edge.target) in expected_edges
            expected_edges.remove((edge.source, edge.target))
        assert len(expected_edges) == 0

    def test_run_graph(self):
        builder, graph = self._build_graph()
        engine = TaiatEngine(
            llm_dict={
                "llm": DummyModel(),
            },
            graph=graph,
            builder=builder,
            output_matcher=lambda x: ["td_summary"],
        )
        engine.run(
            query=TaiatQuery(
                query="",
            ),
            data={
                "dataset": AgentData(name="dataset", data=pd.DataFrame()),
            },
        )

if __name__ == "__main__":
    unittest.main()