from argparse import ArgumentParser

from ml_agents import (
    agent_roster,
    MLAgentState,
    load_dataset,
    logistic_regression,
    random_forest,
    nearest_neighbors,
    clustering,
    predict_and_generate_report,
    results_analysis,
)

from taiat.base import AgentGraphNodeSet, AgentData, State
from taiat.builder import TaiatBuilder

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--request",
        help="The request to be processed",
        type=str,
        default="Evaluate the performance of a logistic regression model on the diabetes dataset")
    print("Building agent graph...")
    agent_graph = AgentGraphNodeSet(
        agent_roster,
    )
    builder = TaiatBuilder(agent_graph)

    matcher = MLOutputMatcher(
        args.request,
    )
    engine = TaiatEngine(
        llm_dict={
            "llm": DummyModel(),
        },
        graph=graph,
        builder=builder,
        output_matcher=matcher.select_outputs,
    )
    graph = builder.build(
        state=MLAgentState(),
        node_set=agent_graph,
        inputs=[],
        terminal_nodes=[],
    )


if __name__ == "__main__":
    main()
