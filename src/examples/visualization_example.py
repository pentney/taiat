"""
Example demonstrating the graph visualization feature in Taiat.
This example shows how to request and use graph visualizations.
"""

import os
import getpass
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from taiat.base import AgentGraphNodeSet, AgentData, State, TaiatQuery
from taiat.builder import TaiatBuilder
from taiat.engine import TaiatEngine
from examples.ml_agents import (
    agent_roster,
    get_llm,
    MLAgentState,
    load_dataset,
    logistic_regression,
    random_forest,
    nearest_neighbors,
    clustering,
    predict_and_generate_report,
    results_analysis,
)
from examples.ml_output_matcher import MLOutputMatcher


def main():
    # Example request
    request = "Evaluate the performance of a logistic regression model on the diabetes dataset"

    print("Building agent graph...")

    matcher = MLOutputMatcher(
        llm=get_llm(),
        request=request,
    )
    # Get datasets and outputs to choose from.
    matcher.load_dataset_list()
    matcher.load_output_list(agent_roster)
    matcher.get_inputs_and_outputs(request)

    builder = TaiatBuilder(llm=get_llm(), verbose=True)
    builder.build(
        agent_roster,
        inputs=[
            AgentData(name="dataset_name", parameters={}, data=matcher.get_dataset())
        ],
        terminal_nodes=["results_analysis"],
    )

    print("dataset", matcher.get_dataset())
    print("outputs", matcher.get_outputs(""))
    engine = TaiatEngine(
        llm=get_llm(),
        builder=builder,
        output_matcher=matcher,
    )

    # Create a query with visualization enabled
    query = TaiatQuery(query=request, visualize_graph=True)

    state = MLAgentState(
        query=query,
        data={
            "dataset_name": matcher.get_dataset(),
            "model_name": matcher.get_model_name(),
        },
    )

    # Run the engine - it will return both state and visualization
    result = engine.run(state)

    if isinstance(result, tuple):
        state, visualization = result
        print("\n" + "=" * 50)
        print("GRAPH VISUALIZATION")
        print("=" * 50)
        if visualization:
            print("DOT source code for the dependency graph:")
            print(visualization)

            # Optionally save to a file
            with open("taiat_graph.dot", "w") as f:
                f.write(visualization)
            print("\nGraph saved to 'taiat_graph.dot'")
            print("You can render it using: dot -Tpng taiat_graph.dot -o graph.png")
        else:
            print("Visualization could not be created (graphviz may not be installed)")
    else:
        state = result

    print(f"\nRequested outputs: {state['query'].inferred_goal_output}")
    for output in state["query"].inferred_goal_output:
        print(f"Results for {output.name}: {state['data'][output.name]}")


if __name__ == "__main__":
    main()
