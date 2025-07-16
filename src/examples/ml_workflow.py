from argparse import ArgumentParser
import os
import getpass
from types import MethodType

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
    parser = ArgumentParser()
    parser.add_argument(
        "--request",
        help="The request to be processed",
        type=str,
        default="Evaluate the performance of a logistic regression model on the diabetes dataset",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the dependency graph for the query run",
    )
    args = parser.parse_args()
    # if "OPENAI_API_KEY" not in os.environ:
    #    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    print("Building agent graph...")

    matcher = MLOutputMatcher(
        llm=get_llm(),
        request=args.request,
    )
    # Get datasets and outputs to choose from.
    matcher.load_dataset_list()
    matcher.load_output_list(agent_roster)
    matcher.get_inputs_and_outputs(args.request)

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
    query = TaiatQuery(query=args.request, visualize_graph=args.visualize)
    state = MLAgentState(
        query=query,
        data={
            "dataset_name": matcher.get_dataset(),
            "model_name": matcher.get_model_name(),
        },
    )
    result = engine.run(state)
    if args.visualize and isinstance(result, tuple):
        state, visualization = result
        print("\n" + "=" * 50)
        print("GRAPH VISUALIZATION")
        print("=" * 50)
        if visualization:
            print("DOT source code for the dependency graph:")
            print(visualization)

            # Save DOT source to file
            with open("taiat_graph.dot", "w") as f:
                f.write(visualization)
            print("\nGraph saved to 'taiat_graph.dot'")

            # Try to render and display the graph
            try:
                import graphviz

                dot = graphviz.Source(visualization)
                dot.render("taiat_graph", format="png", cleanup=True)
                print("Graph rendered as 'taiat_graph.png'")

                # Try to display the image if we're in an environment that supports it
                try:
                    from IPython.display import Image, display

                    display(Image(filename="taiat_graph.png"))
                    print("Graph displayed above")
                except ImportError:
                    print(
                        "IPython not available for display, but image saved as 'taiat_graph.png'"
                    )
                except Exception as e:
                    print(f"Could not display image: {e}")
                    print("Image saved as 'taiat_graph.png' - open it manually to view")

            except Exception as e:
                print(f"Could not render graph: {e}")
                print(
                    "You can render it manually using: dot -Tpng taiat_graph.dot -o graph.png"
                )
        else:
            print("Visualization could not be created (graphviz may not be installed)")
    else:
        state = result
    print(f"\nRequested outputs: {state['query'].inferred_goal_output}")
    for output in state["query"].inferred_goal_output:
        print(f"Results for {output.name}: {state['data'][output.name]}")


if __name__ == "__main__":
    main()
