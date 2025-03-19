from argparse import ArgumentParser
import os
import getpass
from types import MethodType

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from taiat.base import AgentGraphNodeSet, AgentData, State, TaiatQuery
from taiat.builder import TaiatBuilder
from taiat.engine import TaiatEngine
from taiat.examples.ml_agents import (
    agent_roster,
    llm,
    MLAgentState,
    load_dataset,
    logistic_regression,
    random_forest,
    nearest_neighbors,
    clustering,
    predict_and_generate_report,
    results_analysis,
)
from taiat.examples.ml_output_matcher import MLOutputMatcher


def main(): 
    parser = ArgumentParser()
    parser.add_argument(
        "--request",
        help="The request to be processed",
        type=str,
        default="Evaluate the performance of a logistic regression model on the diabetes dataset")
    args = parser.parse_args()
    #if "OPENAI_API_KEY" not in os.environ:
    #    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")



    print("Building agent graph...")

    matcher = MLOutputMatcher(
        llm=llm,
        request=args.request,
    )
    # Get datasets and outputs to choose from.
    matcher.load_dataset_list()
    matcher.load_output_list(agent_roster)
    matcher.get_inputs_and_outputs(args.request)

    builder = TaiatBuilder(llm=llm, verbose=True)
    builder.build(
        agent_roster,
        inputs = [AgentData(name="dataset_name", parameters={}, data=matcher.get_dataset())],
        terminal_nodes = ["results_analysis"],
    )

    print("dataset", matcher.get_dataset())
    print("outputs", matcher.get_outputs(""))
    engine = TaiatEngine(
        llm_dict={
            "llm": ChatOpenAI(model="gpt-4o-mini"),
        },
        builder=builder,
        node_set=agent_roster,
        output_matcher=matcher,
    )
    state = MLAgentState(
        query=TaiatQuery(query=args.request),
        data={
            "dataset_name": matcher.get_dataset(),
            "model_name": matcher.get_model_name(),
        },
    )
    state = engine.run(state)
    print(f"Requested outputs: {state['query'].inferred_goal_output}")
    for output in state["query"].inferred_goal_output:
        print(f"Results for {output.name}: {state['data'][output.name]}")

if __name__ == "__main__":
    main()
