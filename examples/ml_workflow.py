from argparse import ArgumentParser
import os
import getpass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from taiat.base import AgentGraphNodeSet, AgentData, State, TaiatQuery
from taiat.builder import TaiatBuilder
from taiat.engine import TaiatEngine
from taiat.examples.ml_agents import (
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
from taiat.examples.ml_output_matcher import MLOutputMatcher


def main():
    load_dotenv()
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
    builder = TaiatBuilder(agent_roster)

    matcher = MLOutputMatcher(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        request=args.request,
    )
    # Get datasets and outputs to choose from.
    matcher.load_dataset_list()
    matcher.load_output_list(agent_roster)
    matcher.select_task(args.request)

    print("dataset", matcher.get_dataset())
    print("outputs", matcher.get_outputs())
    engine = TaiatEngine(
        llm_dict={
            "llm": ChatOpenAI(model="gpt-4o-mini"),
        },
        builder=builder,
        output_matcher=matcher.get_outputs(),
    )
    state = MLAgentState(
        query=TaiatQuery(query=args.request),
        data={},
    )
    state = engine.run(state)
    print(state)

if __name__ == "__main__":
    main()
