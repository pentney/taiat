# Machine learning agents for a Taiat workflow.
import tempfile
import pandas as pd
from pathlib import Path

import kaggle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData, State

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# Initialize llm after environment variables are loaded
llm = None


def get_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini")
    return llm


class MLAgentState(State):
    model: str
    model_name: str
    model_params: dict
    model_results: dict
    dataset: pd.DataFrame


def load_dataset(state: MLAgentState) -> MLAgentState:
    kaggle.api.authenticate()
    tmpdir = tempfile.mkdtemp()
    print("Downloading dataset: ", state["data"]["dataset_name"], "to", tmpdir)
    kaggle.api.dataset_download_files(
        state["data"]["dataset_name"],
        path=f"{tmpdir}/data",
        unzip=True,
    )
    csv_file = next(Path(f"{tmpdir}/data").glob("*.csv"))
    state["data"]["dataset"] = pd.read_csv(csv_file)

    categorical_columns = (
        state["data"]["dataset"]
        .select_dtypes(include=["object", "category", "string"])
        .columns
    )
    categorical_columns = [col for col in categorical_columns if col != "class"]

    print("categorical_columns: ", categorical_columns)
    # Apply one-hot encoding to categorical columns
    if categorical_columns:
        state["data"]["dataset"] = pd.get_dummies(
            state["data"]["dataset"], columns=categorical_columns, drop_first=False
        )

    state["data"]["dataset"], state["data"]["dataset_test"] = train_test_split(
        state["data"]["dataset"], test_size=0.2, random_state=42
    )

    # Find the target column - diabetes datasets often use different names
    if "class" in state["data"]["dataset"].columns:
        label = "class"
    elif "target" in state["data"]["dataset"].columns:
        label = "target"
    elif "diabetes" in state["data"]["dataset"].columns:
        label = "diabetes"
    else:
        # Use the last column as the target
        label = state["data"]["dataset"].columns[-1]

    state["data"]["label"] = label
    # Do NOT drop the label column here
    state["data"]["model_params"] = {}
    print("downloaded!")
    return state


def logistic_regression(state: MLAgentState) -> MLAgentState:
    model = LogisticRegression(**state["data"]["model_params"])
    print("dataset: ", state["data"]["dataset"].head())
    model.fit(
        state["data"]["dataset"].drop(columns=[state["data"]["label"]]),
        state["data"]["dataset"][state["data"]["label"]],
    )
    state["data"]["model"] = model
    return state


def random_forest(state: MLAgentState) -> MLAgentState:
    model = RandomForestClassifier(**state["data"]["model_params"])
    model.fit(
        state["data"]["dataset"].drop(columns=[state["data"]["label"]]),
        state["data"]["dataset"][state["data"]["label"]],
    )
    state["data"]["model"] = model
    return state


def nearest_neighbors(state: MLAgentState) -> MLAgentState:
    model = KNeighborsClassifier(**state["data"]["model_params"])
    model.fit(
        state["data"]["dataset"].drop(columns=[state["data"]["label"]]),
        state["data"]["dataset"][state["data"]["label"]],
    )
    state["data"]["model"] = model
    return state


def clustering(state: MLAgentState) -> MLAgentState:
    model = KMeans(**state["data"]["model_params"])
    model.fit(state["data"]["dataset"].drop(columns=[state["data"]["label"]]))
    state["data"]["model"] = model
    return state


def predict_and_generate_report(state: MLAgentState) -> MLAgentState:
    state["data"]["model_preds"] = state["data"]["model"].predict(
        state["data"]["dataset_test"].drop(columns=[state["data"]["label"]])
    )
    state["data"]["model_report"] = classification_report(
        state["data"]["dataset_test"][state["data"]["label"]],
        state["data"]["model_preds"],
    )
    return state


summary_prompt_prefix = """
You are a data scientist.
You are given a dataset and a model.
You are given a report of the model's performance.

Using the report, summarize the dataset, the model, and the model's performance in a few sentences.
"""


def results_analysis(state: MLAgentState) -> MLAgentState:
    summary_prompt = f"""
{summary_prompt_prefix}

Dataset: {state["data"]["dataset_name"]}
Model: {state["data"].get("model_name")}
Report:
{state["data"]["model_report"]}
"""
    state["data"]["summary"] = get_llm().invoke(summary_prompt).content
    return state


agent_roster = AgentGraphNodeSet(
    nodes=[
        AgentGraphNode(
            name="load_dataset",
            description="Load the dataset",
            function=load_dataset,
            inputs=[AgentData(name="dataset_name", data="")],
            outputs=[AgentData(name="dataset", data="")],
        ),
        AgentGraphNode(
            name="logistic_regression",
            description="Train a logistic regression model",
            function=logistic_regression,
            inputs=[AgentData(name="dataset")],
            outputs=[
                AgentData(
                    name="model", parameters={"model_type": "logistic_regression"}
                ),
                AgentData(name="model_params", data={}),
            ],
        ),
        AgentGraphNode(
            name="random_forest",
            description="Train a random forest model",
            function=random_forest,
            inputs=[AgentData(name="dataset")],
            outputs=[
                AgentData(name="model", parameters={"model_type": "random_forest"})
            ],
        ),
        AgentGraphNode(
            name="nearest_neighbors",
            description="Train a nearest neighbors model",
            function=nearest_neighbors,
            inputs=[AgentData(name="dataset")],
            outputs=[
                AgentData(name="model", parameters={"model_type": "nearest_neighbors"})
            ],
        ),
        AgentGraphNode(
            name="clustering",
            description="Train a clustering model",
            function=clustering,
            inputs=[AgentData(name="dataset")],
            outputs=[AgentData(name="model", parameters={"model_type": "clustering"})],
        ),
        AgentGraphNode(
            name="predict_and_generate_report",
            description="Make a prediction and generate a report",
            function=predict_and_generate_report,
            inputs=[AgentData(name="model")],
            outputs=[AgentData(name="model_preds"), AgentData(name="model_report")],
        ),
        AgentGraphNode(
            name="results_analysis",
            description="Analyze the results",
            function=results_analysis,
            inputs=[
                AgentData(name="dataset_name", data=""),
                AgentData(name="model_report", data=""),
            ],
            outputs=[AgentData(name="summary", data="")],
        ),
    ]
)
