# Machine learning agents for a Taiat workflow.
import tempfile
import kaggle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData, State

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

class MLAgentState(State):
    model: str
    model_name: str
    model_params: dict
    model_results: dict
    dataset: pd.DataFrame

def load_dataset(state: MLAgentState) -> MLAgentState:
    kaggle.api.authenticate()
    tmpdir = tempfile.mkdtemp()
    kaggle.api.dataset_download_files(
        state.data['dataset_name'],
        path=f"{tmpdir}/data",
        unzip=False,
    )
    state.data["dataset"], state.data["dataset_test"] = \
        train_test_split(state.data["dataset"], test_size=0.2, random_state=42)
    return state

def logistic_regression(state: MLAgentState) -> MLAgentState:
    model = LogisticRegression(**state.data["model_params"])
    model.fit(state.data["dataset"].drop(columns=["Outcome"]), state.data["dataset"]["Outcome"])
    state.data["model"] = model
    return state


def random_forest(state: MLAgentState) -> MLAgentState:
    model = RandomForestClassifier(**state.data["model_params"])
    model.fit(state.data["dataset"].drop(columns=["Outcome"]), state.data["dataset"]["Outcome"])
    state.data["model"] = model
    return state

def nearest_neighbors(state: MLAgentState) -> MLAgentState:
    model = KNeighborsClassifier(**state.data["model_params"])
    model.fit(state.data["dataset"].drop(columns=["Outcome"]), state.data["dataset"]["Outcome"])
    state.data["model"] = model
    return state

def clustering(state: MLAgentState) -> MLAgentState:
    model = KMeans(**state.data["model_params"])
    model.fit(state.data["dataset"].drop(columns=["Outcome"]))
    state.data["model"] = model
    return state

def predict_and_generate_report(state: MLAgentState) -> MLAgentState:
    state.data["model_preds"] = state.data["model"].predict(state.data["dataset_test"].drop(columns=["Outcome"]))
    state.data["model_report"] = classification_report(state.data["dataset_test"]["Outcome"], state.data["model_preds"])
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

Dataset: {state.data['dataset_name']}
Model: {state.data['model_name']}
Report:
{state.data['model_report']}
"""
    state.data['summary'] = llm.invoke(summary_prompt)
    return state

agent_roster = AgentGraphNodeSet(
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
        inputs=[AgentData(name="dataset", data="")],    
        outputs=[AgentData(name="model", data="")],
    ),
    AgentGraphNode(
        name="random_forest",
        description="Train a random forest model",
        function=random_forest,
        inputs=[AgentData(name="dataset", data="")],
        outputs=[AgentData(name="model", data="")],
    ),
    AgentGraphNode(
        name="nearest_neighbors",
        description="Train a nearest neighbors model",
        function=nearest_neighbors,
        inputs=[AgentData(name="dataset", data="")],
        outputs=[AgentData(name="model", data="")],
    ),
    AgentGraphNode(
        name="clustering",
        description="Train a clustering model",
        function=clustering,
        inputs=[AgentData(name="dataset", data="")],
        outputs=[AgentData(name="model", data="")],
    ),
    AgentGraphNode(
        name="predict_and_generate_report",
        description="Make a prediction and generate a report",
        function=predict_and_generate_report,
        inputs=[AgentData(name="model", data="")],
        outputs=[AgentData(name="model_preds", data=""), AgentData(name="model_report", data="")],
    ),
    AgentGraphNode(
        name="results_analysis",
        description="Analyze the results",
        function=results_analysis,
        inputs=[
            AgentData(name="dataset_name", data=""),
            AgentData(name="model_name", data=""),
            AgentData(name="model_report", data="")
        ],
        outputs=[AgentData(name="summary", data="")],
    ),
)
