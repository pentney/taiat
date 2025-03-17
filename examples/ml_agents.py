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


class MLAgentState(State):
    model: str
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

def prediction(state: MLAgentState) -> MLAgentState:
    state.data["model_results"] = state.data["model"].predict(state.data["dataset_test"].drop(columns=["Outcome"]))
    return state

def results_analysis(state: MLAgentState) -> MLAgentState:
    return state



