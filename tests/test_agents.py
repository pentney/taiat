import pandas as pd
from typing import Any, Optional

from taiat.builder import AgentData, AgentGraphNode, AgentGraphNodeSet, State, TaiatQuery

class TestState(State):
    def __init__(self):
        self.query = TaiatQuery()
        self.input = {"dataset":pd.DataFrame()}

def cex_analysis(state: TestState) -> TestState:
    state["cex_data"] = state["dataset"].copy()
    return state

def ppi_analysis(state: TestState) -> TestState:
    state["ppi_data"] = state["dataset"].copy()
    return state

def dea_analysis(state: TestState) -> TestState:
    state["dea_data"] = state["dataset"].copy()
    return state

def tde_analysis(state: TestState) -> TestState:
    state["tde_data"] = state["dataset"].copy()
    return state

TestNodeSet = AgentGraphNodeSet(
    nodes=[
        AgentGraphNode(
            name="dea_analysis",
            function=dea_analysis,
            inputs=[AgentData(name="dataset")],
            outputs=[AgentData(name="dea_data")],
        ),
        AgentGraphNode(
            name="ppi_analysis",
            function=ppi_analysis,
            inputs=[AgentData(name="dea_data")],
            outputs=[AgentData(name="ppi_data")],
        ),
        AgentGraphNode(
            name="cex_analysis",
            function=cex_analysis,
            inputs=[AgentData(name="dea_data")],
            outputs=[AgentData(name="cex_data")],
        ),
        AgentGraphNode(
            name="tde_analysis",
            function=tde_analysis,
            inputs=[AgentData(name="cex_data"), AgentData(name="ppi_data")],
            outputs=[AgentData(name="tde_data")],
        ),
    ],
)
