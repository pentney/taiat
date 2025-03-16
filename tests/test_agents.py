import pandas as pd
from typing import Any, Optional

from taiat.builder import AgentData, AgentGraphNode, AgentGraphNodeSet, State, TaiatQuery

class TestState(State):
    def __init__(self):
        self.query = TaiatQuery()
        self.input = {"dataset":pd.DataFrame({
            "id": [1, 2, 3]
        })}

def cex_analysis(state: TestState) -> TestState:
    print("cex_analysis", state)
    state["data"]["cex_data"] = state["data"]["dataset"].copy()
    state["data"]["cex_data"]["cex"] = 1.0
    return state

def ppi_analysis(state: TestState) -> TestState:
    print("ppi_analysis", state)
    state["data"]["ppi_data"] = state["data"]["dataset"].copy()
    state["data"]["ppi_data"]["ppi"] = 2.0
    return state

def dea_analysis(state: TestState) -> TestState:
    print("dea_analysis", state)
    state["data"]["dea_data"] = state["data"]["dataset"].copy()
    state["data"]["dea_data"]["dea"] = 0.0
    return state

def tde_analysis(state: TestState) -> TestState:
    print("tde_analysis", state)
    state["data"]["tde_data"] = state["data"]["cex_data"].copy()
    state["data"]["tde_data"].join(state["data"]["ppi_data"], on="id", how="inner")
    state["data"]["tde_data"]["score"] = state["data"]["tde_data"]["cex"] + state["data"]["tde_data"]["ppi"]
    return state

def td_summary(state: TestState) -> TestState:
    print("td_summary", state)
    state["data"]["td_summary"] = f"summary of TD. sum: {state['data']['tde_data']['score'].sum()}"
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
        AgentGraphNode(
            name="td_summary",
            function=td_summary,
            inputs=[AgentData(name="tde_data")],
            outputs=[AgentData(name="td_summary")],
        )
    ],
)
