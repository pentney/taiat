import pandas as pd
from taiat.builder import AgentData, AgentGraphNode, AgentGraphNodeSet, State
def cex_analysis(state: State) -> State:
    state["data"]["cex_data"] = state["data"]["dea_data"].model_copy(deep=True)
    state["data"]["cex_data"].data["cex"] = 1.0
    return state
    return state

def ppi_analysis(state: State) -> State:
    state["data"]["ppi_data"] = state["data"]["dea_data"].model_copy(deep=True)
    state["data"]["ppi_data"].data["ppi"] = 2.0
    return state

def dea_analysis(state: State) -> State:
    state["data"]["dea_data"] = state["data"]["dataset"].model_copy(deep=True)
    state["data"]["dea_data"].data["dea"] = 0.0
    return state

def dea_analysis_w_params(state: State) -> State:
    state["data"]["dea_data"] = state["data"]["dataset"].model_copy(deep=True)
    state["data"]["dea_data"].data["dea"] = -1.0
    return state

def tde_analysis(state: State) -> State:
    state["data"]["tde_data"] = state["data"]["cex_data"].model_copy(deep=True)
    state["data"]["tde_data"].data = pd.merge(
        state["data"]["tde_data"].data, state["data"]["ppi_data"].data, on=["id", "dea"], how="left"
    )
    print("tde_data", state["data"]["tde_data"].data)
    state["data"]["tde_data"].data["score"] = \
        state["data"]["tde_data"].data["cex"] + state["data"]["tde_data"].data["ppi"] + \
        state["data"]["tde_data"].data["dea"]
    return state

def td_summary(state: State) -> State:
    state["data"]["td_summary"] = \
        f"summary of TD. sum: {state['data']['tde_data'].data['score'].sum()}"
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

TestNodeSetWithParams = TestNodeSet.model_copy(deep=True)
TestNodeSetWithParams.nodes[2].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes[1].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes.append(
        AgentGraphNode(
            name="dea_analysis_w_params",
            function=dea_analysis_w_params,
            inputs=[AgentData(name="dataset")],
            outputs=[AgentData(name="dea_data", parameters={"param": "value"})],
        )
)
