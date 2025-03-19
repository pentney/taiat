import pandas as pd

from taiat.builder import AgentData, AgentGraphNode, AgentGraphNodeSet, State

def three_analysis(state: State) -> State:
    state["data"]["three_data"] = state["data"]["one_data"].model_copy(deep=True)
    state["data"]["three_data"].data["three"] = 1.0
    return state
    return state

def two_analysis(state: State) -> State:
    state["data"]["two_data"] = state["data"]["one_data"].model_copy(deep=True)
    state["data"]["two_data"].data["two"] = 2.0
    return state

def one_analysis(state: State) -> State:
    state["data"]["one_data"] = state["data"]["dataset"].model_copy(deep=True)
    state["data"]["one_data"].data["one"] = 0.0
    return state

def one_analysis_w_params(state: State) -> State:
    state["data"]["one_data"] = state["data"]["dataset"].model_copy(deep=True)
    state["data"]["one_data"].data["one"] = -1.0
    return state

def four_analysis(state: State) -> State:
    state["data"]["four_data"] = state["data"]["three_data"].model_copy(deep=True)
    state["data"]["four_data"].data = pd.merge(
        state["data"]["four_data"].data, state["data"]["two_data"].data, on=["id", "one"], how="left"
    )
    state["data"]["four_data"].data["score"] = \
        state["data"]["four_data"].data["three"] + state["data"]["four_data"].data["two"] + \
        state["data"]["four_data"].data["one"]
    return state

def four_summary(state: State) -> State:
    state["data"]["four_summary"] = \
        f"summary of FOUR. sum: {state['data']['four_data'].data['score'].sum()}"
    return state

TestNodeSet = AgentGraphNodeSet(
    nodes=[
        AgentGraphNode(
            name="one_analysis",
            function=one_analysis,
            inputs=[AgentData(name="dataset")],
            outputs=[AgentData(name="one_data")],
        ),
        AgentGraphNode(
            name="two_analysis",
            function=two_analysis,
            inputs=[AgentData(name="one_data")],
            outputs=[AgentData(name="two_data")],
        ),
        AgentGraphNode(
            name="three_analysis",
            function=three_analysis,
            inputs=[AgentData(name="one_data")],
            outputs=[AgentData(name="three_data")],
        ),
        AgentGraphNode(
            name="four_analysis",
            function=four_analysis,
            inputs=[AgentData(name="three_data"), AgentData(name="two_data")],
            outputs=[AgentData(name="four_data")],
        ),
        AgentGraphNode(
            name="four_summary",
            function=four_summary,
            inputs=[AgentData(name="four_data")],
            outputs=[AgentData(name="four_summary")],
        )
    ],
)

TestNodeSetWithParams = TestNodeSet.model_copy(deep=True)
TestNodeSetWithParams.nodes[2].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes[1].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes.append(
        AgentGraphNode(
            name="one_analysis_w_params",
            function=one_analysis_w_params,
            inputs=[AgentData(name="dataset")],
            outputs=[AgentData(name="one_data", parameters={"param": "value"})],
        )
)
