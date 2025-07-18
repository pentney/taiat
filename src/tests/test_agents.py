import pandas as pd

from taiat.builder import AgentData, AgentGraphNode, AgentGraphNodeSet, State


def three_analysis(state: State) -> State:
    state["data"]["three_data"] = state["data"]["one_data"].model_copy(deep=True)
    state["data"]["three_data"].data["three"] = 1.0
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
        state["data"]["four_data"].data,
        state["data"]["two_data"].data,
        on=["id", "one"],
        how="left",
    )
    state["data"]["four_data"].data["score"] = (
        state["data"]["four_data"].data["three"]
        + state["data"]["four_data"].data["two"]
        + state["data"]["four_data"].data["one"]
    )
    return state


def four_summary(state: State) -> State:
    state["data"]["four_summary"] = (
        f"summary of FOUR. sum: {state['data']['four_data'].data['score'].sum()}"
    )
    return state


TestNodeSet = AgentGraphNodeSet(
    nodes=[
        AgentGraphNode(
            name="one_analysis",
            description="Perform one analysis",
            function=one_analysis,
            inputs=[AgentData(name="dataset", parameters={}, description="Input dataset", data=None)],
            outputs=[AgentData(name="one_data", parameters={}, description="One analysis result", data=None)],
        ),
        AgentGraphNode(
            name="two_analysis",
            description="Perform two analysis",
            function=two_analysis,
            inputs=[AgentData(name="one_data", parameters={}, description="One analysis result", data=None)],
            outputs=[AgentData(name="two_data", parameters={}, description="Two analysis result", data=None)],
        ),
        AgentGraphNode(
            name="three_analysis",
            description="Perform three analysis",
            function=three_analysis,
            inputs=[AgentData(name="one_data", parameters={}, description="One analysis result", data=None)],
            outputs=[AgentData(name="three_data", parameters={}, description="Three analysis result", data=None)],
        ),
        AgentGraphNode(
            name="four_analysis",
            description="Perform four analysis",
            function=four_analysis,
            inputs=[
                AgentData(name="three_data", parameters={}, description="Three analysis result", data=None),
                AgentData(name="two_data", parameters={}, description="Two analysis result", data=None)
            ],
            outputs=[AgentData(name="four_data", parameters={}, description="Four analysis result", data=None)],
        ),
        AgentGraphNode(
            name="four_summary",
            description="Generate four summary",
            function=four_summary,
            inputs=[AgentData(name="four_data", parameters={}, description="Four analysis result", data=None)],
            outputs=[AgentData(name="four_summary", parameters={}, description="Four summary", data=None)],
        ),
    ],
)

TestNodeSetWithParams = TestNodeSet.model_copy(deep=True)
TestNodeSetWithParams.nodes[2].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes[1].inputs[0].parameters = {"param": "value"}
TestNodeSetWithParams.nodes.append(
    AgentGraphNode(
        name="one_analysis_w_params",
        description="Perform one analysis with parameters",
        function=one_analysis_w_params,
        inputs=[AgentData(name="dataset", parameters={}, description="Input dataset", data=None)],
        outputs=[AgentData(name="one_data", parameters={"param": "value"}, description="One analysis result with parameters", data=None)],
    )
)
