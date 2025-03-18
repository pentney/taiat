
from copy import copy
from typing import Callable, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph

from taiat.base import State, FrozenAgentData, AgentGraphNodeSet
from taiat.builder import TaiatBuilder


class TaiatEngine:
    def __init__(
            self,
            llm_dict: dict[str, BaseChatModel],
            builder: TaiatBuilder,
            node_set: AgentGraphNodeSet,
            output_matcher: Callable[[list[str]], list[str]]):
        self.llms = llm_dict
        self.node_set = node_set
        self.builder = builder
        self.output_matcher = output_matcher

    def run(
          self,
          state: State,
          ) -> State:
        query = state["query"]
        goal_outputs = self.output_matcher(query.query)
        if goal_outputs is None:
            query.status = "error"
            query.error = "No goal output"
            return state
        # Convert goal_outputs to AgentData if they are strings
        for i, goal_output in enumerate(goal_outputs):
            if type(goal_output) == str:
                goal_outputs[i] = FrozenAgentData(name=goal_output, parameters={})
        query.inferred_goal_output = goal_outputs
        graph, query.status, query.error = self.builder.get_plan(goal_outputs)
        if query.status == "error":
            return state
        state = graph.invoke(state)
        query.status = "success"
        return state

