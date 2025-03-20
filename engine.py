
from copy import copy
from typing import Callable, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph

from taiat.base import State, FrozenAgentData, AgentGraphNodeSet, TaiatQuery, AgentData, OutputMatcher
from taiat.builder import TaiatBuilder


class TaiatEngine:
    def __init__(
            self,
            llm_dict: dict[str, BaseChatModel],
            builder: TaiatBuilder,
            output_matcher: OutputMatcher):
        self.llms = llm_dict
        self.builder = builder
        self.output_matcher = output_matcher

    def get_plan(
            self,
            goal_outputs: list[AgentData], 
            query: TaiatQuery,
            verbose: bool = False
    ) -> tuple[StateGraph, str]:
        graph, query.status, query.error = self.builder.get_plan(query, goal_outputs)
        if verbose:
            spacing = "\n   "
            print(f"Outputs: {spacing.join(query.all_outputs)}")
            print(f"Workflow graph: {graph.edges.split(spacing)}")
        return graph, query.status

    def run(
          self,
          state: State,
          ) -> State:
        query = state["query"]
        self.goal_outputs = self.output_matcher.get_outputs(query.query)
        if self.goal_outputs is None:
            query.status = "error"
            query.error = "No goal output"
            return state
        # Convert goal_outputs to AgentData if they are strings
        for i, goal_output in enumerate(self.goal_outputs):
            if type(goal_output) == str:
                self.goal_outputs[i] = FrozenAgentData(name=goal_output, parameters={})
        query.inferred_goal_output = self.goal_outputs
        graph, status = self.get_plan(self.goal_outputs, query)
        if status == "error":
            return state
        state = graph.invoke(state)
        query.status = "success"
        return state

