
from copy import copy
from typing import Callable, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph

from taiat.base import State, FrozenAgentData, AgentGraphNodeSet, TaiatQuery, AgentData, OutputMatcher
from taiat.builder import TaiatBuilder
from taiat.generic_matcher import GenericMatcher

class TaiatEngine:
    def __init__(
            self,
            llm: BaseChatModel,
            builder: TaiatBuilder,
            output_matcher: OutputMatcher | None):
        self.llm = llm
        self.builder = builder
        if output_matcher is None:
            if self.builder.node_set is None or not self.builder.node_set.nodes:
                raise ValueError("Node set is not created or empty. Run builder.build() with a node set first.")
            outputs = []
            for node in self.builder.node_set.nodes:
                outputs.extend(node.outputs)
            self.output_matcher = GenericMatcher(llm, outputs)
        else:
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

