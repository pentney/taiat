
from copy import copy
from typing import Callable, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph

from taiat.builder import AgentData, TaiatQuery, TaiatBuilder, State


class TaiatEngine:
    def __init__(
            self,
            llm_dict: dict[str, BaseChatModel],
            graph: StateGraph,
            builder: TaiatBuilder,
            output_matcher: Callable[[list[str]], list[str]]):
        self.llms = llm_dict
        self.graph = graph
        self.builder = builder
        self.output_matcher = output_matcher

    def run(
          self,
          query: TaiatQuery,
          data: dict[str, Any],
          ) -> State:
        state = State({"query": query, "data": data})
        goal_outputs = self.output_matcher(query.query)
        if goal_outputs is None:
            query.status = "error"
            query.error = "No goal output"
            return state
        query.inferred_goal_output = goal_outputs
        for goal_output in query.inferred_goal_output:
            if goal_output not in self.builder.data_source:
                print("goal_output not in data_source", goal_output)
                query.status = "error"
                query.error = f"Goal output {goal_output} unknown"
                return state
            else:
                output_set = set()
                output_set.add(goal_output)
                while len(output_set) > 0:
                    current_output = output_set.pop()
                    if current_output not in self.builder.data_dependence:
                         query.status = "error"
                         query.error = f"Graph error: bad intermediate {current_output} found"
                         return state
                    needed_outputs = copy(self.builder.data_dependence[current_output])
                    if needed_outputs:
                         while len(needed_outputs) > 0:
                              needed_output = needed_outputs.pop()
                              if needed_output.name not in self.builder.data_dependence:
                                   query.status = "error"
                                   query.error = f"Intermediate data {needed_output.name} unknown"
                                   return state
                              if needed_output.name in needed_outputs:
                                   query.status = "error"
                                   query.error = f"Graph error: circular dependency found: {needed_output}"
                                   return state
                              output_set.add(needed_output.name)
            path = self.builder.get_plan(output_set)
            query.path = path
            state = self.graph.invoke(state)
            query.status = "success"
        return state

