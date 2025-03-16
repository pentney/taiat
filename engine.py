from typing import Callable

from langchain_core.chat_models import BaseChatModel

from taiat.taiat.taiat_builder import AgentGraph
from taiat.taiat.query import TaiatQuery

class TaiatEngine:
    def __init__(
            self,
            llm_dict: dict[str, BaseChatModel],
            graph: AgentGraph,
            output_matcher: Callable[[str], str]):
        self.llms = llm_dict
        self.graph = graph
        self.output_matcher = output_matcher

def run(self, query: TaiatQuery) -> TaiatQuery:
        goal_output = self.output_matcher(query.query)
        query.inferred_goal_output = goal_output
        if goal_output:
            if goal_output not in self.graph.data_dependence:
                query.status = "error"
                query.error = f"Goal output {goal_output} unknown"
                return query
            else:
                output_set = set()
                output_set.add(goal_output)
                while len(output_set) > 0:
                    current_output = output_set.pop()
                    if current_output not in self.graph.data_dependence:
                         query.status = "error"
                         query.error = f"Graph error: bad intermediate {current_output} found"
                         return query
                    needed_outputs = self.graph.data_dependence[current_output]
                    if needed_outputs:
                         for needed_output in needed_outputs:
                              if needed_output not in self.graph.data_depen:
                                   query.status = "error"
                                   query.error = f"Intermediate data {needed_output} unknown"
                                   return query
                              if needed_output in needed_outputs:
                                   query.status = "error"
                                   query.error = f"Graph error: circular dependency found: {needed_output}"
                                   return query
                              needed_outputs.append(needed_output)
            path = self.graph.get_plan(needed_outputs)
            query = self.graph.execute_plan(path, query)
        else:
            query.status = "error"
            query.error = "No goal output"
        return query

