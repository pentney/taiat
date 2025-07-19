from copy import copy
from typing import Callable, Any, Optional, Tuple
import functools

from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import (
    State,
    FrozenAgentData,
    AgentGraphNodeSet,
    TaiatQuery,
    AgentData,
    OutputMatcher,
    TAIAT_TERMINAL_NODE,
)
from taiat.builder import TaiatBuilder
from taiat.generic_matcher import GenericMatcher
from taiat.executor import TaiatExecutor


class TaiatEngine:
    def __init__(
        self,
        llm: BaseChatModel,
        builder: TaiatBuilder,
        output_matcher: OutputMatcher | None,
    ):
        self.llm = llm
        self.builder = builder
        if output_matcher is None:
            if self.builder.node_set is None or not self.builder.node_set.nodes:
                raise ValueError(
                    "Node set is not created or empty. Run builder.build() with a node set first."
                )
            outputs = []
            for node in self.builder.node_set.nodes:
                outputs.extend(node.outputs)
            self.output_matcher = GenericMatcher(llm, outputs)
        else:
            self.output_matcher = output_matcher

    def get_plan(
        self, goal_outputs: list[AgentData], query: TaiatQuery, verbose: bool = False
    ) -> tuple[None, str]:
        """
        Get execution plan using the builder.

        Note: This method now returns None for the graph parameter since we're not using langgraph,
        but maintains the same interface for backward compatibility.
        """
        _, query.status, query.error = self.builder.get_plan(query, goal_outputs)
        if verbose:
            spacing = "\n   "
            print(f"Outputs: {spacing.join(query.all_outputs)}")
        return None, query.status

    def run(
        self,
        state: State,
    ) -> State | Tuple[State, Optional[str]]:
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

        # Get plan and check for errors
        _, status = self.get_plan(self.goal_outputs, query)
        if status == "error":
            return state

        # Create executor and run the execution
        if self.builder.node_set is None:
            query.status = "error"
            query.error = "No node set available"
            return state

        executor = TaiatExecutor(self.builder.node_set, verbose=query.visualize_graph)

        try:
            state = executor.execute(state, self.goal_outputs)
            query.status = "success"
            # Set the execution path
            query.path = executor.get_execution_path()
        except Exception as e:
            query.status = "error"
            query.error = str(e)
            return state

        # Create visualization if requested (after path is set)
        visualization = None
        if query.visualize_graph:
            visualization = self.builder.create_graph_visualization(
                query, self.goal_outputs
            )
        # Return visualization along with state if requested
        if query.visualize_graph:
            return state, visualization
        return state
