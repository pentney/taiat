from copy import copy
from typing import Callable, Any, Optional, Tuple
import functools

from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import (
    State,
    FrozenAgentData,
    TaiatQuery,
    AgentData,
    AgentGraphNodeSet,
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

    @classmethod
    def from_node_set(
        cls,
        node_set: AgentGraphNodeSet,
        llm: BaseChatModel,
        initial_inputs: list[AgentData] | None = None,
        output_matcher: OutputMatcher | None = None,
        verbose: bool = False,
    ) -> "TaiatEngine":
        """
        Build an engine from an existing AgentGraphNodeSet.

        Initial inputs (data available before any node runs) are inferred as any
        AgentData name that appears in a node's inputs but is not produced as an
        output by any node. Pass ``initial_inputs`` explicitly to override.
        """
        if initial_inputs is None:
            produced = {out.name for node in node_set.nodes for out in node.outputs}
            seen: set[str] = set()
            initial_inputs = []
            for node in node_set.nodes:
                for inp in node.inputs:
                    if inp.name not in produced and inp.name not in seen:
                        seen.add(inp.name)
                        initial_inputs.append(inp)

        builder = TaiatBuilder(llm=llm, verbose=verbose)
        builder.build(node_set=node_set, inputs=initial_inputs)
        return cls(llm=llm, builder=builder, output_matcher=output_matcher)

    @classmethod
    def from_description(
        cls,
        description: str,
        llm: BaseChatModel,
        agent_registry: "AgentRegistry",
        output_matcher: OutputMatcher | None = None,
        verbose: bool = False,
    ) -> "TaiatEngine":
        """
        Build an engine from a plain-English workflow description.

        Calls TaiatGraphArchitect to generate the node set, then delegates to
        ``from_node_set`` to wire up the builder and infer initial inputs.
        """
        from taiat.graph_architect import AgentRegistry, TaiatGraphArchitect

        architect = TaiatGraphArchitect(
            llm=llm, agent_registry=agent_registry, verbose=verbose
        )
        node_set = architect.build(description)
        return cls.from_node_set(
            node_set, llm, output_matcher=output_matcher, verbose=verbose
        )

    def get_plan(
        self, goal_outputs: list[AgentData], query: TaiatQuery, verbose: bool = False
    ) -> tuple[None, str]:
        # Return None instead of StateGraph to maintain interface compatibility
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

        # Create executor
        executor = TaiatExecutor(verbose=self.builder.verbose)

        # Get plan and extract nodes
        _, status = self.get_plan(self.goal_outputs, query)
        if status == "error":
            return state

        # Get planned execution path from builder
        if (
            not hasattr(self.builder, "planned_execution_path")
            or not self.builder.planned_execution_path
        ):
            query.status = "error"
            query.error = "No execution path available"
            return state

        # Get nodes in planned execution order
        planned_nodes = []
        for node_name in self.builder.planned_execution_path:
            node = next(
                (n for n in self.builder.node_set.nodes if n.name == node_name), None
            )
            if node:
                planned_nodes.append(node)

        if not planned_nodes:
            query.status = "error"
            query.error = "No valid nodes found in execution path"
            return state

        # Execute using custom executor
        state = executor.execute(planned_nodes, state, self.goal_outputs)

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
