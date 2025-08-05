from typing import Callable, Optional, List, Union
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData


class AgentRegistry:
    """
    A registry that maps function names to actual callable functions.
    This allows the TaiatGraphArchitect to resolve function names to actual functions.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._functions: dict[str, Callable] = {}

    def register(self, name: str, function: Callable) -> None:
        """
        Register a function with a name.

        Args:
            name: The name to register the function under
            function: The callable function to register
        """
        self._functions[name] = function

    def get(self, name: str) -> Optional[Callable]:
        """
        Get a function by name.

        Args:
            name: The name of the function to retrieve

        Returns:
            The registered function, or None if not found
        """
        return self._functions.get(name)

    def list_registered(self) -> List[str]:
        """
        Get all registered function names.

        Returns:
            List of registered function names
        """
        return list(self._functions.keys())

    def clear(self) -> None:
        """Clear all registered functions."""
        self._functions.clear()


class TaiatGraphArchitect:
    """
    An architect that builds AgentGraphNodeSet from descriptions using an LLM
    and resolves function names to actual functions using an AgentRegistry.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        agent_registry: AgentRegistry,
        verbose: bool = False,
        llm_explanation: bool = False,
    ):
        """
        Initialize the TaiatGraphArchitect.

        Args:
            llm: The language model to use for generating node specifications
            agent_registry: Registry containing available agent functions
            verbose: Whether to print verbose output
            llm_explanation: Whether to include LLM explanations in output
        """
        self.llm = llm
        self.agent_registry = agent_registry
        self.verbose = verbose
        self.llm_explanation = llm_explanation

    def build(self, description: str) -> AgentGraphNodeSet:
        """
        Build an AgentGraphNodeSet from a description.

        Args:
            description: Textual description of the desired agent graph

        Returns:
            AgentGraphNodeSet with resolved functions

        Raises:
            ValueError: If function names cannot be resolved
        """
        if self.verbose:
            print(f"Building agent graph from description: {description[:100]}...")

        # Generate node specifications using LLM
        nodes = self._generate_nodes_from_description(description)

        # Resolve function names to actual functions
        resolved_nodes = self._resolve_function_names(nodes)

        if self.verbose:
            print(f"Generated {len(resolved_nodes)} nodes with resolved functions")

        return AgentGraphNodeSet(nodes=resolved_nodes)

    def _generate_nodes_from_description(
        self, description: str
    ) -> List[AgentGraphNode]:
        """
        Generate AgentGraphNode specifications from a description using the LLM.

        Args:
            description: Textual description of the desired agent graph

        Returns:
            List of AgentGraphNode objects with function names as strings
        """
        # Create a prompt for the LLM to generate node specifications
        prompt = self._create_generation_prompt(description)

        response = self.llm.invoke(prompt)

        if self.verbose:
            print(f"LLM Response: {response.content}")

        # Parse the response to extract node specifications
        # For now, we'll create a simple mock implementation
        # In a real implementation, this would parse the LLM response
        return self._parse_llm_response(response.content, description)

    def _create_generation_prompt(self, description: str) -> str:
        """
        Create a prompt for the LLM to generate node specifications.

        Args:
            description: The user's description

        Returns:
            Formatted prompt for the LLM
        """
        available_functions = self.agent_registry.list_registered()

        prompt = f"""
You are tasked with creating an agent graph from a description. 

Available functions in the registry:
{", ".join(available_functions) if available_functions else "No functions registered"}

User description: {description}

Please generate AgentGraphNode specifications that use the available functions.
Each node should have:
- name: descriptive name
- description: what the node does
- function: name of a function from the registry
- inputs: list of AgentData objects
- outputs: list of AgentData objects

Return the specifications in a structured format that can be parsed.
"""
        return prompt

    def _parse_llm_response(
        self, response: str, description: str
    ) -> List[AgentGraphNode]:
        """
        Parse the LLM response to extract node specifications.

        Args:
            response: The LLM response
            description: Original description for context

        Returns:
            List of AgentGraphNode objects
        """
        # This is a simplified implementation
        # In a real implementation, this would parse structured output from the LLM

        # For now, create some example nodes based on the description
        # This is a placeholder implementation
        nodes = []

        # Simple heuristic: if description mentions "process", "analyze", "report"
        # create corresponding nodes
        if "process" in description.lower():
            nodes.append(
                AgentGraphNode(
                    name="data_processor",
                    description="Processes raw data",
                    function="data_processor",
                    inputs=[AgentData(name="raw_data", description="Raw input data")],
                    outputs=[
                        AgentData(name="processed_data", description="Processed data")
                    ],
                )
            )

        if "analyze" in description.lower():
            nodes.append(
                AgentGraphNode(
                    name="data_analyzer",
                    description="Analyzes processed data",
                    function="analyzer",
                    inputs=[
                        AgentData(name="processed_data", description="Processed data")
                    ],
                    outputs=[
                        AgentData(
                            name="analysis_result", description="Analysis results"
                        )
                    ],
                )
            )

        if "report" in description.lower():
            nodes.append(
                AgentGraphNode(
                    name="report_generator",
                    description="Generates final report",
                    function="report_generator",
                    inputs=[
                        AgentData(
                            name="analysis_result", description="Analysis results"
                        )
                    ],
                    outputs=[
                        AgentData(name="final_report", description="Final report")
                    ],
                )
            )

        return nodes

    def _resolve_function_names(
        self, nodes: List[AgentGraphNode]
    ) -> List[AgentGraphNode]:
        """
        Resolve function names to actual functions using the registry.

        Args:
            nodes: List of nodes with function names as strings

        Returns:
            List of nodes with resolved functions

        Raises:
            ValueError: If function names cannot be resolved
        """
        resolved_nodes = []
        missing_functions = []

        for node in nodes:
            if isinstance(node.function, str):
                # Try to resolve the function name
                resolved_function = self.agent_registry.get(node.function)
                if resolved_function is not None:
                    # Create a new node with the resolved function
                    resolved_node = AgentGraphNode(
                        name=node.name,
                        description=node.description,
                        function=resolved_function,
                        inputs=node.inputs,
                        outputs=node.outputs,
                    )
                    resolved_nodes.append(resolved_node)
                else:
                    missing_functions.append(node.function)
            else:
                # Function is already resolved
                resolved_nodes.append(node)

        if missing_functions:
            available_functions = self.agent_registry.list_registered()
            error_msg = f"Function(s) {', '.join(missing_functions)} not found in agent registry. "
            if available_functions:
                error_msg += f"Available functions: {available_functions}"
            else:
                error_msg += "No functions registered."
            raise ValueError(error_msg)

        return resolved_nodes
