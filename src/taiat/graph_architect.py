import json
import re
from pathlib import Path
from typing import Any, Callable, List, Optional

from jinja2 import Environment, FileSystemLoader
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet


def _extract_json(text: str) -> Any:
    """Parse JSON from a response that may be wrapped in a markdown code fence."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


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
        return self._parse_llm_response(response.content)

    def _create_generation_prompt(self, description: str) -> str:
        """
        Create a prompt for the LLM to generate node specifications.

        Args:
            description: The user's description

        Returns:
            Formatted prompt for the LLM
        """
        template = _JINJA_ENV.get_template("generate_nodes.j2")
        return template.render(
            description=description,
            available_functions=self.agent_registry.list_registered(),
        )

    def _parse_llm_response(self, response: str) -> List[AgentGraphNode]:
        """
        Parse the JSON array of node specs returned by the LLM.

        Args:
            response: The LLM response (JSON array, optionally in a markdown fence)

        Returns:
            List of AgentGraphNode objects with function names as strings
        """
        try:
            items = _extract_json(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {exc}\n\n{response}") from exc

        nodes = []
        for item in items:
            inputs = [AgentData(**i) for i in item.get("inputs", [])]
            outputs = [AgentData(**o) for o in item.get("outputs", [])]
            nodes.append(
                AgentGraphNode(
                    name=item["name"],
                    description=item.get("description"),
                    function=item.get("function"),
                    inputs=inputs,
                    outputs=outputs,
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
