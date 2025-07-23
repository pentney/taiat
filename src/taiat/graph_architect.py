from typing import Callable, Dict, Optional
from pydantic import BaseModel

try:
    from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet
except ImportError:
    from base import AgentData, AgentGraphNode, AgentGraphNodeSet
from langchain_core.language_models.chat_models import BaseChatModel


class AgentRegistry:
    """
    A registry that maps function names to actual callable functions.
    Users can register their agent functions with this registry.
    """

    def __init__(self):
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str, function: Callable) -> None:
        """
        Register a function with a given name.

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
            The function if found, None otherwise
        """
        return self._functions.get(name)

    def list_registered(self) -> list[str]:
        """
        Get a list of all registered function names.

        Returns:
            List of registered function names
        """
        return list(self._functions.keys())

    def clear(self) -> None:
        """Clear all registered functions."""
        self._functions.clear()


DEFAULT_TAIAT_GRAPH_ARCHITECT_SYSTEM_PROMPT = """
You are a helpful assistant that builds a list of AgentGraphNodes from a textual description.

An AgentData is a data object that is used to pass data between agents.
It has the following fields:
- name: The name of the data object (a string).
- parameters: A dictionary of parameters that the data object requires (all strings).
- description: A description of the data object (a string).

The parameters of an AgentData object are used to match the AgentData object to the AgentGraphNode that requires it.
The parameters are a dictionary of strings.
If no parameters are necessary, the parameters should be an empty dictionary.
Parameters should be provided when there are multiple AgentData objects in a similar category.

When specifying the inputs and outputs of an AgentGraphNode, you should use the name of the AgentData object, not the description.
The inputs should be AgentData objects.
If parameters are necessary, the AgentData object should have the parameters specified.
Use as few parameters as possible, but use them when necessary to identify specific subtypes of inputs to consume.

An AgentGraphNode is a node in the graph that represents an agent.
The AgentGraphNode has the following fields:
- name: The name of the agent (a string).
- description: A description of the agent (a string).
- inputs: A list of AgentData objects that the agent requires.
- outputs: A list of AgentData objects that the agent produces.
- function: The name of the function that the agent will execute (a string). This should be a function name that can be looked up in an agent registry.

You should provide a GraphArchitectResponse object with the following fields:
- agent_data: A list of AgentData objects.
- agent_graph_nodes: A list of AgentGraphNodes.
- description: A description of the AgentGraphNodeSet.
{explanation}
- error: An error message, if there is an error. If there is no error, this should be null.

When building the AgentGraphNodeSet, you should use the following rules:
- Identify all relevant inputs and outputs mentioned in the description.
- For each input and output, draft an AgentData object.
- Identify every process that consumes an input and/or produces an output.
- For each process, draft an AgentGraphNode. Use the AgentData objects you created earlier to populate the inputs and outputs of the AgentGraphNode.
- For the function field, provide a descriptive function name as a string that represents what the agent should do.
- Once you have drafted all the AgentGraphNodes, you should provide the GraphArchitectResponse object.
"""


class GraphArchitectResponse(BaseModel):
    """
    A response from the graph architect.
    """

    agent_data: list[AgentData]
    agent_graph_nodes: list[AgentGraphNode]
    description: str
    explanation: Optional[str] = None
    error: Optional[str] = None


class TaiatGraphArchitect:
    """
    A class that, with an LLM, will build an AgentGraphNodeSet from a textual description.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        agent_registry: AgentRegistry,
        verbose: bool = False,
        llm_explanation: bool = False,
    ):
        self.llm = llm
        self.agent_registry = agent_registry
        self.verbose = verbose
        self.llm_explanation = llm_explanation

    def _llm_invoke(self, system_prompt: str, user_message: str) -> str:
        """
        Invoke the LLM with a system prompt and user message.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _resolve_function_names(
        self, agent_graph_nodes: list[AgentGraphNode]
    ) -> list[AgentGraphNode]:
        """
        Resolve function names to actual functions using the agent registry.

        Args:
            agent_graph_nodes: List of AgentGraphNodes with function names as strings

        Returns:
            List of AgentGraphNodes with actual functions resolved

        Raises:
            ValueError: If a function name cannot be resolved
        """
        resolved_nodes = []

        for node in agent_graph_nodes:
            # Create a copy of the node to avoid modifying the original
            resolved_node = AgentGraphNode(
                name=node.name,
                description=node.description,
                inputs=node.inputs,
                outputs=node.outputs,
                function=None,  # Will be set below
            )

            # Get the function name from the original node
            if hasattr(node, "function") and node.function is not None:
                if isinstance(node.function, str):
                    function_name = node.function
                    actual_function = self.agent_registry.get(function_name)
                    if actual_function is None:
                        raise ValueError(
                            f"Function '{function_name}' not found in agent registry. Available functions: {self.agent_registry.list_registered()}"
                        )
                    resolved_node.function = actual_function
                else:
                    # If it's already a callable, use it directly
                    resolved_node.function = node.function
            else:
                raise ValueError(
                    f"AgentGraphNode '{node.name}' has no function specified"
                )

            resolved_nodes.append(resolved_node)

        return resolved_nodes

    def build(self, description: str) -> AgentGraphNodeSet:
        """
        Build an AgentGraphNodeSet from a textual description.

        Args:
            description: Textual description of the desired agent graph

        Returns:
            AgentGraphNodeSet with resolved functions

        Raises:
            ValueError: If there's an error in the LLM response or function resolution
        """
        # Prepare the explanation field for the system prompt
        explanation_field = (
            "- explanation: An explanation of the AgentGraphNodeSet.\n"
            if self.llm_explanation
            else ""
        )

        system_prompt = DEFAULT_TAIAT_GRAPH_ARCHITECT_SYSTEM_PROMPT.format(
            explanation=explanation_field
        )

        user_message = f"""
        Description:
        {description}
        
        Please provide your response as a valid JSON object with the following structure:
        {{
            "agent_data": [...],
            "agent_graph_nodes": [...],
            "description": "...",
            "explanation": "...",
            "error": null
        }}
        """

        try:
            response_content = self._llm_invoke(system_prompt, user_message)

            if self.verbose:
                print(f"LLM Response: {response_content}")

            # Parse the response
            graph_architect_response = GraphArchitectResponse.model_validate_json(
                response_content
            )

            if graph_architect_response.error:
                raise ValueError(
                    f"Graph architect error: {graph_architect_response.error}"
                )

            # Resolve function names to actual functions
            resolved_nodes = self._resolve_function_names(
                graph_architect_response.agent_graph_nodes
            )

            # Create the final AgentGraphNodeSet
            agent_graph_node_set = AgentGraphNodeSet(nodes=resolved_nodes)

            return agent_graph_node_set

        except Exception as e:
            if self.verbose:
                print(f"Error in build: {e}")
            raise ValueError(f"Failed to build AgentGraphNodeSet: {str(e)}")
