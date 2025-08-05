from typing import Callable, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentGraphNode, AgentData


class NodeGenerator:
    """
    A generator that creates AgentGraphNode from plain English text descriptions
    and supplied functions using LLM calls.
    """

    node_generator_prompt = """
You are given a plain English description of a task and a function name.
Your job is to generate an AgentGraphNode that represents this task.

The AgentGraphNode should have:
1. A name (use the function name or a descriptive name)
2. A description (based on the plain English description)
3. Inputs (what data/parameters the function needs)
4. Outputs (what data the function produces)

For inputs and outputs, you should create AgentData objects with:
- name: a descriptive name for the data
- description: what this data represents
- parameters: any relevant parameters (optional)

The function will be provided separately and attached to the node.

Plain English Description: {description}
Function Name: {function_name}

Please return the node specification in the following JSON format:
{{
    "name": "descriptive_name",
    "description": "description of what this node does",
    "inputs": [
        {{
            "name": "input_name",
            "description": "description of this input",
            "parameters": {{}}
        }}
    ],
    "outputs": [
        {{
            "name": "output_name", 
            "description": "description of this output",
            "parameters": {{}}
        }}
    ]
}}

Return only the JSON, no other text.
"""

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the NodeGenerator with an LLM.
        """
        self.llm = llm

    def generate_node(
        self, description: str, function: Callable, function_name: Optional[str] = None
    ) -> AgentGraphNode:
        """
        Generate an AgentGraphNode from a plain English description and function.

        Args:
            description: Plain English description of what the node should do
            function: The function to be attached to the node
            function_name: Optional name for the function (defaults to function.__name__)

        Returns:
            AgentGraphNode with the generated specification and attached function
        """
        if function_name is None:
            function_name = function.__name__

        prompt = self.node_generator_prompt.format(
            description=description, function_name=function_name
        )

        response = self.llm.invoke(prompt)

        # Parse the JSON response
        import json

        try:
            node_spec = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Convert the specification to AgentData objects
        inputs = [AgentData(**input_spec) for input_spec in node_spec.get("inputs", [])]
        outputs = [
            AgentData(**output_spec) for output_spec in node_spec.get("outputs", [])
        ]

        return AgentGraphNode(
            name=node_spec["name"],
            description=node_spec.get("description"),
            function=function,
            inputs=inputs,
            outputs=outputs,
        )
