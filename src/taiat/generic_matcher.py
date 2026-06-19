import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentData, OutputMatcher


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


class UnknownOutputException(Exception):
    """
    Exception raised when an unknown output is returned from the matcher.
    """

    pass


class GenericMatcher(OutputMatcher):
    """
    A generic matcher that can be used to match any output, if perhaps suboptimally.
    Specialized output matchers should be used for more precise matching.
    """

    def __init__(self, llm: BaseChatModel, outputs: list[AgentData | dict]):
        """
        Initialize the matcher with a list of outputs.
        """
        self.llm = llm
        self._outputs: list[AgentData] = []
        for output in outputs:
            if isinstance(output, dict):
                output = AgentData(**output)
            self._outputs.append(output)

    def get_outputs(self, query: str) -> list[AgentData]:
        """
        Get the outputs that match the request.
        """
        outputs_json = json.dumps(
            [
                {"name": o.name, "description": o.description, "parameters": o.parameters}
                for o in self._outputs
            ],
            indent=2,
        )
        template = _JINJA_ENV.get_template("generic_output_matcher.j2")
        prompt = template.render(request=query, outputs_json=outputs_json)
        response = self.llm.invoke(prompt)
        return self.process_outputs(response.content)

    def process_outputs(self, response: str) -> list[AgentData]:
        """
        Parse the JSON array of selected outputs returned by the LLM.
        """
        try:
            items = _extract_json(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {exc}\n\n{response}") from exc

        if not items:
            raise UnknownOutputException("No relevant outputs found for request")

        return [
            AgentData(name=item["name"], parameters=item.get("parameters", {}))
            for item in items
        ]
