from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentData, OutputMatcher

UNKNOWN_OUTPUT = "<taiat_unknown_output>"

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
        Initialize the matcher with a list of output names.
        """
        self.llm = llm
        self.output_names = []
        for output in outputs:
            if isinstance(output, dict):
                output = AgentData(**output)
            output_desc = f"{output.name} ..."
            if output.parameters:
                for k, v in output.parameters.items():
                    output_desc += f" {k}:{v}"
            output_desc += f" ... {output.description}"
            self.output_names.append(output_desc)

    def get_outputs(self, query: str) -> list[str]:
        """
        Get the outputs that match the request.
        """
        template = _JINJA_ENV.get_template("generic_output_matcher.j2")
        prompt = template.render(
            request=query,
            outputs="\n".join(self.output_names),
            unknown_output=UNKNOWN_OUTPUT,
        )
        response = self.llm.invoke(
            prompt,
        )
        return self.process_outputs(response.content)

    def process_outputs(self, outputs: str) -> list[str]:
        """
        Process the outputs from the generic matcher.
        """
        lines = [line.strip() for line in outputs.split("\n") if line.strip()]
        output_list = []
        for line in lines:
            if line == UNKNOWN_OUTPUT:
                raise UnknownOutputException("Unknown output returned from matcher")
            else:
                fields = line.split(" ")
                output_name = fields[0]
                output_parameters = {}
                if len(fields) > 1:
                    for field in fields[1:]:
                        fields = field.split(":")
                        output_parameters[fields[0]] = fields[1]
                output_list.append(
                    AgentData(name=output_name, parameters=output_parameters)
                )
        return output_list
