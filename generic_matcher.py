from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import OutputMatcher, AgentData

UNKNOWN_OUTPUT = "<taiat_unknown_output>"

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

    generic_output_matcher_prompt = """
You are given a list of possible desired outputs.
The request is a description of a task to perform.
The outputs are a list of outputs that can be used to solve the request.
The outputs have a name.
Some of the outputs may have a description.
Some of the outputs may have <output_parameters> that can be used to solve the request.
The <output_parameters> are given in the format <parameter_name1>:<parameter_value1> <parameter_name2>:<parameter_value2> ...
The outputs are given in the following format:
<output_name> | <output_parameters> | <output_description>
If the <output_parameters> are relevant to this request, you need to give the <output_parameters> as well.
Do NOT give the <output_parameters> if they are not relevant to the request.
You need to select the most relevant outputs and a relevant dataset for the request.

Return the outputs that are most relevant to the request, each on a new line.
Give the <output_name> and <output_parameters> in the following format:
<output_name> <parameter_name1>:<parameter_value1> <parameter_name2>:<parameter_value2> ...
Do not return the <output_description>.
Return nothing else.
If you cannot find any relevant parameters, return the special token {unknown_output}.

Outputs:
{outputs}

Request:
{request}

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
        prompt = self.generic_output_matcher_prompt.format(
            request=query,
            outputs="\n".join(self.output_names),
            unknown_output=UNKNOWN_OUTPUT
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
                output_list.append(AgentData(name=output_name, parameters=output_parameters))
        return output_list