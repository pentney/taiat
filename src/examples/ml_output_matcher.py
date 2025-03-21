import json
from typing import Annotated
import itertools

import kaggle
from langchain_core.language_models.chat_models import BaseChatModel
from taiat.base import AgentGraphNodeSet, AgentData
from taiat.engine import OutputMatcher
ml_output_matcher_prompt = """
You are a data scientist.
You are given a request, a list of datasets, and a list of outputs.
The request is a description of a machine learning task.
The datasets are a list of datasets that can be used to solve the request.
The outputs are a list of outputs that can be used to solve the request.
Some of the outputs have parameters that can be used to solve the request.
If the parameters are relevant to the request, you need to select the parameters as well.
You need to select the most relevant outputs and a relevant dataset for the request.

Return the output in the form of a JSON object with the following fields:
- dataset: the name of the dataset
- outputs: a list of the names of the outputs
"""

class MLOutputMatcher(OutputMatcher):
    def __init__(self, request: str, llm: BaseChatModel):
        self.request = request
        self.llm = llm

    def load_dataset_list(self) -> list[str]:
        self.datasets = kaggle.api.dataset_list(
            search="uci",
            file_type="csv",
        )
        self.datasets = [dataset.ref for dataset in self.datasets]

    def get_outputs(self, query: str) -> list[str]:
        # query unused here, assumed we've already collected the outputs
        return self.outputs

    def get_dataset(self) -> str:
        return self.dataset
    
    def get_model_name(self) -> str:
        for output in self.outputs:
            if output.name == "model" and "model_name" in output.parameters:
                return output.parameters["model_name"]
        return None

    def load_output_list(
        self,
        agents: AgentGraphNodeSet,
    ) -> list[str]:
        self.outputs = list(itertools.chain.from_iterable(
            [[(output.name, output.parameters) for output in agent.outputs] for agent in agents.nodes]))

    def get_inputs_and_outputs(self, query: str) -> list[str]:
        output_str = ""
        print("outputs", self.outputs)
        for output_name, output_parameters in self.outputs:
            if output_parameters:
                params = ",".join([f"{k}:{v}" for k, v in output_parameters.items()])
                output_str += f"{output_name}, {params}\n"
            else:
                output_str += f"{output_name}\n"

        request =[
                {
                    "role": "system",
                    "content": ml_output_matcher_prompt,
                },
                {
                    "role": "user",
                    "content": f"Request: {self.request}\nDatasets: {self.datasets}\nOutputs: {output_str}",
                },
            ]
        print("request", request)
        response = self.llm.invoke(request)
        if response and response.content:
            print("response", response.content)
            content = "\n".join(response.content.split("\n")[1:-1])
            result = json.loads(content)
            self.dataset = result["dataset"]
            self.outputs = self.process_outputs(result["outputs"])
        else:
            raise ValueError(f"Invalid response from the LLM: {response}")

    def process_outputs(self, outputs: list[str]) -> list[str]:
        processed_outputs = []
        for output in outputs:
            if "," in output:
                name, params = output.split(",")
                params = {k: v for k, v in [param.strip().split(":") for param in params.split(",")]}
                processed_outputs.append(AgentData(name=name, parameters=params))
            else:
                processed_outputs.append(AgentData(name=output))
        print("processed_outputs", processed_outputs)
        return processed_outputs