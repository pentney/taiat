import json
from typing import Annotated
import itertools

import kaggle
from langchain_core.language_models.chat_models import BaseChatModel

from taiat.base import AgentGraphNodeSet

ml_output_matcher_prompt = """
You are a data scientist.
You are given a request, a list of datasets, and a list of outputs.
The request is a description of a machine learning task.
The datasets are a list of datasets that can be used to solve the request.
The outputs are a list of outputs that can be used to solve the request.
You need to select the most relevant outputs and a relevant dataset for the request.

Return the output in the form of a JSON object with the following fields:
- dataset: the name of the dataset
- outputs: a list of the names of the outputs
"""

class MLOutputMatcher:
    def __init__(self, request: str, llm: BaseChatModel):
        self.request = request
        self.llm = llm

    def load_dataset_list(self) -> list[str]:
        self.datasets = kaggle.api.dataset_list(
            search="uci",
            file_type="csv",
        )
        self.datasets = [dataset.ref for dataset in self.datasets]

    def get_outputs(self) -> list[str]:
        return self.outputs

    def get_dataset(self) -> str:
        return self.dataset

    def load_output_list(
        self,
        agents: AgentGraphNodeSet,
    ) -> list[str]:
        self.outputs = list(itertools.chain.from_iterable(
            [agent.outputs for agent in agents.nodes]))
        print("outputs", self.outputs)

    def select_task(self, query: str) -> list[str]:
        response = self.llm.invoke(
            [
                {
                    "role": "system",
                    "content": ml_output_matcher_prompt,
                },
                {
                    "role": "user",
                    "content": f"Request: {self.request}\nDatasets: {self.datasets}\nOutputs: {self.outputs}",
                },
            ]
        )
        print("response", response.content)
        result = json.loads(response.content)
        self.dataset, self.outputs = result["dataset"], result["outputs"]
