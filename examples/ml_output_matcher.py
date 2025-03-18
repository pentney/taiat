
ml_output_matcher_prompt = """
You are a data scientist.
You are given a request and a list of outputs.
You need to select the most relevant outputs for the request.
"""

class MLOutputMatcher:
    def __init__(self, request: str, llm: BaseChatModel):
        self.request = request
        self.llm = llm


    def select_outputs(self, query: str) -> list[str]:
        
