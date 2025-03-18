# taiat
Three Agents In A Trenchcoat (Taiat) is a tool built on top of Langgraph to make building dependency graphs among agents even easier. It takes a series of agents, the expected outputs and needed inputs of each, and then executes workflows specifically to produce a desired output. With the addition of a tool to select desired outputs from an NLP query, this can be a full solution for question answering with an agent workflow.

## Overview

Taiat can take:
- a series of defined agents, with name, parameters (which can represent constraints), descriptions, and associated agent functions
- a query from a user (e.g. "Train a regression model on the diabetes dataset, and summarize the results.")
- an output processor that turns user queries into a selection of input data and desired outputs
- an LLM API

... and produce an agent graph that will; run all necessary agents, including any dependencies for the desired outputs. A simple example is in the examples/ folder, which contains an agent that will perform various kinds of analyses on Kaggle CVS datasets. It can be tested on the command line with
`python taiat/examples/kaggle.py --request="<request here>"`. Give it queries such as:

```
Perform clustering on the points in the Pima dataset.
Give me a report on the results when you train a nearest neighbor classifier on the iris data.
```

and the agent will perform all relevant tasks, including (if appropriate for the request) downloading the data, performing a train/test split, performing the appropriate processing upon the dataset, performing an evaluation, and providing a summary of results. (Note that this requires access to both the OpenAI API and the Kaggle API, in its current implementation.)

## Note
This is pre-alpha software, and some things are missing. Specifically:
- There is not a canonical approach to deal with multiple possible dependency fulfillments; if two agents can both fulfill the requirements of a dependency, it's currently an arbitrary choice.
- Parameter resolution is not rigorous (currently, we assume well-formed parameters that represent constraints and just look for matching supersets).
