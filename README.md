# taiat
Three Agents In A Trenchcoat (Taiat) is a tool built on top of Langgraph to make building dependency graphs among agents even easier. It takes a series of agents, the expected outputs and needed inputs of each, and then executes workflows specifically to produce a desired output. With the addition of a tool to select desired outputs from a natural language query, this can be a full solution for question answering with an agent workflow.

## Overview

Taiat can take:
- a series of defined agents, with name, parameters (which can represent constraints), descriptions, and associated agent functions
- a query from a user (e.g. "Train a regression model on the diabetes dataset, and summarize the results.")
- an output processor that turns user queries into a selection of input data and desired outputs
- an LLM API

... and produce an agent graph that will run all necessary agents, including any dependencies for the desired outputs.


## Agent Graph

Taiat creates two different types of graphs:
- The TaiatBuilder `build()` method creates and returns a full node dependency graph, where each node is an agent, and each edge is a dependency. This graph can be run in the case that all outputs must be produced. The graph is traversed to execute the agents in the correct order.
- The TaiatBuilder `get_plan()` method creates and returns a specific plan subgraph to produce one or more desired outputs. This subgraph will be pruned to only produce inputs necessary for the specified outputs. This is generally the approach an agent will want to use to answer specific queries.

The primary difference between TAIAT and standard Langgraph is that TAIAT allows for the specification of constraints, such that each agent will be run when its dependencies have been satisfied - specifically, that inputs to the agent have been produced by agents that provide them as outputs. The selection of next task is handled by the TaiatManager, which looks for unfulfilled dependencies and selects the next agent to run to fulfill them.

## Query Database

There is a simple postgres implementation of a query/output DB interface for collecting query data and saving outputs, e.g.

```
        db = PostgresDatabase(session_maker=self.session_maker)
        db.add_run(TaiatQuery( ... ))
```

See `tests/db.py` for an example.

## Example

To use Taiat, provide a series of agents, a query, and an output processor. A simple example is in the examples/ folder, which contains an agent that will perform various kinds of analyses on Kaggle CVS datasets. It can be tested on the command line with
`python taiat/examples/ml_workflow.py --request="<request here>"`. Give it queries such as:

```
Perform clustering on the points in the Pima dataset.
Give me a report on the results when you train a nearest neighbor classifier on the iris data.
```

and the agent will perform all relevant tasks, including (if appropriate for the request) downloading the data, performing a train/test split, performing the appropriate processing upon the dataset, performing an evaluation, and providing a summary of results. (Note that this requires access to both the OpenAI API and the Kaggle API, in its current implementation.)

## Note
This is pre-alpha software, and to put it lightly, some things are missing. Specifically:
- There is not a canonical approach to deal with multiple possible dependency fulfillments; if two agents can both fulfill the requirements of a dependency, it's currently an arbitrary choice.
- Parameter resolution is not rigorous (currently, we assume well-formed parameters that represent constraints and just look for matching supersets).
