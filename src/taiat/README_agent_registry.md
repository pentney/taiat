# Agent Registry System

The Agent Registry system allows you to register agent functions and have the TaiatGraphArchitect automatically resolve function names to actual callable functions.

## Overview

The system consists of two main components:

1. **AgentRegistry**: A registry that maps function names to actual callable functions
2. **TaiatGraphArchitect**: Enhanced to use the registry to resolve function names

## How It Works

1. Users register their agent functions with the `AgentRegistry` using descriptive names
2. The LLM in `TaiatGraphArchitect` generates `AgentGraphNode` objects with function names as strings
3. The architect resolves these function names to actual functions using the registry
4. The final `AgentGraphNodeSet` contains nodes with real callable functions

## Usage Example

```python
from taiat.graph_architect import AgentRegistry, TaiatGraphArchitect
from langchain_anthropic import ChatAnthropic
from taiat.base import State, AgentData

# Define your agent functions
def data_processor_agent(state: State) -> State:
    # Process data logic here
    return state

def analyzer_agent(state: State) -> State:
    # Analysis logic here
    return state

# Create and populate the registry
registry = AgentRegistry()
registry.register("data_processor", data_processor_agent)
registry.register("analyzer", analyzer_agent)

# Create the graph architect
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
architect = TaiatGraphArchitect(
    llm=llm,
    agent_registry=registry,
    verbose=True
)

# Build an AgentGraphNodeSet from a description
description = """
I need a pipeline that processes data and then analyzes it.
"""
agent_graph_node_set = architect.build(description)
```

## AgentRegistry API

### Methods

- `register(name: str, function: Callable) -> None`: Register a function with a name
- `get(name: str) -> Optional[Callable]`: Get a function by name
- `list_registered() -> list[str]`: Get all registered function names
- `clear() -> None`: Clear all registered functions

### Example

```python
registry = AgentRegistry()

# Register functions
registry.register("process_data", my_data_processor)
registry.register("analyze_results", my_analyzer)

# List registered functions
print(registry.list_registered())  # ['process_data', 'analyze_results']

# Get a function
func = registry.get("process_data")
if func:
    result = func(state)
```

## TaiatGraphArchitect Changes

The `TaiatGraphArchitect` constructor now requires an `AgentRegistry`:

```python
def __init__(self, llm: BaseChatModel, agent_registry: AgentRegistry, verbose: bool = False, llm_explanation: bool = False):
```

The `build()` method now:
1. Generates `AgentGraphNode` objects with function names as strings
2. Resolves these names to actual functions using the registry
3. Returns a complete `AgentGraphNodeSet` with callable functions

## Error Handling

If a function name cannot be resolved, the system will raise a `ValueError` with:
- The missing function name
- A list of available function names in the registry

Example error:
```
ValueError: Function 'unknown_function' not found in agent registry. Available functions: ['data_processor', 'analyzer', 'report_generator']
```

## Best Practices

1. **Use descriptive function names**: Choose names that clearly describe what the agent does
2. **Register all required functions**: Make sure all functions referenced in your descriptions are registered
3. **Handle errors gracefully**: Check for missing functions and provide helpful error messages
4. **Keep registry organized**: Use consistent naming conventions for your functions

## Example Workflow

1. Define your agent functions
2. Create an `AgentRegistry` and register your functions
3. Create a `TaiatGraphArchitect` with the registry
4. Provide a textual description of your desired agent graph
5. The architect will generate a complete `AgentGraphNodeSet` with resolved functions
6. Use the `AgentGraphNodeSet` in your Taiat workflow

See `examples/agent_registry_example.py` for a complete working example. 