# Prolog Path Planner for Taiat

This directory contains a Prolog-based path planner for determining optimal execution sequences in Taiat queries.

## Overview

The Prolog path planner takes an `AgentGraphNodeSet` and a list of desired outputs, then determines the optimal execution path to produce those outputs. It handles:

- Dependency resolution
- Topological sorting for execution order
- Validation of output availability
- Circular dependency detection

## Files

- `path_planner.pl` - The main Prolog script containing the path planning logic
- `prolog_interface.py` - Python interface for calling the Prolog planner from Taiat
- `test_path_planner.py` - Test script demonstrating usage
- `README.md` - This documentation file

## Requirements

- GNU Prolog (for compiling and running the Prolog scripts)
- Python 3.7+ (for the interface)
- Taiat package (for data structures)

## Installation

1. Install GNU Prolog:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gprolog
   
   # macOS
   brew install gprolog
   
   # Windows
   # Download from http://www.gprolog.org/
   ```

2. Ensure the Taiat package is available in your Python environment.

## Usage

### Basic Usage

```python
from taiat.base import AgentGraphNodeSet, AgentData
from prolog_interface import plan_taiat_path, validate_taiat_outputs

# Create your node set
node_set = AgentGraphNodeSet(nodes=[...])

# Define desired outputs
desired_outputs = [
    AgentData(name="final_result", parameters={}, description="Final result", data=None)
]

# Plan the execution path
execution_path = plan_taiat_path(node_set, desired_outputs)
print(f"Execution path: {execution_path}")

# Validate outputs
is_valid = validate_taiat_outputs(node_set, desired_outputs)
print(f"Outputs can be produced: {is_valid}")
```

### Advanced Usage

```python
from prolog_interface import PrologPathPlanner

# Create planner instance
planner = PrologPathPlanner()

# Plan execution path
execution_path = planner.plan_path(node_set, desired_outputs)

# Validate outputs
is_valid = planner.validate_outputs(node_set, desired_outputs)

# Get all available outputs
available_outputs = planner.get_available_outputs(node_set)
```

## Prolog Script Details

### Main Predicates

- `plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath)` - Main planning predicate
- `validate_outputs(NodeSet, DesiredOutputs, Valid)` - Validate output availability
- `available_outputs(NodeSet, AvailableOutputs)` - Get all available outputs
- `has_circular_dependencies(NodeSet, HasCircular)` - Check for circular dependencies

### Data Structures

The Prolog script uses these data structures:

```prolog
% Agent data
agent_data(Name, Parameters, Description, Data)

% Graph node
node(Name, Description, Inputs, Outputs)

% Node set
agent_graph_node_set(Nodes)
```

### Algorithm

1. **Dependency Resolution**: For each desired output, find all nodes that produce it
2. **Recursive Dependency Collection**: For each producing node, collect all its dependencies
3. **Topological Sorting**: Sort nodes by dependency order to ensure correct execution sequence
4. **Validation**: Check that all required outputs can be produced

## Testing

Run the test script to verify the path planner works correctly:

```bash
cd taiat/src/prolog
python test_path_planner.py
```

The test script includes examples for:
- Simple linear pipeline
- Complex multi-path dependencies
- Multiple output requests
- Invalid output handling

### Prolog Unit Tests

You can also run the Prolog unit tests directly:

```bash
cd taiat/src/prolog
gplc path_planner_test.pl path_planner.pl -o test_path_planner
./test_path_planner
```

The Prolog unit tests cover:
- All helper predicates
- Path planning algorithms
- Validation functions
- Edge cases and error conditions

## Integration with TaiatManager

The Prolog path planner can be integrated into the TaiatManager to replace or supplement the current path planning logic. The interface is designed to be easily callable from the existing Taiat codebase.

### Example Integration

```python
# In TaiatManager or TaiatBuilder
from prolog_interface import plan_taiat_path

def get_plan_with_prolog(self, query: TaiatQuery, goal_outputs: list[AgentData]):
    # Convert goal outputs to the format expected by the planner
    execution_path = plan_taiat_path(self.node_set, goal_outputs)
    
    if execution_path is None:
        return (None, "error", "Failed to plan execution path")
    
    # Convert path back to AgentGraphNode objects
    path_nodes = [self._get_node_by_name(name) for name in execution_path]
    query.path = path_nodes
    
    return (self.graph, "success", "")
```

## Error Handling

The Prolog path planner handles several error conditions:

- **Missing outputs**: When desired outputs cannot be produced by any node
- **Circular dependencies**: When the dependency graph contains cycles
- **Invalid node sets**: When the node set structure is malformed
- **Timeout**: When Prolog execution takes too long (30-second default timeout)

## Performance Considerations

- The Prolog planner uses a 30-second timeout by default
- For large node sets, consider breaking them into smaller subsets
- The topological sort algorithm has O(V + E) complexity where V is nodes and E is edges
- Memory usage scales with the number of nodes and their dependencies

## Future Enhancements

Potential improvements to the Prolog path planner:

1. **Parameter matching**: More sophisticated parameter subset matching
2. **Cost optimization**: Consider execution costs when choosing between multiple paths
3. **Parallel execution**: Identify nodes that can run in parallel
4. **Caching**: Cache planning results for repeated queries
5. **Alternative paths**: Find multiple valid execution paths

## Troubleshooting

### Common Issues

1. **GNU Prolog not found**: Ensure GNU Prolog is installed and in your PATH
2. **Compilation errors**: Check that the Prolog syntax is compatible with GNU Prolog
3. **Import errors**: Make sure the Taiat package is available in your Python environment
4. **Timeout errors**: Increase the timeout value for complex node sets
5. **Parsing errors**: Check that your AgentData and AgentGraphNode objects are properly formatted

### Debug Mode

Enable debug output by modifying the Prolog interface:

```python
# In prolog_interface.py, add debug=True to subprocess.run calls
result = subprocess.run(
    ['gplc', '-o', temp_file_path + '.exe', temp_file_path],
    capture_output=True,
    text=True,
    timeout=30
)
print(f"Prolog compilation stdout: {result.stdout}")
print(f"Prolog compilation stderr: {result.stderr}")

exec_result = subprocess.run(
    [temp_file_path + '.exe'],
    capture_output=True,
    text=True,
    timeout=30
)
print(f"Prolog execution stdout: {exec_result.stdout}")
print(f"Prolog execution stderr: {exec_result.stderr}")
``` 