# Taiat Haskell Path Planner

A high-performance Haskell implementation of the Taiat path planning system, providing better performance and type safety.

## Features

- **Type-safe path planning** with comprehensive error handling
- **Parameter matching** with flexible subset matching
- **Topological sorting** for dependency resolution
- **Circular dependency detection**
- **JSON serialization** for easy integration
- **Performance measurement** capabilities
- **Comprehensive test suite** with 29+ test cases

## Building

```bash
cabal build
```

## Running Tests

### Simple Test Runner (Recommended)
```bash
cabal exec -- runhaskell SimpleTestRunner.hs
```

### HUnit Test Suite
```bash
cabal exec -- runhaskell PathPlannerTest.hs
```

## Test Coverage

The test suite covers all functionality for the path planning system:

### Simple Tests (9 tests)
- `test_agent_data_name` - Basic agent data name extraction
- `test_agent_data_match` - Exact agent data matching
- `test_agent_data_match_params_subset` - Parameter subset matching
- `test_agent_data_match_params_conflict` - Parameter conflict detection
- `test_agent_data_match_params_empty` - Empty parameter handling
- `test_agent_data_match_name_mismatch` - Name mismatch detection
- `test_remove_duplicates` - Duplicate removal with order preservation
- `test_remove_duplicates_empty` - Empty list handling
- `test_remove_duplicates_single` - Single element handling

### Node Operation Tests (5 tests)
- `test_node_produces_output` - Node output production verification
- `test_nodes_producing_output_name` - Output name-based node finding
- `test_node_dependencies` - Node dependency resolution
- `test_node_ready` - Node readiness checking
- `test_node_not_ready` - Node not ready state

### Path Planning Tests (3 tests)
- `test_required_nodes` - Required node identification
- `test_topological_sort` - Dependency ordering
- `test_plan_execution_path` - Execution path generation

### Validation Tests (3 tests)
- `test_validate_outputs` - Output validation
- `test_invalid_output` - Invalid output detection
- `test_available_outputs` - Available output enumeration

### Edge Case Tests (4 tests)
- `test_empty_node_set` - Empty graph handling
- `test_no_required_nodes` - Impossible output handling
- `test_circular_dependencies` - Circular dependency detection
- `test_no_circular_dependencies` - Valid graph verification

### Parameter Matching Tests (3 tests)
- `test_flexible_parameter_matching` - Bidirectional parameter matching
- `test_specificity_scoring` - Parameter specificity calculation
- `test_parameter_matching` - Real-world parameter matching

### Complex Path Planning Tests (2 tests)
- `test_complex_path_planning` - Multi-path execution planning
- `test_multiple_outputs` - Multiple output handling

## Performance

The Haskell implementation provides significant performance improvements:

- **Faster execution** due to compiled code
- **Better memory management** with lazy evaluation
- **Type safety** preventing runtime errors
- **Optimized algorithms** for path planning

## API

### Core Functions

```haskell
-- Plan execution path for desired outputs
planExecutionPath :: AgentGraphNodeSet -> [AgentData] -> [Text]

-- Validate that outputs can be produced
validateOutputs :: AgentGraphNodeSet -> [AgentData] -> Bool

-- Find available outputs
availableOutputs :: AgentGraphNodeSet -> [AgentData]

-- Check for circular dependencies
hasCircularDependencies :: AgentGraphNodeSet -> Bool
```

### Data Structures

```haskell
data AgentData = AgentData
    { agentDataName :: Text
    , agentDataParameters :: Map Text Text
    , agentDataDescription :: Text
    , agentDataData :: Maybe Text
    }

data Node = Node
    { nodeName :: Text
    , nodeDescription :: Text
    , nodeInputs :: [AgentData]
    , nodeOutputs :: [AgentData]
    }

data AgentGraphNodeSet = AgentGraphNodeSet
    { agentGraphNodeSetNodes :: [Node]
    }
```

## Integration

The Haskell path planner can be integrated with Python through the `haskell_interface.py` module, providing the primary path planning mechanism for Taiat.

## Development

### Adding New Tests

1. Add test function to `SimpleTestRunner.hs`
2. Add test to appropriate test suite in the main function
3. Run tests to verify functionality

### Extending Functionality

1. Add new functions to `PathPlanner.hs`
2. Add corresponding tests
3. Update this README with new features

## License

MIT License - see LICENSE file for details. 