# Haskell Path Planner

This directory contains the Haskell implementation of the Taiat path planner, providing high-performance execution path planning for agent graphs.

## Overview

The Haskell path planner is implemented as a daemon that maintains a persistent connection to the Python interface, eliminating the overhead of spawning new processes for each request. This approach provides better performance for frequent path planning operations.

## Features

- **High Performance**: Optimized Haskell implementation for complex path planning
- **Daemon Architecture**: Persistent connection eliminates process startup overhead
- **Parameter Constraint Enforcement**: Strict parameter matching for accurate path planning
- **Circular Dependency Detection**: Prevents infinite loops in execution paths
- **JSON Communication**: Standardized interface for cross-language communication

## Build Requirements

- **GHC**: Glasgow Haskell Compiler (version 8.10 or later)
- **Cabal**: Cabal package manager for Haskell
- **aeson**: JSON parsing library (automatically installed by Cabal)

## Building

### Quick Build

Use the provided build script:

```bash
cd haskell
./build.sh
```

This will:
1. Clean previous builds
2. Build the binary using Cabal
3. Copy the binary to the current directory
4. Make it executable

### Manual Build

If you prefer to build manually:

```bash
cd haskell
cabal clean
cabal build
```

The binary will be created in `dist-newstyle/build/` and can be copied to the current directory.

## Usage

### Python Interface

The Haskell path planner is integrated into the Python interface via the `path_planner_interface.py` module.

#### Simple Usage

```python
from haskell.path_planner_interface import plan_path, validate_outputs
from taiat.base import AgentGraphNodeSet, AgentData

# Plan an execution path
result = plan_path(node_set, desired_outputs, external_inputs)

# Validate outputs
is_valid = validate_outputs(node_set, desired_outputs)
```

#### Advanced Usage

```python
from haskell.path_planner_interface import PathPlanner

# Create a planner instance with custom configuration
planner = PathPlanner(haskell_binary_path="/path/to/binary", auto_start=True)

# Use context manager for automatic cleanup
with PathPlanner() as planner:
    result = planner.plan_path(node_set, desired_outputs, external_inputs)
```

### Available Functions

- `plan_path(node_set, desired_outputs, external_inputs=None)`: Plan execution path
- `validate_outputs(node_set, desired_outputs)`: Validate that outputs can be produced
- `get_available_outputs(node_set)`: Get all available outputs
- `has_circular_dependencies(node_set)`: Check for circular dependencies

## Architecture

### Daemon Mode

The Haskell binary runs in daemon mode (`--daemon` flag) and communicates with Python via:

1. **stdin/stdout**: JSON-based request/response protocol
2. **Request IDs**: Each request has a unique ID for response matching
3. **Background Thread**: Python maintains a reader thread for responses
4. **Thread Safety**: Lock-based synchronization for concurrent requests

### Global Instance

The Python interface maintains a global daemon instance that:
- Automatically starts on first use
- Handles reconnection if the process dies
- Provides thread-safe access
- Manages lifecycle automatically

## Performance

The daemon approach provides several performance benefits:

- **Eliminates Process Startup Overhead**: No need to spawn new processes
- **Persistent Connection**: Maintains connection between requests
- **Better Resource Utilization**: Reduces system resource usage
- **Consistent Latency**: Lower variance in response times
- **Efficient for High-Frequency Usage**: Optimized for frequent calls

### Performance Characteristics

- **Startup Time**: ~10ms for first call
- **Subsequent Calls**: ~10ms per call
- **Memory Usage**: Minimal impact
- **Concurrent Requests**: Thread-safe with locking

## Testing

Run the test suite:

```bash
# Run all path planner tests
python3 -m pytest tests/test_path_planner.py -v

# Run performance comparison
python3 haskell/performance_comparison.py
```

## Troubleshooting

### Common Issues

1. **Binary Not Found**: Run `./build.sh` to build the binary
2. **Cabal Not Found**: Install Cabal with `cabal-install`
3. **Permission Denied**: Ensure the binary is executable (`chmod +x taiat-path-planner`)
4. **Daemon Not Starting**: Check that the binary supports `--daemon` mode

### Debug Mode

For debugging, you can run the Haskell binary directly:

```bash
./taiat-path-planner --daemon
```

This will start the daemon and you can send JSON requests via stdin.

## Development

### Adding New Functions

1. Add the function to `PathPlanner.hs`
2. Add the handler to `Main.hs` in `processDaemonRequest`
3. Add the Python interface method to `PathPlanner` class
4. Add convenience function if needed
5. Update tests

### Modifying the Protocol

The JSON protocol is defined in the Haskell code. Changes should maintain backward compatibility or include versioning.

## License

This implementation is part of the Taiat project and follows the same licensing terms. 