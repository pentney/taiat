# Prolog Integration in Taiat

This document describes the Prolog integration in taiat, including build process, dependencies, and usage.

## Overview

Taiat includes Prolog modules for advanced path planning and dependency resolution. The Prolog integration provides:

- **Path Planning**: Optimal execution sequence determination
- **Dependency Resolution**: Automatic dependency graph analysis
- **Validation**: Output availability and circular dependency detection
- **Performance**: Compiled Prolog executables for fast execution

## Prerequisites

### Required: GNU Prolog

Taiat requires GNU Prolog (gprolog) for full functionality. The build process will automatically detect if gprolog is available and provide installation instructions if needed.

#### Installation by Platform

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install gprolog
```

**Fedora:**
```bash
sudo dnf install gprolog
```

**CentOS/RHEL:**
```bash
sudo yum install gprolog
```

**Arch Linux:**
```bash
sudo pacman -S gprolog
```

**macOS:**
```bash
brew install gprolog
```

**Windows:**
Download from [http://www.gprolog.org/](http://www.gprolog.org/) and add to PATH.

## Build Process

### Automatic Build

The build process automatically handles Prolog compilation:

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

During installation, the setup script will:
1. Check for gprolog availability
2. Compile all `.pl` files to executables
3. Place compiled files in `src/taiat/prolog/compiled/`
4. Include compiled files in the package distribution

### Manual Build

For development, you can manually compile Prolog files:

```bash
# Using make
make build-prolog

# Or manually
mkdir -p src/taiat/prolog/compiled
gplc src/taiat/prolog/path_planner.pl -o src/taiat/prolog/compiled/path_planner
gplc src/taiat/prolog/path_planner_test.pl -o src/taiat/prolog/compiled/path_planner_test
```

## File Structure

```
src/taiat/prolog/
├── __init__.py                    # Python package initialization
├── path_planner.pl               # Main Prolog path planning logic
├── path_planner_test.pl          # Prolog unit tests
├── optimized_prolog_interface.py # Python interface to Prolog
├── proper_global_comparison.py   # Additional Prolog utilities
├── test_path_planner.py          # Python test script
├── README.md                     # Prolog-specific documentation
└── compiled/                     # Compiled Prolog executables (auto-generated)
    ├── path_planner
    └── path_planner_test
```

## Usage

### Basic Usage

```python
from taiat.prolog import optimized_prolog_interface
from taiat.base import AgentGraphNodeSet, AgentData

# Create planner instance
planner = optimized_prolog_interface.PrologPathPlanner()

# Plan execution path
node_set = AgentGraphNodeSet(nodes=[...])
desired_outputs = [AgentData(name="result", ...)]

execution_path = planner.plan_path(node_set, desired_outputs)
```

### Integration with TaiatManager

```python
from taiat.prolog import optimized_prolog_interface

class TaiatManager:
    def get_plan_with_prolog(self, query, goal_outputs):
        planner = optimized_prolog_interface.PrologPathPlanner()
        execution_path = planner.plan_path(self.node_set, goal_outputs)
        
        if execution_path is None:
            return (None, "error", "Failed to plan execution path")
        
        # Convert path back to AgentGraphNode objects
        path_nodes = [self._get_node_by_name(name) for name in execution_path]
        query.path = path_nodes
        
        return (self.graph, "success", "")
```

## Testing

### Running Tests

```bash
# Run all tests including Prolog integration
make test

# Run only Prolog-specific tests
make test-prolog

# Run Python tests without Prolog
pytest tests/ --ignore=tests/test_prolog_integration.py
```

### Test Structure

- `tests/test_prolog_integration.py` - Integration tests for Prolog functionality
- `src/taiat/prolog/test_path_planner.py` - Python tests for Prolog interface
- `src/taiat/prolog/path_planner_test.pl` - Prolog unit tests

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Install gprolog
make install-gprolog

# 2. Install taiat in development mode
make develop

# 3. Build Prolog files
make build-prolog

# 4. Run tests
make test
```

### Adding New Prolog Files

1. Add your `.pl` file to `src/taiat/prolog/`
2. Update `setup.py` if needed (usually automatic)
3. Add tests to `tests/test_prolog_integration.py`
4. Update this documentation

### Continuous Integration

The GitHub Actions workflow automatically:
- Installs gprolog in CI environment
- Compiles Prolog files
- Runs integration tests
- Tests graceful degradation without gprolog

## Troubleshooting

### Common Issues

**"gprolog not found"**
- Install gprolog using your system's package manager
- Reinstall taiat: `pip install --force-reinstall .`

**"Prolog compilation failed"**
- Check that gprolog is properly installed: `gplc --version`
- Verify Prolog syntax in your `.pl` files
- Check file permissions

**"Import error for taiat.prolog"**
- Ensure taiat is installed: `pip install -e .`
- Check that `src/taiat/prolog/__init__.py` exists

**"Compiled Prolog executable not found"**
- Rebuild Prolog files: `make build-prolog`
- Check that `src/taiat/prolog/compiled/` directory exists

### Debug Mode

Enable debug output by setting environment variable:
```bash
export TAIAT_PROLOG_DEBUG=1
```

### Performance Optimization

- Compiled Prolog executables are much faster than interpreted
- Large node sets may require optimization of Prolog predicates
- Consider caching frequently used path planning results

## Future Enhancements

- **Alternative Prolog Engines**: Support for SWI-Prolog, YAP
- **Parallel Execution**: Identify parallelizable execution paths
- **Cost Optimization**: Consider execution costs in path planning
- **Caching**: Cache planning results for repeated queries
- **WebAssembly**: Compile Prolog to WebAssembly for web deployment

## Contributing

When contributing to the Prolog integration:

1. Follow Prolog coding standards
2. Add comprehensive tests
3. Update documentation
4. Test on multiple platforms
5. Ensure graceful degradation without gprolog 