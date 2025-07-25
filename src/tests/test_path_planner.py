"""
Test script for the Global Optimized Prolog Path Planner.

This script demonstrates how to use the global optimized Prolog path planner to determine
execution paths for Taiat queries.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData
from prolog.taiat_path_planner import (
    TaiatPathPlanner,
    plan_taiat_path,
    plan_taiat_path_global,
    get_global_planner,
    clear_global_planner,
)


def create_example_node_set():
    """
    Create an example AgentGraphNodeSet for testing.

    This creates a simple data processing pipeline:
    data_loader -> preprocessor -> analyzer -> visualizer
    """

    # Define the nodes
    data_loader = AgentGraphNode(
        name="data_loader",
        description="Load data from source",
        inputs=[],
        outputs=[
            AgentData(
                name="raw_data",
                parameters={},
                description="Raw data from source",
                data=None,
            )
        ],
    )

    preprocessor = AgentGraphNode(
        name="preprocessor",
        description="Preprocess the data",
        inputs=[
            AgentData(
                name="raw_data",
                parameters={},
                description="Raw data from source",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="processed_data",
                parameters={},
                description="Preprocessed data",
                data=None,
            )
        ],
    )

    analyzer = AgentGraphNode(
        name="analyzer",
        description="Analyze the data",
        inputs=[
            AgentData(
                name="processed_data",
                parameters={},
                description="Preprocessed data",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="analysis_results",
                parameters={},
                description="Analysis results",
                data=None,
            )
        ],
    )

    visualizer = AgentGraphNode(
        name="visualizer",
        description="Create visualizations",
        inputs=[
            AgentData(
                name="analysis_results",
                parameters={},
                description="Analysis results",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="visualizations",
                parameters={},
                description="Generated visualizations",
                data=None,
            )
        ],
    )

    # Create the node set
    node_set = AgentGraphNodeSet(
        nodes=[data_loader, preprocessor, analyzer, visualizer]
    )
    return node_set


def create_complex_example_node_set():
    """
    Create a more complex example with multiple paths and dependencies.
    """

    # Data loading nodes
    csv_loader = AgentGraphNode(
        name="csv_loader",
        description="Load CSV data",
        inputs=[],
        outputs=[
            AgentData(name="csv_data", parameters={}, description="CSV data", data=None)
        ],
    )

    json_loader = AgentGraphNode(
        name="json_loader",
        description="Load JSON data",
        inputs=[],
        outputs=[
            AgentData(
                name="json_data", parameters={}, description="JSON data", data=None
            )
        ],
    )

    # Data processing nodes
    csv_processor = AgentGraphNode(
        name="csv_processor",
        description="Process CSV data",
        inputs=[
            AgentData(name="csv_data", parameters={}, description="CSV data", data=None)
        ],
        outputs=[
            AgentData(
                name="processed_csv",
                parameters={},
                description="Processed CSV data",
                data=None,
            )
        ],
    )

    json_processor = AgentGraphNode(
        name="json_processor",
        description="Process JSON data",
        inputs=[
            AgentData(
                name="json_data", parameters={}, description="JSON data", data=None
            )
        ],
        outputs=[
            AgentData(
                name="processed_json",
                parameters={},
                description="Processed JSON data",
                data=None,
            )
        ],
    )

    # Data merging node
    data_merger = AgentGraphNode(
        name="data_merger",
        description="Merge processed data",
        inputs=[
            AgentData(
                name="processed_csv",
                parameters={},
                description="Processed CSV data",
                data=None,
            ),
            AgentData(
                name="processed_json",
                parameters={},
                description="Processed JSON data",
                data=None,
            ),
        ],
        outputs=[
            AgentData(
                name="merged_data", parameters={}, description="Merged data", data=None
            )
        ],
    )

    # Analysis nodes
    statistical_analyzer = AgentGraphNode(
        name="statistical_analyzer",
        description="Perform statistical analysis",
        inputs=[
            AgentData(
                name="merged_data", parameters={}, description="Merged data", data=None
            )
        ],
        outputs=[
            AgentData(
                name="statistical_results",
                parameters={},
                description="Statistical analysis results",
                data=None,
            )
        ],
    )

    ml_analyzer = AgentGraphNode(
        name="ml_analyzer",
        description="Perform machine learning analysis",
        inputs=[
            AgentData(
                name="merged_data", parameters={}, description="Merged data", data=None
            )
        ],
        outputs=[
            AgentData(
                name="ml_results",
                parameters={},
                description="ML analysis results",
                data=None,
            )
        ],
    )

    # Report generator
    report_generator = AgentGraphNode(
        name="report_generator",
        description="Generate comprehensive report",
        inputs=[
            AgentData(
                name="statistical_results",
                parameters={},
                description="Statistical analysis results",
                data=None,
            ),
            AgentData(
                name="ml_results",
                parameters={},
                description="ML analysis results",
                data=None,
            ),
        ],
        outputs=[
            AgentData(
                name="final_report",
                parameters={},
                description="Final comprehensive report",
                data=None,
            )
        ],
    )

    # Create the node set
    node_set = AgentGraphNodeSet(
        nodes=[
            csv_loader,
            json_loader,
            csv_processor,
            json_processor,
            data_merger,
            statistical_analyzer,
            ml_analyzer,
            report_generator,
        ]
    )
    return node_set


def test_simple_path():
    """Test simple path planning."""
    print("Testing simple path planning...")

    node_set = create_example_node_set()
    desired_outputs = [
        AgentData(
            name="visualizations",
            parameters={},
            description="Generated visualizations",
            data=None,
        )
    ]

    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    if execution_path:
        print(f"✅ Simple path planning successful: {execution_path}")
        expected_path = ["data_loader", "preprocessor", "analyzer", "visualizer"]
        if execution_path == expected_path:
            print("✅ Path matches expected order")
        else:
            print(f"⚠️  Path differs from expected: {expected_path}")
    else:
        print("❌ Simple path planning failed")


def test_complex_path():
    """Test complex path planning with multiple dependencies."""
    print("\nTesting complex path planning...")

    node_set = create_complex_example_node_set()
    desired_outputs = [
        AgentData(
            name="final_report",
            parameters={},
            description="Final comprehensive report",
            data=None,
        )
    ]

    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    if execution_path:
        print(f"✅ Complex path planning successful: {execution_path}")
        # Check that dependencies are satisfied
        print("✅ Path planning completed")
    else:
        print("❌ Complex path planning failed")


def test_multiple_outputs():
    """Test planning for multiple outputs."""
    print("\nTesting multiple outputs planning...")

    node_set = create_complex_example_node_set()
    desired_outputs = [
        AgentData(
            name="statistical_results",
            parameters={},
            description="Statistical analysis results",
            data=None,
        ),
        AgentData(
            name="ml_results",
            parameters={},
            description="ML analysis results",
            data=None,
        ),
    ]

    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    if execution_path:
        print(f"✅ Multiple outputs planning successful: {execution_path}")
        print("✅ Path planning completed")
    else:
        print("❌ Multiple outputs planning failed")


def test_invalid_output():
    """Test planning with invalid output."""
    print("\nTesting invalid output...")

    node_set = create_example_node_set()
    desired_outputs = [
        AgentData(
            name="nonexistent_output",
            parameters={},
            description="Output that doesn't exist",
            data=None,
        )
    ]

    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    if execution_path is None:
        print("✅ Correctly handled invalid output")
    else:
        print("❌ Should have failed for invalid output")


def test_convenience_functions():
    """Test the convenience functions."""
    print("\nTesting convenience functions...")

    node_set = create_example_node_set()
    desired_outputs = [
        AgentData(
            name="visualizations",
            parameters={},
            description="Generated visualizations",
            data=None,
        )
    ]

    # Test the global optimized function
    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    if execution_path:
        print("✅ Global optimized function works")
    else:
        print("❌ Global optimized function failed")


def test_prolog_unit_tests():
    """Run the Prolog unit tests directly."""
    print("\nRunning Prolog unit tests...")

    try:
        # Test the planner directly
        planner = TaiatPathPlanner()
        print("✅ Prolog planner initialization successful")

        # Test with simple data
        node_set = create_example_node_set()
        desired_outputs = [
            AgentData(
                name="visualizations",
                parameters={},
                description="Generated visualizations",
                data=None,
            )
        ]

        execution_path = planner.plan_path(node_set, desired_outputs)
        if execution_path:
            print("✅ Prolog planner path planning successful")
        else:
            print("❌ Prolog planner path planning failed")

    except Exception as e:
        print(f"❌ Prolog unit tests failed: {e}")


def test_param_subset_match():
    """Test parameter subset matching (should succeed)."""
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    node = AgentGraphNode(
        name="model_node",
        description="Provides a model",
        inputs=[],
        outputs=[
            AgentData(
                name="model",
                parameters={"type": "logistic_regression", "version": "v1"},
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="model",
            parameters={"type": "logistic_regression"},
            description="",
            data=None,
        )
    ]
    path = plan_taiat_path_global(node_set, desired_outputs)
    if path == ["model_node"]:
        print("✅ test_param_subset_match passed")
    else:
        print(f"❌ test_param_subset_match failed: {path}")


def test_param_conflict():
    """Test parameter conflict (should fail)."""
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    node = AgentGraphNode(
        name="model_node",
        description="Provides a model",
        inputs=[],
        outputs=[
            AgentData(
                name="model",
                parameters={"type": "neural_network", "version": "v1"},
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="model",
            parameters={"type": "logistic_regression"},
            description="",
            data=None,
        )
    ]
    path = plan_taiat_path_global(node_set, desired_outputs)
    if not path:
        print("✅ test_param_conflict passed")
    else:
        print(f"❌ test_param_conflict failed: {path}")


def test_param_empty():
    """Test empty input parameters (should succeed)."""
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    node = AgentGraphNode(
        name="model_node",
        description="Provides a model",
        inputs=[],
        outputs=[
            AgentData(
                name="model",
                parameters={"type": "neural_network"},
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(name="model", parameters={}, description="", data=None)
    ]
    path = plan_taiat_path_global(node_set, desired_outputs)
    if path == ["model_node"]:
        print("✅ test_param_empty passed")
    else:
        print(f"❌ test_param_empty failed: {path}")


def test_name_mismatch():
    """Test name mismatch (should fail)."""
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    node = AgentGraphNode(
        name="model_node",
        description="Provides a model",
        inputs=[],
        outputs=[
            AgentData(
                name="modelB",
                parameters={"type": "logistic_regression"},
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="modelA",
            parameters={"type": "logistic_regression"},
            description="",
            data=None,
        )
    ]
    path = plan_taiat_path_global(node_set, desired_outputs)
    if not path:
        print("✅ test_name_mismatch passed")
    else:
        print(f"❌ test_name_mismatch failed: {path}")


def test_parameter_constraints_for_intermediate_nodes():
    """
    Test that parameter constraints are respected for intermediate nodes.

    This test verifies that when a node requires input X with parameters {"A":"B"},
    the planner will not select a node that produces output X with parameters {"A":"C"}.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Create test nodes
    producer_wrong_params = AgentGraphNode(
        name="producer_wrong_params",
        description="Produces output X with wrong parameters",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "C"},  # Wrong parameter value
                description="Output X with wrong parameters",
                data=None,
            )
        ],
    )

    producer_correct_params = AgentGraphNode(
        name="producer_correct_params",
        description="Produces output X with correct parameters",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},  # Correct parameter value
                description="Output X with correct parameters",
                data=None,
            )
        ],
    )

    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input X with specific parameters",
        function=dummy_function,
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},  # Requires specific parameter value
                description="Input X with specific parameters",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set
    node_set = AgentGraphNodeSet(
        nodes=[producer_wrong_params, producer_correct_params, consumer]
    )

    # Define desired outputs
    desired_outputs = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    # Plan execution path
    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    # Debug output
    print(f"Intermediate node execution path: {execution_path}")

    # Verify the path includes the correct producer and excludes the wrong one
    assert execution_path is not None, "Path planning should succeed"

    # The path should include producer_correct_params but NOT producer_wrong_params
    assert "producer_correct_params" in execution_path, (
        "Path should include the correct producer"
    )
    assert "producer_wrong_params" not in execution_path, (
        "Path should NOT include the wrong producer"
    )
    assert "consumer" in execution_path, "Path should include the consumer"

    # Verify the execution order is correct
    correct_producer_index = execution_path.index("producer_correct_params")
    consumer_index = execution_path.index("consumer")
    assert correct_producer_index < consumer_index, (
        "Producer should come before consumer"
    )

    print("✓ Parameter constraints for intermediate nodes test passed")


def test_parameter_constraints_with_multiple_options():
    """
    Test parameter constraints when multiple nodes can produce the same output type
    but with different parameters.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Create multiple producers with different parameter combinations
    producer_1 = AgentGraphNode(
        name="producer_1",
        description="Produces output Y with parameters {type: 'A', version: '1'}",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="Y",
                parameters={"type": "A", "version": "1"},
                description="Output Y with type A, version 1",
                data=None,
            )
        ],
    )

    producer_2 = AgentGraphNode(
        name="producer_2",
        description="Produces output Y with parameters {type: 'B', version: '1'}",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="Y",
                parameters={"type": "B", "version": "1"},
                description="Output Y with type B, version 1",
                data=None,
            )
        ],
    )

    producer_3 = AgentGraphNode(
        name="producer_3",
        description="Produces output Y with parameters {type: 'A', version: '2'}",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="Y",
                parameters={"type": "A", "version": "2"},
                description="Output Y with type A, version 2",
                data=None,
            )
        ],
    )

    # Consumer that requires specific parameters
    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input Y with specific parameters",
        function=dummy_function,
        inputs=[
            AgentData(
                name="Y",
                parameters={"type": "A", "version": "1"},  # Requires exact match
                description="Input Y with specific parameters",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set
    node_set = AgentGraphNodeSet(nodes=[producer_1, producer_2, producer_3, consumer])

    # Define desired outputs
    desired_outputs = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    # Plan execution path
    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    # Debug output
    print(f"Multiple options execution path: {execution_path}")

    # Verify the path includes only the correct producer
    assert execution_path is not None, "Path planning should succeed"

    # Should include producer_1 (exact match) but not the others
    assert "producer_1" in execution_path, (
        "Path should include producer_1 (exact match)"
    )
    assert "producer_2" not in execution_path, (
        "Path should NOT include producer_2 (wrong type)"
    )
    assert "producer_3" not in execution_path, (
        "Path should NOT include producer_3 (wrong version)"
    )
    assert "consumer" in execution_path, "Path should include the consumer"

    # Verify execution order
    producer_1_index = execution_path.index("producer_1")
    consumer_index = execution_path.index("consumer")
    assert producer_1_index < consumer_index, "Producer should come before consumer"

    print("✓ Parameter constraints with multiple options test passed")


def test_parameter_constraints_with_superset_matching():
    """
    Test that parameter constraints work correctly with superset matching.
    A node that produces output with additional parameters should still match
    if it includes all the required parameters.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Producer with additional parameters (superset)
    producer_superset = AgentGraphNode(
        name="producer_superset",
        description="Produces output Z with additional parameters",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="Z",
                parameters={"type": "A", "version": "1", "extra": "value"},  # Superset
                description="Output Z with additional parameters",
                data=None,
            )
        ],
    )

    # Consumer that requires only a subset of parameters
    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input Z with subset of parameters",
        function=dummy_function,
        inputs=[
            AgentData(
                name="Z",
                parameters={
                    "type": "A",
                    "version": "1",
                },  # Subset of producer's parameters
                description="Input Z with subset of parameters",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set
    node_set = AgentGraphNodeSet(nodes=[producer_superset, consumer])

    # Define desired outputs
    desired_outputs = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    # Plan execution path
    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    # Debug output
    print(f"Superset matching execution path: {execution_path}")

    # Verify the path includes the producer with superset parameters
    assert execution_path is not None, "Path planning should succeed"
    assert "producer_superset" in execution_path, (
        "Path should include producer with superset parameters"
    )
    assert "consumer" in execution_path, "Path should include the consumer"

    # Verify execution order
    producer_index = execution_path.index("producer_superset")
    consumer_index = execution_path.index("consumer")
    assert producer_index < consumer_index, "Producer should come before consumer"

    print("✓ Parameter constraints with superset matching test passed")


def test_parameter_constraints_no_match():
    """
    Test that path planning fails when no node can produce the required parameters.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import plan_taiat_path_global

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Producer with completely different parameters
    producer_wrong = AgentGraphNode(
        name="producer_wrong",
        description="Produces output W with wrong parameters",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="W",
                parameters={"type": "X", "version": "1"},  # Completely different
                description="Output W with wrong parameters",
                data=None,
            )
        ],
    )

    # Consumer that requires specific parameters
    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input W with specific parameters",
        function=dummy_function,
        inputs=[
            AgentData(
                name="W",
                parameters={"type": "Y", "version": "2"},  # No match available
                description="Input W with specific parameters",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set
    node_set = AgentGraphNodeSet(nodes=[producer_wrong, consumer])

    # Define desired outputs
    desired_outputs = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    # Plan execution path - should fail
    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    # Debug output
    print(f"No match execution path: {execution_path}")

    # Verify that path planning fails when no suitable producer exists
    assert execution_path is None, (
        "Path planning should fail when no suitable producer exists"
    )

    print("✓ Parameter constraints no match test passed")


def test_parameter_matching_debug():
    """
    Debug test to understand what's happening with parameter matching.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import TaiatPathPlanner

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Create a simple test case
    producer = AgentGraphNode(
        name="producer",
        description="Produces output X with parameters {A: 'C'}",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "C"},
                description="Output X with A=C",
                data=None,
            )
        ],
    )

    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input X with parameters {A: 'B'}",
        function=dummy_function,
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},
                description="Input X with A=B",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set
    node_set = AgentGraphNodeSet(nodes=[producer, consumer])

    # Print the Prolog representation
    planner = TaiatPathPlanner()
    node_set_str = planner._node_set_to_prolog(node_set)
    print(f"Node set Prolog representation:")
    print(node_set_str)

    # Test the agent_data_to_prolog conversion
    input_data = AgentData(name="X", parameters={"A": "B"}, description="", data=None)
    output_data = AgentData(name="X", parameters={"A": "C"}, description="", data=None)

    input_str = planner._agent_data_to_prolog(input_data)
    output_str = planner._agent_data_to_prolog(output_data)

    print(f"Input Prolog: {input_str}")
    print(f"Output Prolog: {output_str}")

    print("✓ Parameter matching debug test completed")


def test_direct_parameter_matching():
    """
    Test parameter matching directly by creating a minimal Prolog query.
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
    from prolog.taiat_path_planner import TaiatPathPlanner

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Create a simple test case with only one producer and one consumer
    # The producer has wrong parameters, so the path should fail
    producer_wrong = AgentGraphNode(
        name="producer_wrong",
        description="Produces output X with wrong parameters",
        function=dummy_function,
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "C"},  # Wrong parameter value
                description="Output X with wrong parameters",
                data=None,
            )
        ],
    )

    consumer = AgentGraphNode(
        name="consumer",
        description="Consumes input X with specific parameters",
        function=dummy_function,
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},  # Requires specific parameter value
                description="Input X with specific parameters",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )

    # Create node set with only the wrong producer
    node_set = AgentGraphNodeSet(nodes=[producer_wrong, consumer])

    # Define desired outputs
    desired_outputs = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    # Plan execution path - should fail because no suitable producer exists
    from prolog.taiat_path_planner import plan_taiat_path_global

    execution_path = plan_taiat_path_global(node_set, desired_outputs)

    print(f"Direct parameter matching test execution path: {execution_path}")

    # This should fail because there's no producer with the correct parameters
    if execution_path is None:
        print(
            "✓ Direct parameter matching test passed - correctly failed when no suitable producer exists"
        )
    else:
        print(
            f"❌ Direct parameter matching test failed - found path {execution_path} when it should have failed"
        )
        assert False, "Path planning should fail when no suitable producer exists"


def test_missing_input_error_reporting():
    """
    Test that the enhanced error reporting correctly identifies missing inputs.
    """
    from prolog.taiat_path_planner import TaiatPathPlanner
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData

    # Create a simple test case where we can test the error analysis directly
    # Create a node that needs an input that no one provides
    problem_node = AgentGraphNode(
        name="problem_node",
        description="Needs missing input",
        inputs=[
            AgentData(
                name="missing_input",
                parameters={},
                description="Input that no one provides",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="problem_output",
                parameters={},
                description="Problem output",
                data=None,
            )
        ],
    )
    
    node_set = AgentGraphNodeSet(nodes=[problem_node])
    
    # Request the problem output
    desired_outputs = [
        AgentData(
            name="problem_output",
            parameters={},
            description="Problem output",
            data=None,
        )
    ]
    
    # Test the error analysis directly
    planner = TaiatPathPlanner()
    error_details = planner._analyze_failure(node_set, desired_outputs)
    
    # Verify that the error analysis correctly identifies the missing input
    assert "missing_input" in error_details, "Error analysis should mention the missing input"
    assert "problem_node" in error_details, "Error analysis should mention which node needs the input"
    
    print("✅ Missing input error reporting test passed")


def test_secondary_path_with_agent_failure():
    """
    Test that TaiatPathPlanner finds a secondary path when the primary path fails due to an agent failure.
    """
    from prolog.taiat_path_planner import TaiatPathPlanner
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData

    # Path 1: A -> B -> OUT
    # Path 2: X -> Y -> OUT
    node_a = AgentGraphNode(
        name="A",
        description="Start A",
        inputs=[],
        outputs=[AgentData(name="mid", parameters={}, description="mid")],
    )
    node_b = AgentGraphNode(
        name="B",
        description="B",
        inputs=[AgentData(name="mid", parameters={}, description="mid")],
        outputs=[AgentData(name="OUT", parameters={}, description="out")],
    )
    node_x = AgentGraphNode(
        name="X",
        description="Start X",
        inputs=[],
        outputs=[AgentData(name="altmid", parameters={}, description="altmid")],
    )
    node_y = AgentGraphNode(
        name="Y",
        description="Y",
        inputs=[AgentData(name="altmid", parameters={}, description="altmid")],
        outputs=[AgentData(name="OUT", parameters={}, description="out")],
    )
    node_set = AgentGraphNodeSet(nodes=[node_a, node_b, node_x, node_y])
    desired_outputs = [AgentData(name="OUT", parameters={}, description="out")]

    planner = TaiatPathPlanner()
    # Find the first path
    path1 = planner.plan_path(node_set, desired_outputs)
    assert path1 is not None and len(path1) > 0, "Should find a path"

    # Determine which path was chosen first
    if "A" in path1 and "B" in path1:
        primary_nodes = {"A", "B"}
        secondary_nodes = {"X", "Y"}
    elif "X" in path1 and "Y" in path1:
        primary_nodes = {"X", "Y"}
        secondary_nodes = {"A", "B"}
    else:
        assert False, f"Unexpected path returned: {path1}"

    # Simulate failure of the primary path (remove those nodes)
    remaining_nodes = [n for n in [node_a, node_b, node_x, node_y] if n.name not in primary_nodes]
    node_set_failed = AgentGraphNodeSet(nodes=remaining_nodes)
    path2 = planner.plan_path(node_set_failed, desired_outputs)
    assert path2 is not None and len(path2) > 0, "Should find a secondary path"
    assert all(n in path2 for n in secondary_nodes), "Secondary path should use the alternative nodes"
    assert all(n not in path2 for n in primary_nodes), "Secondary path should not use failed nodes"
    print("test_secondary_path_with_agent_failure passed.")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TAIAT PATH PLANNER TESTS")
    print("=" * 60)

    test_simple_path()
    test_complex_path()
    test_multiple_outputs()
    test_invalid_output()
    test_convenience_functions()
    test_prolog_unit_tests()
    test_param_subset_match()
    test_param_conflict()
    test_param_empty()
    test_name_mismatch()
    test_parameter_constraints_for_intermediate_nodes()
    test_parameter_constraints_with_multiple_options()
    test_parameter_constraints_with_superset_matching()
    test_parameter_constraints_no_match()
    test_parameter_matching_debug()
    test_direct_parameter_matching()
    test_missing_input_error_reporting()
    test_secondary_path_with_agent_failure()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
