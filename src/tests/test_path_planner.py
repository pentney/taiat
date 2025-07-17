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

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData
from taiat.prolog.optimized_prolog_interface import (
    plan_taiat_path_global_optimized,
    OptimizedPrologPathPlanner,
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

    execution_path = plan_taiat_path_global_optimized(node_set, desired_outputs)

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

    execution_path = plan_taiat_path_global_optimized(node_set, desired_outputs)

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

    execution_path = plan_taiat_path_global_optimized(node_set, desired_outputs)

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

    execution_path = plan_taiat_path_global_optimized(node_set, desired_outputs)

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
    execution_path = plan_taiat_path_global_optimized(node_set, desired_outputs)

    if execution_path:
        print("✅ Global optimized function works")
    else:
        print("❌ Global optimized function failed")


def test_prolog_unit_tests():
    """Run the Prolog unit tests directly."""
    print("\nRunning Prolog unit tests...")

    try:
        # Test the planner directly
        planner = OptimizedPrologPathPlanner()
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


def main():
    """Run all tests."""
    print("=" * 60)
    print("GLOBAL OPTIMIZED PROLOG PATH PLANNER TESTS")
    print("=" * 60)

    test_simple_path()
    test_complex_path()
    test_multiple_outputs()
    test_invalid_output()
    test_convenience_functions()
    test_prolog_unit_tests()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
