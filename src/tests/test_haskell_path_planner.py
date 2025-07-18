"""
Test script for the Haskell Path Planner.

This script tests the Haskell implementation of the Taiat path planner
using the same test cases as the Prolog version for comparison.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData
from haskell.haskell_interface import HaskellPathPlanner


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

    # Final report node
    report_generator = AgentGraphNode(
        name="report_generator",
        description="Generate final report",
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
    """Test simple path planning with Haskell engine."""
    print("Testing simple path planning with Haskell...")

    node_set = create_example_node_set()
    desired_outputs = [
        AgentData(
            name="visualizations",
            parameters={},
            description="Generated visualizations",
            data=None,
        )
    ]

    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return False

        execution_path = planner.plan_path(node_set, desired_outputs)

        if execution_path:
            print(f"✅ Simple path planning successful: {execution_path}")
            expected_path = ["data_loader", "preprocessor", "analyzer", "visualizer"]
            if execution_path == expected_path:
                print("✅ Path matches expected order")
                return True
            else:
                print(f"⚠️  Path differs from expected: {expected_path}")
                return False
        else:
            print("❌ Simple path planning failed")
            return False
    except Exception as e:
        print(f"❌ Haskell simple path planning error: {e}")
        return False


def test_complex_path():
    """Test complex path planning with Haskell engine."""
    print("\nTesting complex path planning with Haskell...")

    node_set = create_complex_example_node_set()
    desired_outputs = [
        AgentData(
            name="final_report",
            parameters={},
            description="Final comprehensive report",
            data=None,
        )
    ]

    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return False

        execution_path = planner.plan_path(node_set, desired_outputs)

        if execution_path:
            print(f"✅ Complex path planning successful: {execution_path}")
            print("✅ Path planning completed")
            return True
        else:
            print("❌ Complex path planning failed")
            return False
    except Exception as e:
        print(f"❌ Haskell complex path planning error: {e}")
        return False


def test_multiple_outputs():
    """Test planning for multiple outputs with Haskell engine."""
    print("\nTesting multiple outputs planning with Haskell...")

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

    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return False

        execution_path = planner.plan_path(node_set, desired_outputs)

        if execution_path:
            print(f"✅ Multiple outputs planning successful: {execution_path}")
            print("✅ Path planning completed")
            return True
        else:
            print("❌ Multiple outputs planning failed")
            return False
    except Exception as e:
        print(f"❌ Haskell multiple outputs planning error: {e}")
        return False


def test_invalid_output():
    """Test planning with invalid output using Haskell engine."""
    print("\nTesting invalid output with Haskell...")

    node_set = create_example_node_set()
    desired_outputs = [
        AgentData(
            name="nonexistent_output",
            parameters={},
            description="Output that doesn't exist",
            data=None,
        )
    ]

    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return False

        execution_path = planner.plan_path(node_set, desired_outputs)

        if execution_path is None or len(execution_path) == 0:
            print("✅ Correctly handled invalid output")
            return True
        else:
            print("❌ Should have failed for invalid output")
            return False
    except Exception as e:
        print(f"❌ Haskell invalid output test error: {e}")
        return False


def test_parameter_matching():
    """Test parameter matching with Haskell engine."""
    print("\nTesting parameter matching with Haskell...")

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

    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return False

        path = planner.plan_path(node_set, desired_outputs)
        if path == ["model_node"]:
            print("✅ Parameter matching test passed")
            return True
        else:
            print(f"❌ Parameter matching test failed: {path}")
            return False
    except Exception as e:
        print(f"❌ Haskell parameter matching test error: {e}")
        return False


def run_all_tests():
    """Run all Haskell path planner tests."""
    print("Haskell Path Planner Tests")
    print("=" * 50)
    
    tests = [
        ("Simple Path", test_simple_path),
        ("Complex Path", test_complex_path),
        ("Multiple Outputs", test_multiple_outputs),
        ("Invalid Output", test_invalid_output),
        ("Parameter Matching", test_parameter_matching),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All Haskell path planner tests passed!")
        return True
    else:
        print("❌ Some Haskell path planner tests failed!")
        return False


if __name__ == "__main__":
    run_all_tests() 