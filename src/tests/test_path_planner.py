"""
Test script for the Path Planner.

This script demonstrates how to use the path planner to determine
execution paths for Taiat queries.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData
from haskell.path_planner_interface import plan_path, PathPlanner


def test_ml_workflow_model_selection():
    """
    Test that the planner correctly selects only the random_forest model
    when a generic 'model' input is required, and doesn't include logistic_regression.

    This test mimics the ml_workflow.py scenario where:
    1. We have multiple model producers (logistic_regression, random_forest, etc.)
    2. We request a specific model (random_forest) and a generic model_report
    3. The predict_and_generate_report node requires a generic 'model' input
    4. The planner should select only random_forest, not logistic_regression
    """
    from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData

    # Define dummy functions for the test nodes
    def dummy_function(state):
        return state

    # Create the ML workflow graph structure (mimicking ml_agents.py)
    load_dataset_node = AgentGraphNode(
        name="load_dataset",
        description="Load the dataset",
        function=dummy_function,
        inputs=[
            AgentData(name="dataset_name", parameters={}, description="", data=None)
        ],
        outputs=[AgentData(name="dataset", parameters={}, description="", data=None)],
    )

    logistic_regression_node = AgentGraphNode(
        name="logistic_regression",
        description="Train a logistic regression model",
        function=dummy_function,
        inputs=[AgentData(name="dataset", parameters={}, description="", data=None)],
        outputs=[
            AgentData(
                name="model",
                parameters={"model_type": "logistic_regression"},
                description="",
                data=None,
            ),
            AgentData(name="model_params", parameters={}, description="", data=None),
        ],
    )

    random_forest_node = AgentGraphNode(
        name="random_forest",
        description="Train a random forest model",
        function=dummy_function,
        inputs=[AgentData(name="dataset", parameters={}, description="", data=None)],
        outputs=[
            AgentData(
                name="model",
                parameters={"model_type": "random_forest"},
                description="",
                data=None,
            ),
        ],
    )

    nearest_neighbors_node = AgentGraphNode(
        name="nearest_neighbors",
        description="Train a nearest neighbors model",
        function=dummy_function,
        inputs=[AgentData(name="dataset", parameters={}, description="", data=None)],
        outputs=[
            AgentData(
                name="model",
                parameters={"model_type": "nearest_neighbors"},
                description="",
                data=None,
            ),
        ],
    )

    clustering_node = AgentGraphNode(
        name="clustering",
        description="Train a clustering model",
        function=dummy_function,
        inputs=[AgentData(name="dataset", parameters={}, description="", data=None)],
        outputs=[
            AgentData(
                name="model",
                parameters={"model_type": "clustering"},
                description="",
                data=None,
            ),
        ],
    )

    predict_and_generate_report_node = AgentGraphNode(
        name="predict_and_generate_report",
        description="Make a prediction and generate a report",
        function=dummy_function,
        inputs=[
            AgentData(name="model", parameters={}, description="", data=None)
        ],  # Generic model input
        outputs=[
            AgentData(name="model_preds", parameters={}, description="", data=None),
            AgentData(name="model_report", parameters={}, description="", data=None),
        ],
    )

    results_analysis_node = AgentGraphNode(
        name="results_analysis",
        description="Analyze the results",
        function=dummy_function,
        inputs=[
            AgentData(name="dataset_name", parameters={}, description="", data=None),
            AgentData(name="model_report", parameters={}, description="", data=None),
        ],
        outputs=[AgentData(name="summary", parameters={}, description="", data=None)],
    )

    # Create the node set (mimicking agent_roster)
    node_set = AgentGraphNodeSet(
        nodes=[
            load_dataset_node,
            logistic_regression_node,
            random_forest_node,
            nearest_neighbors_node,
            clustering_node,
            predict_and_generate_report_node,
            results_analysis_node,
        ]
    )

    # Define desired outputs (mimicking the ml_workflow.py scenario)
    # We want a specific random_forest model and a generic model_report
    desired_outputs = [
        AgentData(
            name="model",
            parameters={"model_type": "random_forest"},
            description="",
            data=None,
        ),
        AgentData(name="model_report", parameters={}, description="", data=None),
    ]

    # Plan execution path using planner
    # Include external inputs that are needed (dataset_name)
    external_inputs = [
        AgentData(name="dataset_name", parameters={}, description="", data=None)
    ]
    execution_path = plan_path(node_set, desired_outputs, external_inputs)

    print(f"ML workflow model selection test execution path: {execution_path}")

    # Verify the expected behavior:
    # 1. Should include dataset_name_external (external input)
    # 2. Should include load_dataset
    # 3. Should include random_forest (the specific model requested)
    # 4. Should include predict_and_generate_report (to generate model_report)
    # 5. Should NOT include logistic_regression, nearest_neighbors, or clustering
    # 6. Should NOT include results_analysis (not requested)

    expected_nodes = {
        "dataset_name_external",
        "load_dataset",
        "random_forest",
        "predict_and_generate_report",
    }
    unexpected_nodes = {
        "logistic_regression",
        "nearest_neighbors",
        "clustering",
        "results_analysis",
    }

    if execution_path is None:
        print("❌ ML workflow model selection test failed - no execution path found")
        assert False, "Execution path should be found"

    execution_path_set = set(execution_path)

    # Check that all expected nodes are present
    missing_nodes = expected_nodes - execution_path_set
    if missing_nodes:
        print(
            f"❌ ML workflow model selection test failed - missing nodes: {missing_nodes}"
        )
        assert False, f"Missing expected nodes: {missing_nodes}"

    # Check that no unexpected nodes are present
    extra_nodes = unexpected_nodes & execution_path_set
    if extra_nodes:
        print(
            f"❌ ML workflow model selection test failed - unexpected nodes included: {extra_nodes}"
        )
        assert False, f"Unexpected nodes included: {extra_nodes}"

    print("✓ ML workflow model selection test passed")


def test_parameter_constraint():
    """Test that input parameter constraints are properly enforced in planner."""
    print("Testing parameter constraint enforcement...")

    # Create a producer that outputs X with parameters {"A": "B", "C": "D"}
    producer = AgentGraphNode(
        name="producer",
        description="Produces output X with specific parameters",
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "B", "C": "D"},
                description="Output X with parameters A:B and C:D",
                data=None,
            )
        ],
    )

    # Create a consumer that requires input X with parameter {"A": "B"}
    # This should match because the output has A:B
    consumer_correct = AgentGraphNode(
        name="consumer_correct",
        description="Consumes input X with parameter A:B (should match)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},
                description="Input X requiring parameter A:B",
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

    # Create a consumer that requires input X with parameter {"A": "C"}
    # This should NOT match because the output has A:B, not A:C
    consumer_incorrect = AgentGraphNode(
        name="consumer_incorrect",
        description="Consumes input X with parameter A:C (should NOT match)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "C"},
                description="Input X requiring parameter A:C",
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

    # Test case 1: Correct parameter constraint (should succeed)
    print("\n--- Test Case 1: Correct Parameter Constraint ---")
    node_set_1 = AgentGraphNodeSet(nodes=[producer, consumer_correct])
    desired_outputs_1 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    try:
        planner = PathPlanner()
        if not planner.available:
            print("❌ Planner not available")
            return

        path_1 = planner.plan_path(node_set_1, desired_outputs_1)
        print(f"Path result: {path_1}")

        if path_1 == ["producer", "consumer_correct"]:
            print("✅ Correct parameter constraint test PASSED")
        else:
            print(f"❌ Correct parameter constraint test FAILED: {path_1}")

    except Exception as e:
        print(f"❌ Error in correct parameter test: {e}")

    # Test case 2: Incorrect parameter constraint (should fail)
    print("\n--- Test Case 2: Incorrect Parameter Constraint ---")
    node_set_2 = AgentGraphNodeSet(nodes=[producer, consumer_incorrect])
    desired_outputs_2 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    try:
        path_2 = planner.plan_path(node_set_2, desired_outputs_2)
        print(f"Path result: {path_2}")

        if path_2 == []:
            print("✅ Incorrect parameter constraint test PASSED (correctly rejected)")
        else:
            print(f"❌ Incorrect parameter constraint test FAILED: {path_2}")

    except Exception as e:
        print(f"❌ Error in incorrect parameter test: {e}")

    # Test case 3: Additional parameters in output (should succeed)
    print("\n--- Test Case 3: Additional Parameters in Output ---")
    # Create a consumer that requires input X with parameter {"A": "B"}
    # The producer outputs {"A": "B", "C": "D"}, so this should match
    consumer_additional = AgentGraphNode(
        name="consumer_additional",
        description="Consumes input X with parameter A:B (output has additional C:D)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},
                description="Input X requiring parameter A:B",
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

    node_set_3 = AgentGraphNodeSet(nodes=[producer, consumer_additional])
    desired_outputs_3 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]

    try:
        path_3 = planner.plan_path(node_set_3, desired_outputs_3)
        print(f"Path result: {path_3}")

        if path_3 == ["producer", "consumer_additional"]:
            print("✅ Additional parameters test PASSED")
        else:
            print(f"❌ Additional parameters test FAILED: {path_3}")

    except Exception as e:
        print(f"❌ Error in additional parameters test: {e}")


def test_minimal_path_planning():
    """Test basic path planning without parameter constraints."""
    print("Testing minimal path planning...")

    # Create a simple node with no parameter constraints
    node = AgentGraphNode(
        name="simple_node",
        description="Simple node",
        inputs=[],
        outputs=[
            AgentData(
                name="output",
                parameters={},  # No parameters
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="output",
            parameters={},  # No parameters
            description="",
            data=None,
        )
    ]

    print(f"Node: {node.name}")
    print(f"Node outputs: {node.outputs}")
    print(f"Desired outputs: {desired_outputs}")

    try:
        planner = PathPlanner()
        if not planner.available:
            print("❌ Planner not available")
            return

        print("✅ Planner available")

        # Test the plan_path function
        path = planner.plan_path(node_set, desired_outputs)
        print(f"Path result: {path}")

        # Test validation
        validation_result = planner.validate_outputs(node_set, desired_outputs)
        print(f"Validation result: {validation_result}")

        # Test available outputs
        available = planner.get_available_outputs(node_set)
        print(f"Available outputs: {available}")

        # Test circular dependencies
        has_circular = planner.has_circular_dependencies(node_set)
        print(f"Has circular dependencies: {has_circular}")

        if path == ["simple_node"]:
            print("✅ Minimal path planning test PASSED")
        else:
            print(f"❌ Minimal path planning test FAILED: {path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def test_agent_data_matching():
    """Test agent data matching functionality."""
    print("Testing agent data matching...")

    # Test basic agent_data_match functionality
    data1 = AgentData(name="test", parameters={}, description="test", data=None)
    data2 = AgentData(name="test", parameters={}, description="test", data=None)

    # Test through the planner's internal logic
    node = AgentGraphNode(
        name="test_node", description="test", inputs=[], outputs=[data2]
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired = [data1]

    try:
        planner = PathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return

        path = planner.plan_path(node_set, desired)
        print(f"Basic matching path: {path}")

        if path == ["test_node"]:
            print("✅ Basic agent data matching test PASSED")
        else:
            print(f"❌ Basic agent data matching test FAILED: {path}")

        # Test parameter subset matching
        input_data = AgentData(
            name="model",
            parameters={"type": "logistic_regression"},
            description="input",
            data=None,
        )
        output_data = AgentData(
            name="model",
            parameters={"type": "logistic_regression", "version": "v1"},
            description="output",
            data=None,
        )

        node2 = AgentGraphNode(
            name="model_node", description="test", inputs=[], outputs=[output_data]
        )
        node_set2 = AgentGraphNodeSet(nodes=[node2])
        desired2 = [input_data]

        path2 = planner.plan_path(node_set2, desired2)
        print(f"Parameter subset matching path: {path2}")

        if path2 == ["model_node"]:
            print("✅ Parameter subset matching test PASSED")
        else:
            print(f"❌ Parameter subset matching test FAILED: {path2}")

        # Test parameter conflict rejection
        input_data3 = AgentData(
            name="model",
            parameters={"type": "logistic_regression"},
            description="input",
            data=None,
        )
        output_data3 = AgentData(
            name="model",
            parameters={"type": "neural_network", "version": "v1"},
            description="output",
            data=None,
        )

        node3 = AgentGraphNode(
            name="model_node3", description="test", inputs=[], outputs=[output_data3]
        )
        node_set3 = AgentGraphNodeSet(nodes=[node3])
        desired3 = [input_data3]

        path3 = planner.plan_path(node_set3, desired3)
        print(f"Parameter conflict path: {path3}")

        if path3 == []:  # Should fail due to parameter conflict
            print("✅ Parameter conflict rejection test PASSED")
        else:
            print(f"❌ Parameter conflict rejection test FAILED: {path3}")

    except Exception as e:
        print(f"❌ Error in agent data matching test: {e}")
        import traceback

        traceback.print_exc()


def test_simple_pipeline():
    """Test a simple linear pipeline."""
    print("Testing simple pipeline...")

    # Create a simple data processing pipeline: data_loader -> preprocessor -> analyzer -> visualizer
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

    desired_outputs = [
        AgentData(
            name="visualizations",
            parameters={},
            description="Generated visualizations",
            data=None,
        )
    ]

    try:
        planner = PathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return

        execution_path = planner.plan_path(node_set, desired_outputs)
        print(f"Simple pipeline execution path: {execution_path}")

        if execution_path:
            expected_path = ["data_loader", "preprocessor", "analyzer", "visualizer"]
            if execution_path == expected_path:
                print("✅ Simple pipeline test PASSED")
            else:
                print(f"⚠️  Path differs from expected: {expected_path}")
        else:
            print("❌ Simple pipeline test FAILED")

    except Exception as e:
        print(f"❌ Error in simple pipeline test: {e}")


def main():
    """Run all Haskell path planner tests."""
    print("=" * 60)
    print("TAIAT HASKELL PATH PLANNER TESTS")
    print("=" * 60)

    # Run Haskell-specific tests
    test_ml_workflow_model_selection()
    test_parameter_constraint()
    test_minimal_path_planning()
    test_agent_data_matching()
    test_simple_pipeline()

    print("\n" + "=" * 60)
    print("All Haskell path planner tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
