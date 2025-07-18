"""
Comprehensive Haskell Unit Test Suite for Taiat Path Planner

This test suite covers all the test cases from path_planner_test.pl,
ensuring the Haskell implementation matches the Prolog behavior.
"""

import sys
import os
from pathlib import Path
import pytest

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
from haskell.haskell_interface import HaskellPathPlanner


class TestHaskellPathPlanner:
    """Test suite for Haskell Path Planner implementation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.planner = HaskellPathPlanner()
        if not self.planner.available:
            pytest.skip("Haskell planner not available")
        
        # Test data matching the Prolog test cases
        self.test_node_set = AgentGraphNodeSet(nodes=[
            AgentGraphNode(
                name="data_loader",
                description="Load data from source",
                inputs=[],
                outputs=[
                    AgentData("raw_data", {}, "Raw data from source", None)
                ]
            ),
            AgentGraphNode(
                name="preprocessor",
                description="Preprocess the data",
                inputs=[
                    AgentData("raw_data", {}, "Raw data from source", None)
                ],
                outputs=[
                    AgentData("processed_data", {}, "Preprocessed data", None)
                ]
            ),
            AgentGraphNode(
                name="analyzer",
                description="Analyze the data",
                inputs=[
                    AgentData("processed_data", {}, "Preprocessed data", None)
                ],
                outputs=[
                    AgentData("analysis_results", {}, "Analysis results", None)
                ]
            ),
            AgentGraphNode(
                name="visualizer",
                description="Create visualizations",
                inputs=[
                    AgentData("analysis_results", {}, "Analysis results", None)
                ],
                outputs=[
                    AgentData("visualizations", {}, "Generated visualizations", None)
                ]
            )
        ])
        
        self.test_desired_outputs = [
            AgentData("visualizations", {}, "Generated visualizations", None)
        ]
        
        # Complex test data with multiple paths
        self.test_complex_node_set = AgentGraphNodeSet(nodes=[
            AgentGraphNode(
                name="csv_loader",
                description="Load CSV data",
                inputs=[],
                outputs=[
                    AgentData("csv_data", {}, "CSV data", None)
                ]
            ),
            AgentGraphNode(
                name="json_loader",
                description="Load JSON data",
                inputs=[],
                outputs=[
                    AgentData("json_data", {}, "JSON data", None)
                ]
            ),
            AgentGraphNode(
                name="csv_processor",
                description="Process CSV data",
                inputs=[
                    AgentData("csv_data", {}, "CSV data", None)
                ],
                outputs=[
                    AgentData("processed_csv", {}, "Processed CSV data", None)
                ]
            ),
            AgentGraphNode(
                name="json_processor",
                description="Process JSON data",
                inputs=[
                    AgentData("json_data", {}, "JSON data", None)
                ],
                outputs=[
                    AgentData("processed_json", {}, "Processed JSON data", None)
                ]
            ),
            AgentGraphNode(
                name="data_merger",
                description="Merge processed data",
                inputs=[
                    AgentData("processed_csv", {}, "Processed CSV data", None),
                    AgentData("processed_json", {}, "Processed JSON data", None)
                ],
                outputs=[
                    AgentData("merged_data", {}, "Merged data", None)
                ]
            ),
            AgentGraphNode(
                name="analyzer",
                description="Analyze merged data",
                inputs=[
                    AgentData("merged_data", {}, "Merged data", None)
                ],
                outputs=[
                    AgentData("analysis_results", {}, "Analysis results", None)
                ]
            )
        ])
        
        self.test_complex_desired_outputs = [
            AgentData("analysis_results", {}, "Analysis results", None)
        ]
    
    def test_agent_data_name(self):
        """Test agent_data_name functionality."""
        agent_data = AgentData("test", {}, "test", None)
        assert agent_data.name == "test"
    
    def test_agent_data_match(self):
        """Test basic agent_data_match functionality."""
        data1 = AgentData("test", {}, "test", None)
        data2 = AgentData("test", {}, "test", None)
        
        # Test through the planner's internal logic
        node = AgentGraphNode("test_node", "test", [], [data2])
        node_set = AgentGraphNodeSet([node])
        desired = [data1]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == ["test_node"]
    
    def test_agent_data_match_params_subset(self):
        """Test parameter subset matching."""
        input_data = AgentData("model", {"type": "logistic_regression"}, "input", None)
        output_data = AgentData("model", {"type": "logistic_regression", "version": "v1"}, "output", None)
        
        node = AgentGraphNode("model_node", "test", [], [output_data])
        node_set = AgentGraphNodeSet([node])
        desired = [input_data]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == ["model_node"]
    
    def test_agent_data_match_params_conflict(self):
        """Test parameter conflict rejection."""
        input_data = AgentData("model", {"type": "logistic_regression"}, "input", None)
        output_data = AgentData("model", {"type": "neural_network", "version": "v1"}, "output", None)
        
        node = AgentGraphNode("model_node", "test", [], [output_data])
        node_set = AgentGraphNodeSet([node])
        desired = [input_data]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == []  # Should fail due to parameter conflict
    
    def test_agent_data_match_params_empty(self):
        """Test matching with empty input parameters."""
        input_data = AgentData("model", {}, "input", None)
        output_data = AgentData("model", {"type": "neural_network"}, "output", None)
        
        node = AgentGraphNode("model_node", "test", [], [output_data])
        node_set = AgentGraphNodeSet([node])
        desired = [input_data]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == ["model_node"]  # Should match (empty params are subset of any params)
    
    def test_agent_data_match_name_mismatch(self):
        """Test name mismatch rejection."""
        input_data = AgentData("modelA", {"type": "logistic_regression"}, "input", None)
        output_data = AgentData("modelB", {"type": "logistic_regression"}, "output", None)
        
        node = AgentGraphNode("model_node", "test", [], [output_data])
        node_set = AgentGraphNodeSet([node])
        desired = [input_data]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == []  # Should fail due to name mismatch
    
    def test_node_produces_output(self):
        """Test node_produces_output functionality."""
        # Find the data_loader node
        data_loader_node = None
        for node in self.test_node_set.nodes:
            if node.name == "data_loader":
                data_loader_node = node
                break
        
        assert data_loader_node is not None
        assert data_loader_node.name == "data_loader"
        
        # Check if it produces raw_data
        raw_data = AgentData("raw_data", {}, "Raw data from source", None)
        node_set = AgentGraphNodeSet([data_loader_node])
        desired = [raw_data]
        
        path = self.planner.plan_path(node_set, desired)
        assert path == ["data_loader"]
    
    def test_nodes_producing_output_name(self):
        """Test nodes_producing_output_name functionality."""
        # Test that data_loader produces raw_data
        raw_data = AgentData("raw_data", {}, "Raw data from source", None)
        path = self.planner.plan_path(self.test_node_set, [raw_data])
        assert "data_loader" in path
    
    def test_node_dependencies(self):
        """Test node_dependencies functionality."""
        # Test that preprocessor depends on data_loader
        path = self.planner.plan_path(self.test_node_set, self.test_desired_outputs)
        assert path == ["data_loader", "preprocessor", "analyzer", "visualizer"]
        
        # Verify dependency order
        data_loader_idx = path.index("data_loader")
        preprocessor_idx = path.index("preprocessor")
        assert data_loader_idx < preprocessor_idx
    
    def test_node_ready(self):
        """Test node_ready functionality."""
        # Test that preprocessor is ready when data_loader is executed
        path = self.planner.plan_path(self.test_node_set, self.test_desired_outputs)
        assert path == ["data_loader", "preprocessor", "analyzer", "visualizer"]
        
        # Verify that dependencies are satisfied in order
        assert path[0] == "data_loader"  # No dependencies
        assert path[1] == "preprocessor"  # Depends on data_loader
        assert path[2] == "analyzer"      # Depends on preprocessor
        assert path[3] == "visualizer"    # Depends on analyzer
    
    def test_required_nodes(self):
        """Test required_nodes functionality."""
        path = self.planner.plan_path(self.test_node_set, self.test_desired_outputs)
        
        # Should include all 4 nodes
        assert len(path) == 4
        assert "data_loader" in path
        assert "preprocessor" in path
        assert "analyzer" in path
        assert "visualizer" in path
    
    def test_topological_sort(self):
        """Test topological_sort functionality."""
        path = self.planner.plan_path(self.test_node_set, self.test_desired_outputs)
        
        # Verify topological order
        assert path == ["data_loader", "preprocessor", "analyzer", "visualizer"]
        
        # Verify dependencies are satisfied
        assert path[0] == "data_loader"  # No dependencies
        assert path[1] == "preprocessor"  # Depends on data_loader
        assert path[2] == "analyzer"      # Depends on preprocessor
        assert path[3] == "visualizer"    # Depends on analyzer
    
    def test_plan_execution_path(self):
        """Test plan_execution_path functionality."""
        path = self.planner.plan_path(self.test_node_set, self.test_desired_outputs)
        assert path == ["data_loader", "preprocessor", "analyzer", "visualizer"]
    
    def test_validate_outputs(self):
        """Test validate_outputs functionality."""
        # Test valid outputs
        result = self.planner.validate_outputs(self.test_node_set, self.test_desired_outputs)
        assert result is True
        
        # Test invalid outputs
        invalid_outputs = [AgentData("nonexistent", {}, "Nonexistent output", None)]
        result = self.planner.validate_outputs(self.test_node_set, invalid_outputs)
        assert result is False
    
    def test_available_outputs(self):
        """Test available_outputs functionality."""
        available = self.planner.get_available_outputs(self.test_node_set)
        
        # Check that all expected outputs are available
        output_names = [output.name for output in available]
        assert "raw_data" in output_names
        assert "processed_data" in output_names
        assert "analysis_results" in output_names
        assert "visualizations" in output_names
    
    def test_complex_path_planning(self):
        """Test complex path planning with multiple paths."""
        path = self.planner.plan_path(self.test_complex_node_set, self.test_complex_desired_outputs)
        
        # Should include all required nodes
        assert "csv_loader" in path
        assert "json_loader" in path
        assert "csv_processor" in path
        assert "json_processor" in path
        assert "data_merger" in path
        assert "analyzer" in path
        
        # Verify dependency order
        csv_loader_idx = path.index("csv_loader")
        csv_processor_idx = path.index("csv_processor")
        assert csv_loader_idx < csv_processor_idx
        
        json_loader_idx = path.index("json_loader")
        json_processor_idx = path.index("json_processor")
        assert json_loader_idx < json_processor_idx
        
        data_merger_idx = path.index("data_merger")
        assert csv_processor_idx < data_merger_idx
        assert json_processor_idx < data_merger_idx
        
        analyzer_idx = path.index("analyzer")
        assert data_merger_idx < analyzer_idx
    
    def test_invalid_output(self):
        """Test invalid output handling."""
        invalid_outputs = [AgentData("nonexistent", {}, "Nonexistent output", None)]
        result = self.planner.validate_outputs(self.test_node_set, invalid_outputs)
        assert result is False
    
    def test_remove_duplicates(self):
        """Test remove_duplicates functionality."""
        # Test through the planner's behavior with duplicate nodes
        # Create a node set with duplicate outputs
        duplicate_node_set = AgentGraphNodeSet(nodes=[
            AgentGraphNode("node1", "test", [], [AgentData("output", {}, "test", None)]),
            AgentGraphNode("node2", "test", [], [AgentData("output", {}, "test", None)])
        ])
        
        desired = [AgentData("output", {}, "test", None)]
        path = self.planner.plan_path(duplicate_node_set, desired)
        
        # Should return a path with one of the nodes (not both)
        assert len(path) == 1
        assert path[0] in ["node1", "node2"]
    
    def test_circular_dependencies(self):
        """Test circular dependency detection."""
        # Create a circular dependency: A -> B -> C -> A
        node_a = AgentGraphNode("A", "test", [AgentData("C_output", {}, "test", None)], 
                               [AgentData("A_output", {}, "test", None)])
        node_b = AgentGraphNode("B", "test", [AgentData("A_output", {}, "test", None)], 
                               [AgentData("B_output", {}, "test", None)])
        node_c = AgentGraphNode("C", "test", [AgentData("B_output", {}, "test", None)], 
                               [AgentData("C_output", {}, "test", None)])
        
        circular_node_set = AgentGraphNodeSet([node_a, node_b, node_c])
        desired = [AgentData("A_output", {}, "test", None)]
        
        # Should detect circular dependency and return empty path
        path = self.planner.plan_path(circular_node_set, desired)
        assert path == []
    
    def test_multiple_outputs(self):
        """Test planning for multiple outputs."""
        # Test planning for both processed_data and visualizations
        multiple_desired = [
            AgentData("processed_data", {}, "Preprocessed data", None),
            AgentData("visualizations", {}, "Generated visualizations", None)
        ]
        
        path = self.planner.plan_path(self.test_node_set, multiple_desired)
        
        # Should include all nodes needed for both outputs
        assert "data_loader" in path
        assert "preprocessor" in path
        assert "analyzer" in path
        assert "visualizer" in path
    
    def test_external_inputs(self):
        """Test handling of external inputs (not produced by any node)."""
        # Create a node that requires an external input
        external_node = AgentGraphNode(
            "external_consumer",
            "test",
            [AgentData("external_input", {}, "External input", None)],
            [AgentData("final_output", {}, "Final output", None)]
        )
        
        node_set = AgentGraphNodeSet([external_node])
        desired = [AgentData("final_output", {}, "Final output", None)]
        
        # Should handle external input gracefully
        path = self.planner.plan_path(node_set, desired)
        assert path == ["external_consumer"]
    
    def test_empty_node_set(self):
        """Test behavior with empty node set."""
        empty_node_set = AgentGraphNodeSet([])
        desired = [AgentData("test", {}, "test", None)]
        
        path = self.planner.plan_path(empty_node_set, desired)
        assert path == []
    
    def test_empty_desired_outputs(self):
        """Test behavior with empty desired outputs."""
        path = self.planner.plan_path(self.test_node_set, [])
        assert path == []


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 