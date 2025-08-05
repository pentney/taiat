import unittest
import json
from unittest.mock import Mock
from typing import Any

from taiat.node_generator import NodeGenerator
from taiat.base import AgentGraphNode, AgentData


class TestNodeGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.generator = NodeGenerator(self.mock_llm)

    def test_generate_node_basic(self):
        """Test basic node generation with a simple function."""

        def test_function(state):
            return state

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "name": "test_processor",
                "description": "A test function that processes data",
                "inputs": [
                    {
                        "name": "input_data",
                        "description": "Input data to be processed",
                        "parameters": {},
                    }
                ],
                "outputs": [
                    {
                        "name": "processed_data",
                        "description": "The processed output data",
                        "parameters": {},
                    }
                ],
            }
        )
        self.mock_llm.invoke.return_value = mock_response

        # Generate node
        node = self.generator.generate_node(
            description="Process some input data and return processed results",
            function=test_function,
        )

        # Verify the node was created correctly
        self.assertIsInstance(node, AgentGraphNode)
        self.assertEqual(node.name, "test_processor")
        self.assertEqual(node.description, "A test function that processes data")
        self.assertEqual(node.function, test_function)

        # Verify inputs
        self.assertEqual(len(node.inputs), 1)
        self.assertEqual(node.inputs[0].name, "input_data")
        self.assertEqual(node.inputs[0].description, "Input data to be processed")

        # Verify outputs
        self.assertEqual(len(node.outputs), 1)
        self.assertEqual(node.outputs[0].name, "processed_data")
        self.assertEqual(node.outputs[0].description, "The processed output data")

        # Verify LLM was called with correct prompt
        self.mock_llm.invoke.assert_called_once()
        call_args = self.mock_llm.invoke.call_args[0][0]
        self.assertIn("Process some input data and return processed results", call_args)
        self.assertIn("test_function", call_args)

    def test_generate_node_with_parameters(self):
        """Test node generation with parameters in inputs/outputs."""

        def data_analyzer(state):
            return state

        # Mock LLM response with parameters
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "name": "data_analyzer",
                "description": "Analyzes data with configurable parameters",
                "inputs": [
                    {
                        "name": "raw_data",
                        "description": "Raw data to analyze",
                        "parameters": {"format": "csv", "encoding": "utf-8"},
                    }
                ],
                "outputs": [
                    {
                        "name": "analysis_results",
                        "description": "Analysis results with metrics",
                        "parameters": {"confidence": "0.95", "method": "statistical"},
                    }
                ],
            }
        )
        self.mock_llm.invoke.return_value = mock_response

        # Generate node
        node = self.generator.generate_node(
            description="Analyze raw data and produce statistical results",
            function=data_analyzer,
            function_name="data_analyzer",
        )

        # Verify parameters are correctly set
        self.assertEqual(
            node.inputs[0].parameters, {"format": "csv", "encoding": "utf-8"}
        )
        self.assertEqual(
            node.outputs[0].parameters, {"confidence": "0.95", "method": "statistical"}
        )

    def test_generate_node_multiple_inputs_outputs(self):
        """Test node generation with multiple inputs and outputs."""

        def multi_processor(state):
            return state

        # Mock LLM response with multiple inputs/outputs
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "name": "multi_processor",
                "description": "Processes multiple inputs and produces multiple outputs",
                "inputs": [
                    {
                        "name": "primary_data",
                        "description": "Primary dataset",
                        "parameters": {},
                    },
                    {
                        "name": "secondary_data",
                        "description": "Secondary dataset for comparison",
                        "parameters": {},
                    },
                ],
                "outputs": [
                    {
                        "name": "combined_results",
                        "description": "Combined analysis results",
                        "parameters": {},
                    },
                    {
                        "name": "summary_report",
                        "description": "Summary of the analysis",
                        "parameters": {"format": "text"},
                    },
                ],
            }
        )
        self.mock_llm.invoke.return_value = mock_response

        # Generate node
        node = self.generator.generate_node(
            description="Combine and analyze multiple datasets to produce results and summary",
            function=multi_processor,
        )

        # Verify multiple inputs and outputs
        self.assertEqual(len(node.inputs), 2)
        self.assertEqual(len(node.outputs), 2)
        self.assertEqual(node.inputs[0].name, "primary_data")
        self.assertEqual(node.inputs[1].name, "secondary_data")
        self.assertEqual(node.outputs[0].name, "combined_results")
        self.assertEqual(node.outputs[1].name, "summary_report")

    def test_generate_node_invalid_json_response(self):
        """Test handling of invalid JSON response from LLM."""

        def test_function(state):
            return state

        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        self.mock_llm.invoke.return_value = mock_response

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.generator.generate_node(
                description="Test function", function=test_function
            )

        self.assertIn("Failed to parse LLM response as JSON", str(context.exception))

    def test_generate_node_custom_function_name(self):
        """Test node generation with custom function name."""

        def my_function(state):
            return state

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "name": "custom_processor",
                "description": "Custom processing function",
                "inputs": [],
                "outputs": [],
            }
        )
        self.mock_llm.invoke.return_value = mock_response

        # Generate node with custom function name
        node = self.generator.generate_node(
            description="Custom processing task",
            function=my_function,
            function_name="custom_processor",
        )

        # Verify custom function name was used in prompt
        call_args = self.mock_llm.invoke.call_args[0][0]
        self.assertIn("custom_processor", call_args)
        self.assertNotIn("my_function", call_args)


if __name__ == "__main__":
    unittest.main()
