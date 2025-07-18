"""
Python interface for the Haskell path planner.

This module provides a Python interface to the Haskell path planning implementation,
allowing Python code to use the high-performance Haskell planner for determining
execution paths in Taiat graphs.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import Taiat types
try:
    from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet
except ImportError:
    # Fallback for testing
    class AgentData:
        def __init__(
            self,
            name: str,
            parameters: Dict[str, str],
            description: str = "",
            data: Any = None,
        ):
            self.name = name
            self.parameters = parameters
            self.description = description
            self.data = data

        def __repr__(self):
            return f"AgentData(name='{self.name}', parameters={self.parameters})"

    class AgentGraphNode:
        def __init__(
            self,
            name: str,
            description: str,
            inputs: List[AgentData],
            outputs: List[AgentData],
            function=None,
        ):
            self.name = name
            self.description = description
            self.inputs = inputs
            self.outputs = outputs
            self.function = function

    class AgentGraphNodeSet:
        def __init__(self, nodes: List[AgentGraphNode]):
            self.nodes = nodes


class PathPlanner:
    """
    Python interface to the Haskell path planner.

    This class provides methods to interact with the compiled Haskell binary
    for path planning operations.
    """

    def __init__(self, haskell_binary_path: Optional[str] = None):
        """
        Initialize the PathPlanner.

        Args:
            haskell_binary_path: Optional path to the Haskell binary. If not provided,
                                will attempt to find or build it automatically.
        """
        self.haskell_binary_path = haskell_binary_path
        self.available = self._check_availability()

    def _find_haskell_binary(self) -> str:
        """Find the Haskell binary in the expected location."""
        # Look for the binary in the haskell directory
        haskell_dir = Path(__file__).parent
        binary_name = "taiat-path-planner"

        # Check if binary exists in haskell directory
        binary_path = haskell_dir / binary_name
        if binary_path.exists() and os.access(binary_path, os.X_OK):
            return str(binary_path)

        # Look in dist-newstyle directory (Cabal build output)
        dist_dir = haskell_dir / "dist-newstyle"
        if dist_dir.exists():
            # Find the binary in any subdirectory
            for root, dirs, files in os.walk(dist_dir):
                for file in files:
                    if file == binary_name:
                        binary_path = Path(root) / file
                        if os.access(binary_path, os.X_OK):
                            return str(binary_path)

        # If not found, try to build it
        self._build_haskell_binary(haskell_dir)

        # Check again after building
        binary_path = haskell_dir / binary_name
        if binary_path.exists() and os.access(binary_path, os.X_OK):
            return str(binary_path)

        raise FileNotFoundError(
            f"Could not find or build Haskell binary: {binary_name}"
        )

    def _build_haskell_binary(self, directory: Path) -> None:
        """Build the Haskell binary using Cabal."""
        try:
            # Run cabal build
            result = subprocess.run(
                ["cabal", "build"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build Haskell binary: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Build timed out")
        except FileNotFoundError:
            raise RuntimeError(
                "Cabal not found. Please install Cabal to build the Haskell binary."
            )

    def _check_availability(self) -> bool:
        """Check if the Haskell binary is available and working."""
        try:
            if self.haskell_binary_path is None:
                self.haskell_binary_path = self._find_haskell_binary()

            # Test the binary with a simple request
            test_input = {
                "function": "hasCircularDependencies",
                "input": {"agentGraphNodeSetNodes": []},
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(test_input, f)
                input_file = f.name

            try:
                result = subprocess.run(
                    [self.haskell_binary_path, input_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    return True
                else:
                    return False

            finally:
                os.unlink(input_file)

        except Exception:
            return False

    def _convert_agent_data_to_json(self, agent_data: AgentData) -> Dict[str, Any]:
        """Convert AgentData to JSON-serializable format."""
        return {
            "agentDataName": agent_data.name,
            "agentDataParameters": agent_data.parameters,
            "agentDataDescription": agent_data.description,
            "agentDataData": None,  # Always use None for consistency
        }

    def _convert_node_to_json(self, node) -> Dict[str, Any]:
        """Convert AgentGraphNode to JSON-serializable format."""
        return {
            "nodeName": node.name,
            "nodeDescription": node.description,
            "nodeInputs": [
                self._convert_agent_data_to_json(input_data)
                for input_data in node.inputs
            ],
            "nodeOutputs": [
                self._convert_agent_data_to_json(output_data)
                for output_data in node.outputs
            ],
        }

    def _convert_node_set_to_json(self, node_set: AgentGraphNodeSet) -> Dict[str, Any]:
        """Convert AgentGraphNodeSet to JSON-serializable format."""
        return {
            "agentGraphNodeSetNodes": [
                self._convert_node_to_json(node) for node in node_set.nodes
            ]
        }

    def _call_haskell_function(
        self, function_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a Haskell function with the given input data."""
        if not self.available:
            raise RuntimeError("Haskell binary is not available")

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"function": function_name, "input": input_data}, f)
            input_file = f.name

        try:
            # Call Haskell binary
            result = subprocess.run(
                [self.haskell_binary_path, input_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Haskell execution failed: {result.stderr}")

            # Parse output
            output_data = json.loads(result.stdout)
            return output_data

        finally:
            # Clean up temporary file
            os.unlink(input_file)

    def plan_path(
        self,
        node_set: AgentGraphNodeSet,
        desired_outputs: List[AgentData],
        external_inputs: Optional[List[AgentData]] = None,
    ) -> List[str]:
        """
        Plan execution path using the Haskell implementation.

        Args:
            node_set: The agent graph node set
            desired_outputs: List of desired outputs to produce
            external_inputs: Optional list of external inputs that are available

        Returns:
            List of node names in execution order

        Raises:
            RuntimeError: If Haskell binary is not available or execution fails
        """
        input_data = {
            "nodeSet": self._convert_node_set_to_json(node_set),
            "desiredOutputs": [
                self._convert_agent_data_to_json(output) for output in desired_outputs
            ],
        }

        if external_inputs is not None:
            input_data["externalInputs"] = [
                self._convert_agent_data_to_json(input_data)
                for input_data in external_inputs
            ]

        result = self._call_haskell_function("planExecutionPath", input_data)
        path_result = result.get("result", [])
        # Return None for empty paths to maintain compatibility with existing tests
        return None if not path_result else path_result

    def validate_outputs(
        self, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
    ) -> bool:
        """
        Validate that all desired outputs can be produced.

        Args:
            node_set: The agent graph node set
            desired_outputs: List of desired outputs to validate

        Returns:
            True if all outputs can be produced, False otherwise
        """
        input_data = {
            "nodeSet": self._convert_node_set_to_json(node_set),
            "desiredOutputs": [
                self._convert_agent_data_to_json(output) for output in desired_outputs
            ],
        }

        result = self._call_haskell_function("validateOutputs", input_data)
        return result.get("result", False)

    def get_available_outputs(self, node_set: AgentGraphNodeSet) -> List[AgentData]:
        """
        Get all outputs that can be produced by the node set.

        Args:
            node_set: The agent graph node set

        Returns:
            List of available outputs
        """
        input_data = {"nodeSet": self._convert_node_set_to_json(node_set)}

        result = self._call_haskell_function("availableOutputs", input_data)
        outputs_json = result.get("result", [])

        return [
            AgentData(
                name=output["agentDataName"],
                parameters=output["agentDataParameters"],
                description=output["agentDataDescription"],
                data=output["agentDataData"],
            )
            for output in outputs_json
        ]

    def has_circular_dependencies(self, node_set: AgentGraphNodeSet) -> bool:
        """
        Check if the node set has circular dependencies.

        Args:
            node_set: The agent graph node set

        Returns:
            True if circular dependencies exist, False otherwise
        """
        input_data = {"nodeSet": self._convert_node_set_to_json(node_set)}

        result = self._call_haskell_function("hasCircularDependencies", input_data)
        return result.get("result", False)


# Convenience functions for backward compatibility
def plan_path(
    node_set: AgentGraphNodeSet,
    desired_outputs: List[AgentData],
    external_inputs: Optional[List[AgentData]] = None,
) -> List[str]:
    """
    Plan execution path using the Haskell implementation.

    This is a convenience function that creates a PathPlanner instance
    and calls plan_path on it.
    """
    planner = PathPlanner()
    return planner.plan_path(node_set, desired_outputs, external_inputs)


def validate_outputs(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> bool:
    """
    Validate that all desired outputs can be produced.

    This is a convenience function that creates a PathPlanner instance
    and calls validate_outputs on it.
    """
    planner = PathPlanner()
    return planner.validate_outputs(node_set, desired_outputs)


def get_available_outputs(node_set: AgentGraphNodeSet) -> List[AgentData]:
    """
    Get all outputs that can be produced by the node set.

    This is a convenience function that creates a PathPlanner instance
    and calls get_available_outputs on it.
    """
    planner = PathPlanner()
    return planner.get_available_outputs(node_set)


def has_circular_dependencies(node_set: AgentGraphNodeSet) -> bool:
    """
    Check if the node set has circular dependencies.

    This is a convenience function that creates a PathPlanner instance
    and calls has_circular_dependencies on it.
    """
    planner = PathPlanner()
    return planner.has_circular_dependencies(node_set)
