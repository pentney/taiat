#!/usr/bin/env python3
"""
Haskell Path Planner Interface for Taiat

This module provides a Python interface to the Haskell path planner,
replacing the Prolog implementation with better performance and type safety.
"""

import json
import subprocess
import tempfile
import os
import time
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    from taiat.base import AgentGraphNodeSet, AgentData
except ImportError:
    # Fallback for testing without full taiat package
    class AgentData:
        def __init__(self, name: str, parameters: Dict[str, str], description: str = "", data: Any = None):
            self.name = name
            self.parameters = parameters
            self.description = description
            self.data = data
        
        def __repr__(self):
            return f"AgentData(name='{self.name}', parameters={self.parameters})"
    
    class AgentGraphNode:
        def __init__(self, name: str, description: str, inputs: List[AgentData], outputs: List[AgentData], function=None):
            self.name = name
            self.description = description
            self.inputs = inputs
            self.outputs = outputs
            self.function = function
    
    class AgentGraphNodeSet:
        def __init__(self, nodes: List[AgentGraphNode]):
            self.nodes = nodes


class HaskellPathPlanner:
    """
    Python interface to the Haskell path planner.
    
    This class provides methods to interact with the Haskell implementation
    of the Taiat path planner, offering better performance and type safety
    compared to the Prolog version.
    """
    
    def __init__(self, haskell_binary_path: Optional[str] = None):
        """
        Initialize the Haskell path planner interface.
        
        Args:
            haskell_binary_path: Path to the compiled Haskell binary. If None,
                                will try to find it in the current directory.
        """
        self.haskell_binary_path = haskell_binary_path or self._find_haskell_binary()
        self.available = self._check_availability()
    
    def _find_haskell_binary(self) -> str:
        """Find the Haskell binary in the current directory."""
        current_dir = Path(__file__).parent
        binary_path = current_dir / "taiat-path-planner"
        
        if not binary_path.exists():
            # Try to build it
            self._build_haskell_binary(current_dir)
        
        return str(binary_path)
    
    def _build_haskell_binary(self, directory: Path) -> None:
        """Build the Haskell binary using cabal."""
        try:
            subprocess.run(
                ["cabal", "build"],
                cwd=directory,
                check=True,
                capture_output=True
            )
            # Copy the binary to the current directory
            dist_dir = directory / "dist-newstyle" / "build" / "x86_64-linux" / "ghc-9.4.7" / "taiat-path-planner-0.1.0.0" / "x" / "taiat-path-planner" / "build" / "taiat-path-planner" / "taiat-path-planner"
            if dist_dir.exists():
                import shutil
                shutil.copy2(dist_dir, directory / "taiat-path-planner")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not build Haskell binary: {e}")
            print("Please ensure cabal is installed and run 'cabal build' manually")
    
    def _check_availability(self) -> bool:
        """Check if the Haskell binary is available and working."""
        try:
            # Check if the binary file exists and is executable
            if not os.path.exists(self.haskell_binary_path):
                return False
            
            # Test with a simple call to see if it works
            result = subprocess.run(
                [self.haskell_binary_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            # The binary should run without arguments and produce some output
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _convert_agent_data_to_json(self, agent_data: AgentData) -> Dict[str, Any]:
        """Convert AgentData to JSON-serializable format."""
        return {
            "agentDataName": agent_data.name,
            "agentDataParameters": agent_data.parameters,
            "agentDataDescription": agent_data.description,
            "agentDataData": agent_data.data
        }
    
    def _convert_node_to_json(self, node) -> Dict[str, Any]:
        """Convert AgentGraphNode to JSON-serializable format."""
        return {
            "nodeName": node.name,
            "nodeDescription": node.description,
            "nodeInputs": [self._convert_agent_data_to_json(input_data) for input_data in node.inputs],
            "nodeOutputs": [self._convert_agent_data_to_json(output_data) for output_data in node.outputs]
        }
    
    def _convert_node_set_to_json(self, node_set: AgentGraphNodeSet) -> Dict[str, Any]:
        """Convert AgentGraphNodeSet to JSON-serializable format."""
        return {
            "agentGraphNodeSetNodes": [self._convert_node_to_json(node) for node in node_set.nodes]
        }
    
    def _call_haskell_function(self, function_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a Haskell function with the given input data."""
        if not self.available:
            raise RuntimeError("Haskell binary is not available")
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "function": function_name,
                "input": input_data
            }, f)
            input_file = f.name
        
        try:
            # Call Haskell binary
            result = subprocess.run(
                [self.haskell_binary_path, input_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Haskell execution failed: {result.stderr}")
            
            # Parse output
            output_data = json.loads(result.stdout)
            return output_data
            
        finally:
            # Clean up temporary file
            os.unlink(input_file)
    
    def plan_path(self, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]) -> List[str]:
        """
        Plan execution path using the Haskell implementation.
        
        Args:
            node_set: The agent graph node set
            desired_outputs: List of desired outputs to produce
            
        Returns:
            List of node names in execution order
            
        Raises:
            RuntimeError: If Haskell binary is not available or execution fails
        """
        input_data = {
            "nodeSet": self._convert_node_set_to_json(node_set),
            "desiredOutputs": [self._convert_agent_data_to_json(output) for output in desired_outputs]
        }
        
        result = self._call_haskell_function("planExecutionPath", input_data)
        return result.get("result", [])
    
    def validate_outputs(self, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]) -> bool:
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
            "desiredOutputs": [self._convert_agent_data_to_json(output) for output in desired_outputs]
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
        input_data = {
            "nodeSet": self._convert_node_set_to_json(node_set)
        }
        
        result = self._call_haskell_function("availableOutputs", input_data)
        outputs_json = result.get("result", [])
        
        return [
            AgentData(
                name=output["agentDataName"],
                parameters=output["agentDataParameters"],
                description=output["agentDataDescription"],
                data=output["agentDataData"]
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
        input_data = {
            "nodeSet": self._convert_node_set_to_json(node_set)
        }
        
        result = self._call_haskell_function("hasCircularDependencies", input_data)
        return result.get("result", False)


# Convenience functions for backward compatibility
def plan_taiat_path(node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]) -> List[str]:
    """
    Plan execution path using the Haskell implementation.
    
    This is a convenience function that creates a HaskellPathPlanner instance
    and calls plan_path on it.
    """
    planner = HaskellPathPlanner()
    return planner.plan_path(node_set, desired_outputs)


def validate_taiat_outputs(node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]) -> bool:
    """
    Validate that all desired outputs can be produced.
    
    This is a convenience function that creates a HaskellPathPlanner instance
    and calls validate_outputs on it.
    """
    planner = HaskellPathPlanner()
    return planner.validate_outputs(node_set, desired_outputs)


def get_taiat_available_outputs(node_set: AgentGraphNodeSet) -> List[AgentData]:
    """
    Get all outputs that can be produced by the node set.
    
    This is a convenience function that creates a HaskellPathPlanner instance
    and calls get_available_outputs on it.
    """
    planner = HaskellPathPlanner()
    return planner.get_available_outputs(node_set)


def has_taiat_circular_dependencies(node_set: AgentGraphNodeSet) -> bool:
    """
    Check if the node set has circular dependencies.
    
    This is a convenience function that creates a HaskellPathPlanner instance
    and calls has_circular_dependencies on it.
    """
    planner = HaskellPathPlanner()
    return planner.has_circular_dependencies(node_set)


# Performance comparison function
def compare_performance(node_set: AgentGraphNodeSet, desired_outputs: List[AgentData], 
                       prolog_planner=None) -> Dict[str, Any]:
    """
    Compare performance between Haskell and Prolog implementations.
    
    Args:
        node_set: The agent graph node set
        desired_outputs: List of desired outputs
        prolog_planner: Optional Prolog planner instance for comparison
        
    Returns:
        Dictionary with performance comparison results
    """
    results = {}
    
    # Test Haskell implementation
    haskell_planner = HaskellPathPlanner()
    if haskell_planner.available:
        start_time = time.time()
        try:
            haskell_result = haskell_planner.plan_path(node_set, desired_outputs)
            haskell_time = time.time() - start_time
            results["haskell"] = {
                "available": True,
                "time": haskell_time,
                "result": haskell_result,
                "success": True
            }
        except Exception as e:
            haskell_time = time.time() - start_time
            results["haskell"] = {
                "available": True,
                "time": haskell_time,
                "result": None,
                "success": False,
                "error": str(e)
            }
    else:
        results["haskell"] = {
            "available": False,
            "time": None,
            "result": None,
            "success": False,
            "error": "Haskell binary not available"
        }
    
    # Test Prolog implementation if available
    if prolog_planner is not None:
        start_time = time.time()
        try:
            prolog_result = prolog_planner.plan_path(node_set, desired_outputs)
            prolog_time = time.time() - start_time
            results["prolog"] = {
                "available": True,
                "time": prolog_time,
                "result": prolog_result,
                "success": True
            }
        except Exception as e:
            prolog_time = time.time() - start_time
            results["prolog"] = {
                "available": True,
                "time": prolog_time,
                "result": None,
                "success": False,
                "error": str(e)
            }
    else:
        results["prolog"] = {
            "available": False,
            "time": None,
            "result": None,
            "success": False,
            "error": "Prolog planner not provided"
        }
    
    # Calculate performance improvement
    if (results["haskell"]["available"] and results["haskell"]["success"] and
        results["prolog"]["available"] and results["prolog"]["success"]):
        haskell_time = results["haskell"]["time"]
        prolog_time = results["prolog"]["time"]
        improvement = ((prolog_time - haskell_time) / prolog_time) * 100
        results["performance_improvement"] = improvement
    
    return results


if __name__ == "__main__":
    # Test the interface
    print("Testing Haskell Path Planner Interface...")
    
    # Create a simple test case
    input_a = AgentData("input_a", {}, "Input A")
    output_b = AgentData("output_b", {}, "Output B")
    output_c = AgentData("output_c", {}, "Output C")
    final_output = AgentData("final_output", {}, "Final Output")
    
    node_a = type('Node', (), {
        'name': 'node_a',
        'description': 'Process A',
        'inputs': [input_a],
        'outputs': [output_b]
    })()
    
    node_b = type('Node', (), {
        'name': 'node_b',
        'description': 'Process B',
        'inputs': [output_b],
        'outputs': [output_c]
    })()
    
    node_c = type('Node', (), {
        'name': 'node_c',
        'description': 'Process C',
        'inputs': [output_c],
        'outputs': [final_output]
    })()
    
    node_set = AgentGraphNodeSet([node_a, node_b, node_c])
    desired_outputs = [final_output]
    
    # Test the planner
    try:
        planner = HaskellPathPlanner()
        if planner.available:
            print("Haskell planner is available")
            result = planner.plan_path(node_set, desired_outputs)
            print(f"Execution path: {result}")
        else:
            print("Haskell planner is not available")
    except Exception as e:
        print(f"Error testing Haskell planner: {e}") 