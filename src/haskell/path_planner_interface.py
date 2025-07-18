"""
Python interface for the Haskell path planner.

This module provides a Python interface to the Haskell path planning implementation,
using the daemon approach for better performance and resource efficiency.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    Python interface to the Haskell path planner using daemon approach.

    This class maintains a persistent connection to the Haskell binary,
    avoiding the overhead of spawning a new process for each request.
    """

    def __init__(
        self, haskell_binary_path: Optional[str] = None, auto_start: bool = True
    ):
        """
        Initialize the PathPlanner.

        Args:
            haskell_binary_path: Optional path to the Haskell binary. If not provided,
                                will attempt to find or build it automatically.
            auto_start: Whether to automatically start the daemon on initialization.
        """
        self.haskell_binary_path = haskell_binary_path
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.request_id = 0
        self.pending_requests: Dict[int, Dict[str, Any]] = {}
        self.results: Dict[int, Any] = {}
        self.error: Optional[str] = None
        self.available = False

        if auto_start:
            self.start()

    def _find_haskell_binary(self) -> str:
        """Find the Haskell binary in the expected location."""
        # Look for the binary in the haskell directory
        haskell_dir = Path(__file__).parent
        binary_name = "taiat-path-planner"

        # Check if binary exists in haskell directory (copied from dist-newstyle)
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

    def start(self) -> None:
        """Start the Haskell daemon process."""
        if self.process is not None:
            return  # Already running

        try:
            if self.haskell_binary_path is None:
                self.haskell_binary_path = self._find_haskell_binary()

            # Start the Haskell process in daemon mode
            self.process = subprocess.Popen(
                [self.haskell_binary_path, "--daemon"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Start the response reader thread
            self.reader_thread = threading.Thread(
                target=self._read_responses, daemon=True
            )
            self.reader_thread.start()

            # Wait a moment for the process to start
            time.sleep(0.1)

            # Test the connection
            if self.process.poll() is None:
                self.available = True
            else:
                raise RuntimeError("Haskell daemon failed to start")

        except Exception as e:
            self.error = str(e)
            self.available = False
            if self.process:
                self.process.terminate()
                self.process = None
            raise

    def stop(self) -> None:
        """Stop the Haskell daemon process."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.available = False

    def _read_responses(self) -> None:
        """Background thread to read responses from the Haskell process."""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break

                try:
                    response = json.loads(line.strip())
                    request_id = response.get("request_id")
                    if request_id is not None:
                        with self.lock:
                            self.results[request_id] = response
                            if request_id in self.pending_requests:
                                del self.pending_requests[request_id]
                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue
        except Exception as e:
            self.error = f"Response reader error: {e}"
            self.available = False

    def _send_request(
        self, function_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a request to the Haskell daemon and wait for response."""
        if not self.available or self.process is None:
            raise RuntimeError("Haskell daemon is not available")

        with self.lock:
            self.request_id += 1
            request_id = self.request_id

        # Prepare the request
        request = {
            "request_id": request_id,
            "function": function_name,
            "input": input_data,
        }

        # Send the request
        try:
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to send request: {e}")

        # Wait for response
        timeout = 30  # 30 seconds timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    result = self.results.pop(request_id)
                    return result

            time.sleep(0.01)  # Small delay to avoid busy waiting

        # Timeout
        with self.lock:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]

        raise RuntimeError("Request timed out")

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

    def plan_path(
        self,
        node_set: AgentGraphNodeSet,
        desired_outputs: List[AgentData],
        external_inputs: Optional[List[AgentData]] = None,
    ) -> Optional[List[str]]:
        """
        Plan execution path using the Haskell implementation.

        Args:
            node_set: The agent graph node set
            desired_outputs: List of desired outputs to produce
            external_inputs: Optional list of external inputs that are available

        Returns:
            List of node names in execution order, or None if no path found

        Raises:
            RuntimeError: If Haskell daemon is not available or execution fails
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

        result = self._send_request("planExecutionPath", input_data)
        # Handle double-wrapped result from Haskell daemon
        inner_result = result.get("result", {})
        if isinstance(inner_result, dict):
            path_result = inner_result.get("result", [])
        else:
            path_result = inner_result
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

        result = self._send_request("validateOutputs", input_data)
        # Handle double-wrapped result from Haskell daemon
        inner_result = result.get("result", {})
        if isinstance(inner_result, dict):
            return inner_result.get("result", False)
        return inner_result

    def get_available_outputs(self, node_set: AgentGraphNodeSet) -> List[AgentData]:
        """
        Get all outputs that can be produced by the node set.

        Args:
            node_set: The agent graph node set

        Returns:
            List of available outputs
        """
        input_data = {"nodeSet": self._convert_node_set_to_json(node_set)}

        result = self._send_request("availableOutputs", input_data)
        # Handle double-wrapped result from Haskell daemon
        inner_result = result.get("result", {})
        if isinstance(inner_result, dict):
            outputs_json = inner_result.get("result", [])
        else:
            outputs_json = inner_result

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

        result = self._send_request("hasCircularDependencies", input_data)
        # Handle double-wrapped result from Haskell daemon
        inner_result = result.get("result", {})
        if isinstance(inner_result, dict):
            return inner_result.get("result", False)
        return inner_result

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global daemon instance for convenience functions
_global_daemon: Optional[PathPlanner] = None
_daemon_lock = threading.Lock()


def _get_global_daemon() -> PathPlanner:
    """Get or create the global daemon instance."""
    global _global_daemon
    with _daemon_lock:
        if _global_daemon is None or not _global_daemon.available:
            if _global_daemon:
                _global_daemon.stop()
            _global_daemon = PathPlanner()
        return _global_daemon


# Convenience functions for backward compatibility
def plan_path(
    node_set: AgentGraphNodeSet,
    desired_outputs: List[AgentData],
    external_inputs: Optional[List[AgentData]] = None,
) -> Optional[List[str]]:
    """
    Plan execution path using the Haskell implementation.

    This is a convenience function that uses the global daemon instance.
    """
    daemon = _get_global_daemon()
    return daemon.plan_path(node_set, desired_outputs, external_inputs)


def validate_outputs(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> bool:
    """
    Validate that all desired outputs can be produced.

    This is a convenience function that uses the global daemon instance.
    """
    daemon = _get_global_daemon()
    return daemon.validate_outputs(node_set, desired_outputs)


def get_available_outputs(node_set: AgentGraphNodeSet) -> List[AgentData]:
    """
    Get all outputs that can be produced by the node set.

    This is a convenience function that uses the global daemon instance.
    """
    daemon = _get_global_daemon()
    return daemon.get_available_outputs(node_set)


def has_circular_dependencies(node_set: AgentGraphNodeSet) -> bool:
    """
    Check if the node set has circular dependencies.

    This is a convenience function that uses the global daemon instance.
    """
    daemon = _get_global_daemon()
    return daemon.has_circular_dependencies(node_set)


def stop_global_daemon() -> None:
    """Stop the global daemon instance."""
    global _global_daemon
    with _daemon_lock:
        if _global_daemon:
            _global_daemon.stop()
            _global_daemon = None
