#!/usr/bin/env python3
"""
Taiat Path Planner Interface

This module provides the standard Taiat path planner that uses Prolog
for optimal execution path determination.
"""

import os
import subprocess
import tempfile
import time
import signal
import atexit
from typing import Optional, List, Dict, Any
from pathlib import Path

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData


class TaiatPathPlanner:
    """
    Taiat Path Planner that uses Prolog for optimal path planning.

    This planner runs a single Prolog daemon process and communicates with it
    via stdin/stdout for all queries, providing efficient execution path determination.
    """

    def __init__(self, prolog_script_path: Optional[str] = None, max_nodes: int = 1000):
        """
        Initialize the Taiat Path Planner.

        Args:
            prolog_script_path: Path to the Prolog script file. If None, uses default path.
            max_nodes: Maximum number of nodes allowed in a node set (default: 1000)
        """
        if prolog_script_path is None:
            # Use the default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prolog_script_path = os.path.join(current_dir, "path_planner.pl")
        else:
            self.prolog_script_path = prolog_script_path

        self.max_nodes = max_nodes
        self.daemon_process = None
        self.daemon_stdin = None
        self.daemon_stdout = None

        # Verify that the Prolog script exists
        if not os.path.exists(self.prolog_script_path):
            raise FileNotFoundError(
                f"Prolog script not found: {self.prolog_script_path}"
            )

        # Verify that gplc is available
        try:
            subprocess.run(["gplc", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("GNU Prolog (gplc) is not available. Please install it.")

        # Start the Prolog daemon
        self._start_daemon()

    def _start_daemon(self):
        """Start the Prolog daemon process."""
        print("Starting Prolog path planner daemon...")

        # Create a wrapper program that runs as a daemon
        wrapper_program = f"""
:- initialization(main).
:- include('{self.prolog_script_path}').

% This is a daemon that reads queries from stdin and writes results to stdout
main :-
    % Set up signal handling for graceful shutdown
    catch(
        daemon_loop,
        _,
        halt
    ).

daemon_loop :-
    % Read the query data from stdin
    read(NodeSet),
    read(DesiredOutputs),
    % Execute the path planning
    (plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath) ->
        % Success - output the result
        write('SUCCESS:'), write(ExecutionPath), nl,
        flush_output
    ;   % Failure - output empty result
        write('FAILURE:'), nl,
        flush_output
    ),
    % Continue listening for more queries
    daemon_loop.
"""

        # Write the wrapper program
        self.temp_dir = tempfile.mkdtemp(prefix="taiat_prolog_daemon_")
        wrapper_path = os.path.join(self.temp_dir, "daemon.pl")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_program)

        # Compile the daemon
        start_time = time.time()
        
        # Set memory environment variables
        memory_env = {**os.environ, "GLOBALSZ": "1048576", "TRAILSZ": "524288"}
        
        compiled_daemon = os.path.join(self.temp_dir, "daemon.exe")
        result = subprocess.run(
            ["gplc", "-o", compiled_daemon, wrapper_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=memory_env,
        )
        compilation_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to compile Prolog daemon: {result.stderr}"
            )

        print(f"Prolog daemon compiled successfully in {compilation_time:.3f}s")

        # Start the daemon process
        exec_env = {**os.environ, "GLOBALSZ": "1048576", "TRAILSZ": "524288", "HEAPSZ": "1048576"}
        
        self.daemon_process = subprocess.Popen(
            [compiled_daemon],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=exec_env,
            bufsize=1,  # Line buffered
        )

        # Test the daemon with a simple query to ensure it's working
        test_node_set = "agent_graph_node_set([])."
        test_outputs = "[]."
        
        try:
            self.daemon_process.stdin.write(f"{test_node_set}\n{test_outputs}\n")
            self.daemon_process.stdin.flush()
            
            # Read response with timeout
            import select
            ready, _, _ = select.select([self.daemon_process.stdout], [], [], 5.0)
            if ready:
                response = self.daemon_process.stdout.readline().strip()
                if response.startswith("SUCCESS:") or response.startswith("FAILURE:"):
                    print("Prolog daemon started successfully")
                else:
                    raise RuntimeError(f"Unexpected daemon response: {response}")
            else:
                raise RuntimeError("Daemon did not respond within timeout")
                
        except Exception as e:
            self._cleanup_daemon()
            raise RuntimeError(f"Failed to start Prolog daemon: {e}")

        # Clean up wrapper file
        os.unlink(wrapper_path)

    def _cleanup_daemon(self):
        """Clean up the daemon process and temporary files."""
        if self.daemon_process is not None:
            try:
                # Send SIGTERM to the daemon
                self.daemon_process.terminate()
                
                # Wait for it to terminate gracefully
                try:
                    self.daemon_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    self.daemon_process.kill()
                    self.daemon_process.wait()
                    
            except Exception as e:
                print(f"Warning: Error cleaning up daemon process: {e}")
            finally:
                self.daemon_process = None

        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary directory: {e}")

    def _check_node_set_size(self, node_set: AgentGraphNodeSet) -> bool:
        """
        Check if the node set size is within limits.

        Args:
            node_set: The node set to check

        Returns:
            True if the node set is within limits, False otherwise
        """
        if len(node_set.nodes) > self.max_nodes:
            print(
                f"Warning: Node set size ({len(node_set.nodes)}) exceeds maximum ({self.max_nodes})"
            )
            return False
        return True

    def _agent_data_to_prolog(self, agent_data: AgentData) -> str:
        """
        Convert AgentData to Prolog format.

        Args:
            agent_data: The AgentData object to convert

        Returns:
            Prolog representation as a string
        """
        # Convert parameters to Prolog format with hyphens and proper quoting
        if not agent_data.parameters:
            params_str = "[]"
        else:
            param_list = [f"'{k}'-'{v}'" for k, v in agent_data.parameters.items()]
            params_str = f"[{', '.join(param_list)}]"

        # Escape single quotes in description
        description = (
            agent_data.description.replace("'", "\\'") if agent_data.description else ""
        )
        return f"agent_data('{agent_data.name}', {params_str}, '{description}', null)"

    def _node_to_prolog(self, node: AgentGraphNode) -> str:
        """
        Convert AgentGraphNode to Prolog format.

        Args:
            node: The AgentGraphNode object to convert

        Returns:
            Prolog representation as a string
        """
        inputs_str = (
            "["
            + ", ".join(
                self._agent_data_to_prolog(input_data) for input_data in node.inputs
            )
            + "]"
        )
        outputs_str = (
            "["
            + ", ".join(
                self._agent_data_to_prolog(output_data) for output_data in node.outputs
            )
            + "]"
        )

        description = node.description or ""
        description = description.replace("'", "\\'")  # Escape single quotes
        return f"node('{node.name}', '{description}', {inputs_str}, {outputs_str})"

    def _node_set_to_prolog(self, node_set: AgentGraphNodeSet) -> str:
        """
        Convert AgentGraphNodeSet to Prolog format.

        Args:
            node_set: The AgentGraphNodeSet object to convert

        Returns:
            Prolog representation as a string
        """
        nodes_str = (
            "[" + ", ".join(self._node_to_prolog(node) for node in node_set.nodes) + "]"
        )
        return f"agent_graph_node_set({nodes_str})"

    def _desired_outputs_to_prolog(self, desired_outputs: List[AgentData]) -> str:
        """
        Convert list of desired outputs to Prolog format.

        Args:
            desired_outputs: List of AgentData objects representing desired outputs

        Returns:
            Prolog representation as a string
        """
        outputs_str = (
            "["
            + ", ".join(
                self._agent_data_to_prolog(output) for output in desired_outputs
            )
            + "]"
        )
        return outputs_str

    def plan_path(
        self, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
    ) -> Optional[List[str]]:
        """
        Plan the execution path using the Prolog daemon.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            desired_outputs: List of desired outputs to produce

        Returns:
            List of node names in execution order, or None if planning failed
        """
        # Check node set size
        if not self._check_node_set_size(node_set):
            return None

        # Check if daemon is still running
        if self.daemon_process is None or self.daemon_process.poll() is not None:
            raise RuntimeError("Prolog daemon is not running")

        try:
            # Convert data to Prolog format
            node_set_str = self._node_set_to_prolog(node_set)
            outputs_str = self._desired_outputs_to_prolog(desired_outputs)

            # Send query to daemon
            query = f"{node_set_str}.\n{outputs_str}.\n"
            self.daemon_process.stdin.write(query)
            self.daemon_process.stdin.flush()

            # Read response with timeout
            import select
            ready, _, _ = select.select([self.daemon_process.stdout], [], [], 30.0)
            if not ready:
                raise RuntimeError("Daemon did not respond within timeout")

            response = self.daemon_process.stdout.readline().strip()
            
            if response.startswith("SUCCESS:"):
                result_str = response[len("SUCCESS:"):].strip()
                # Parse the Prolog list of node names (quoted or unquoted)
                import re

                # Try quoted atoms first
                node_pattern_quoted = r"'([^']+)'"
                node_names = re.findall(node_pattern_quoted, result_str)
                if not node_names:
                    # Fallback: unquoted atoms (e.g., [data_loader,preprocessor,...])
                    node_pattern_unquoted = r"([a-zA-Z0-9_]+)"
                    node_names = re.findall(node_pattern_unquoted, result_str)
                # Remove any matches that are not actual node names (e.g., 'ExecutionPath')
                node_names = [
                    n for n in node_names if n not in ("ExecutionPath", "[]")
                ]
                if node_names:
                    return node_names
                else:
                    print(f"Could not parse execution path: {result_str}")
                    return None
            elif response.startswith("FAILURE:"):
                return None
            else:
                print(f"Unexpected daemon response: {response}")
                return None

        except Exception as e:
            print(f"Error in Prolog daemon planning: {e}")
            return None

    def __del__(self):
        """Clean up daemon process and temporary files when the object is destroyed."""
        self._cleanup_daemon()


def plan_taiat_path(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> Optional[List[str]]:
    """
    Plan Taiat execution path using Prolog.

    Args:
        node_set: The AgentGraphNodeSet containing all available nodes
        desired_outputs: List of desired outputs to produce

    Returns:
        List of node names in execution order, or None if planning failed
    """
    planner = TaiatPathPlanner()
    try:
        return planner.plan_path(node_set, desired_outputs)
    finally:
        planner._cleanup_daemon()


# Global planner instance for reuse (avoids daemon restart)
_global_planner = None


def get_global_planner() -> TaiatPathPlanner:
    """Get or create a global planner instance."""
    global _global_planner
    if _global_planner is None:
        _global_planner = TaiatPathPlanner()
    return _global_planner


def clear_global_planner():
    """Clear the global planner cache to force daemon restart."""
    global _global_planner
    if _global_planner is not None:
        _global_planner._cleanup_daemon()
        _global_planner = None


def plan_taiat_path_global(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> Optional[List[str]]:
    """
    Plan execution path using the global Prolog daemon.

    This reuses the same daemon process across multiple calls,
    eliminating both compilation and process startup overhead.

    Args:
        node_set: The AgentGraphNodeSet containing all available nodes
        desired_outputs: List of desired outputs to produce

    Returns:
        List of node names in execution order, or None if planning failed
    """
    planner = get_global_planner()
    return planner.plan_path(node_set, desired_outputs)


# Register cleanup function to be called on exit
atexit.register(clear_global_planner)
