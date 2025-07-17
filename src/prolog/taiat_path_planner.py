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
from typing import Optional, List, Dict, Any
from pathlib import Path

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData


class TaiatPathPlanner:
    """
    Taiat Path Planner that uses Prolog for optimal path planning.

    This planner compiles the Prolog path planner once and reuses it for all queries,
    providing efficient execution path determination.
    """

    def __init__(self, prolog_script_path: Optional[str] = None, max_nodes: int = 200):
        """
        Initialize the Taiat Path Planner.

        Args:
            prolog_script_path: Path to the Prolog script file. If None, uses default path.
            max_nodes: Maximum number of nodes allowed in a node set (default: 200)
        """
        if prolog_script_path is None:
            # Use the default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prolog_script_path = os.path.join(current_dir, "path_planner.pl")
        else:
            self.prolog_script_path = prolog_script_path

        self.max_nodes = max_nodes

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

        # Compile the path planner once
        self._compile_path_planner()

    def _compile_path_planner(self):
        """Compile the path planner once and store the executable path."""
        print("Compiling Prolog path planner (one-time setup)...")

        # Create a temporary directory for the compiled program
        self.temp_dir = tempfile.mkdtemp(prefix="taiat_prolog_")
        self.compiled_program = os.path.join(self.temp_dir, "path_planner.exe")

        # Create a wrapper program that includes the path planner
        wrapper_program = f"""
:- initialization(main).
:- include('{self.prolog_script_path}').

% This is a wrapper that will be compiled once
% The actual queries will be passed as data
main :-
    % Read the query data from stdin
    read(NodeSet),
    read(DesiredOutputs),
    % Execute the path planning
    plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath),
    % Output the result
    write('EXECUTION_PATH:'), write(ExecutionPath), nl,
    halt.
"""

        # Write the wrapper program
        wrapper_path = os.path.join(self.temp_dir, "wrapper.pl")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_program)

        # Compile the wrapper program
        start_time = time.time()
        result = subprocess.run(
            ["gplc", "-o", self.compiled_program, wrapper_path],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "GLOBALSZ": "65536", "TRAILSZ": "32768"},
        )
        compilation_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to compile Prolog path planner: {result.stderr}"
            )

        print(f"Prolog path planner compiled successfully in {compilation_time:.3f}s")
        print(f"Compiled program: {self.compiled_program}")

        # Clean up wrapper file
        os.unlink(wrapper_path)

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
        # Convert parameters to Prolog format with hyphens
        if not agent_data.parameters:
            params_str = "[]"
        else:
            param_list = [f"{k}-{v}" for k, v in agent_data.parameters.items()]
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
        Plan the execution path using the pre-compiled Prolog program.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            desired_outputs: List of desired outputs to produce

        Returns:
            List of node names in execution order, or None if planning failed
        """
        # Check node set size
        if not self._check_node_set_size(node_set):
            return None

        try:
            # Convert data to Prolog format
            node_set_str = self._node_set_to_prolog(node_set)
            outputs_str = self._desired_outputs_to_prolog(desired_outputs)

            # Run the compiled Prolog program
            proc = subprocess.run(
                [self.compiled_program],
                input=f"{node_set_str}.\n{outputs_str}.\n",
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                print(f"Prolog planner error: {proc.stderr}")
                return None

            # Parse the output
            for line in proc.stdout.splitlines():
                if line.startswith("EXECUTION_PATH:"):
                    result_str = line[len("EXECUTION_PATH:") :].strip()
                    # If the result is an empty list, treat as failure
                    if result_str == "[]":
                        return None
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
            print("No execution path found in Prolog output")
            return None
        except Exception as e:
            print(f"Error in optimized Prolog planning: {e}")
            return None

    def plan_multiple_paths(
        self, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
    ) -> List[List[str]]:
        """
        Plan multiple alternative execution paths using the pre-compiled Prolog program.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            desired_outputs: List of desired outputs to produce

        Returns:
            List of alternative execution paths, each as a list of node names
        """
        # Check node set size
        if not self._check_node_set_size(node_set):
            return []

        try:
            # Convert data to Prolog format
            node_set_str = self._node_set_to_prolog(node_set)
            outputs_str = self._desired_outputs_to_prolog(desired_outputs)

            # Create a wrapper program for multiple path planning
            wrapper_program = f"""
:- initialization(main).
:- include('{self.prolog_script_path}').

main :-
    read(NodeSet),
    read(DesiredOutputs),
    plan_multiple_execution_paths(NodeSet, DesiredOutputs, ExecutionPaths),
    write('MULTIPLE_PATHS:'), write(ExecutionPaths), nl,
    halt.
"""

            # Write temporary wrapper
            wrapper_path = os.path.join(self.temp_dir, "multi_path_wrapper.pl")
            with open(wrapper_path, "w") as f:
                f.write(wrapper_program)

            # Compile the wrapper program
            multi_path_program = os.path.join(self.temp_dir, "multi_path_planner.exe")
            result = subprocess.run(
                ["gplc", "-o", multi_path_program, wrapper_path],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "GLOBALSZ": "65536", "TRAILSZ": "32768"},
            )

            if result.returncode != 0:
                print(f"Failed to compile multi-path planner: {result.stderr}")
                return []

            # Run the compiled program
            proc = subprocess.run(
                [multi_path_program],
                input=f"{node_set_str}.\n{outputs_str}.\n",
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                print(f"Multi-path Prolog planner error: {proc.stderr}")
                return []

            # Parse the output
            for line in proc.stdout.splitlines():
                if line.startswith("MULTIPLE_PATHS:"):
                    result_str = line[len("MULTIPLE_PATHS:") :].strip()
                    return self._parse_multiple_paths(result_str)

            print("No multiple paths found in Prolog output")
            return []

        except Exception as e:
            print(f"Error in multi-path Prolog planning: {e}")
            return []

    def plan_path_excluding_failed(
        self,
        node_set: AgentGraphNodeSet,
        desired_outputs: List[AgentData],
        failed_nodes: List[str],
    ) -> Optional[List[str]]:
        """
        Plan execution path excluding failed nodes.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            desired_outputs: List of desired outputs to produce
            failed_nodes: List of node names that have failed

        Returns:
            List of node names in execution order, or None if planning failed
        """
        # Check node set size
        if not self._check_node_set_size(node_set):
            return None

        try:
            # Convert data to Prolog format
            node_set_str = self._node_set_to_prolog(node_set)
            outputs_str = self._desired_outputs_to_prolog(desired_outputs)
            failed_nodes_str = (
                "[" + ", ".join([f"'{node}'" for node in failed_nodes]) + "]"
            )

            # Create a wrapper program for failed node exclusion
            wrapper_program = f"""
:- initialization(main).
:- include('{self.prolog_script_path}').

main :-
    read(NodeSet),
    read(DesiredOutputs),
    read(FailedNodes),
    plan_execution_path_excluding_failed(NodeSet, DesiredOutputs, FailedNodes, ExecutionPath),
    write('EXECUTION_PATH:'), write(ExecutionPath), nl,
    halt.
"""

            # Write temporary wrapper
            wrapper_path = os.path.join(self.temp_dir, "exclude_failed_wrapper.pl")
            with open(wrapper_path, "w") as f:
                f.write(wrapper_program)

            # Compile the wrapper program
            exclude_program = os.path.join(self.temp_dir, "exclude_failed_planner.exe")
            result = subprocess.run(
                ["gplc", "-o", exclude_program, wrapper_path],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "GLOBALSZ": "65536", "TRAILSZ": "32768"},
            )

            if result.returncode != 0:
                print(f"Failed to compile exclude-failed planner: {result.stderr}")
                return None

            # Run the compiled program
            proc = subprocess.run(
                [exclude_program],
                input=f"{node_set_str}.\n{outputs_str}.\n{failed_nodes_str}.\n",
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                print(f"Exclude-failed Prolog planner error: {proc.stderr}")
                return None

            # Parse the output
            for line in proc.stdout.splitlines():
                if line.startswith("EXECUTION_PATH:"):
                    result_str = line[len("EXECUTION_PATH:") :].strip()
                    # If the result is an empty list, treat as failure
                    if result_str == "[]":
                        return None
                    # Parse the Prolog list of node names
                    import re

                    node_pattern_quoted = r"'([^']+)'"
                    node_names = re.findall(node_pattern_quoted, result_str)
                    if not node_names:
                        node_pattern_unquoted = r"([a-zA-Z0-9_]+)"
                        node_names = re.findall(node_pattern_unquoted, result_str)
                    node_names = [
                        n for n in node_names if n not in ("ExecutionPath", "[]")
                    ]
                    if node_names:
                        return node_names
                    else:
                        print(f"Could not parse execution path: {result_str}")
                        return None
            print("No execution path found in Prolog output")
            return None

        except Exception as e:
            print(f"Error in exclude-failed Prolog planning: {e}")
            return None

    def find_alternative_paths_on_failure(
        self,
        node_set: AgentGraphNodeSet,
        desired_outputs: List[AgentData],
        failed_node: str,
    ) -> List[List[str]]:
        """
        Find alternative paths when a specific node fails.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            desired_outputs: List of desired outputs to produce
            failed_node: Name of the node that failed

        Returns:
            List of alternative execution paths
        """
        # Check node set size
        if not self._check_node_set_size(node_set):
            return []

        try:
            # Convert data to Prolog format
            node_set_str = self._node_set_to_prolog(node_set)
            outputs_str = self._desired_outputs_to_prolog(desired_outputs)
            failed_node_str = f"'{failed_node}'"

            # Create a wrapper program for alternative path finding
            wrapper_program = f"""
:- initialization(main).
:- include('{self.prolog_script_path}').

main :-
    read(NodeSet),
    read(DesiredOutputs),
    read(FailedNode),
    find_alternative_paths_on_failure(NodeSet, DesiredOutputs, FailedNode, AlternativePaths),
    write('ALTERNATIVE_PATHS:'), write(AlternativePaths), nl,
    halt.
"""

            # Write temporary wrapper
            wrapper_path = os.path.join(self.temp_dir, "alternative_paths_wrapper.pl")
            with open(wrapper_path, "w") as f:
                f.write(wrapper_program)

            # Compile the wrapper program
            alternative_program = os.path.join(
                self.temp_dir, "alternative_paths_planner.exe"
            )
            result = subprocess.run(
                ["gplc", "-o", alternative_program, wrapper_path],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "GLOBALSZ": "65536", "TRAILSZ": "32768"},
            )

            if result.returncode != 0:
                print(f"Failed to compile alternative paths planner: {result.stderr}")
                return []

            # Run the compiled program
            proc = subprocess.run(
                [alternative_program],
                input=f"{node_set_str}.\n{outputs_str}.\n{failed_node_str}.\n",
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                print(f"Alternative paths Prolog planner error: {proc.stderr}")
                return []

            # Parse the output
            for line in proc.stdout.splitlines():
                if line.startswith("ALTERNATIVE_PATHS:"):
                    result_str = line[len("ALTERNATIVE_PATHS:") :].strip()
                    return self._parse_multiple_paths(result_str)

            print("No alternative paths found in Prolog output")
            return []

        except Exception as e:
            print(f"Error in alternative paths Prolog planning: {e}")
            return []

    def _parse_multiple_paths(self, result_str: str) -> List[List[str]]:
        """
        Parse multiple paths from Prolog output.

        Args:
            result_str: String representation of multiple paths from Prolog

        Returns:
            List of paths, each as a list of node names
        """
        import re

        # Handle empty result
        if result_str == "[]":
            return []

        # Parse nested lists structure
        # This is a simplified parser - in practice, you might need more sophisticated parsing
        paths = []

        # Find all path lists in the result
        path_pattern = r"\[([^\]]+)\]"
        path_matches = re.findall(path_pattern, result_str)

        for path_match in path_matches:
            # Parse individual node names in each path
            node_pattern_quoted = r"'([^']+)'"
            node_names = re.findall(node_pattern_quoted, path_match)
            if not node_names:
                node_pattern_unquoted = r"([a-zA-Z0-9_]+)"
                node_names = re.findall(node_pattern_unquoted, path_match)

            # Filter out non-node names
            node_names = [
                n
                for n in node_names
                if n not in ("ExecutionPath", "[]", "AlternativePaths")
            ]

            if node_names:
                paths.append(node_names)

        return paths

    def validate_path_viability(
        self,
        node_set: AgentGraphNodeSet,
        execution_path: List[str],
        excluded_nodes: List[str],
    ) -> bool:
        """
        Validate if a path is still viable after excluding certain nodes.

        Args:
            node_set: The AgentGraphNodeSet containing all available nodes
            execution_path: List of node names in the execution path
            excluded_nodes: List of node names that are excluded

        Returns:
            True if the path is viable, False otherwise
        """
        # Check if any node in the execution path is in the excluded list
        for node_name in execution_path:
            if node_name in excluded_nodes:
                return False
        return True

    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        try:
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                import shutil

                shutil.rmtree(self.temp_dir)
        except Exception:
            pass


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
    return planner.plan_path(node_set, desired_outputs)


# Global planner instance for reuse (avoids recompilation)
_global_planner = None


def get_global_planner() -> TaiatPathPlanner:
    """Get or create a global planner instance."""
    global _global_planner
    if _global_planner is None:
        _global_planner = TaiatPathPlanner()
    return _global_planner


def clear_global_planner():
    """Clear the global planner cache to force recompilation."""
    global _global_planner
    if _global_planner is not None:
        del _global_planner
        _global_planner = None


def plan_taiat_path_global(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData]
) -> Optional[List[str]]:
    """
    Plan execution path using the global Prolog engine.

    This reuses the same compiled executable across multiple calls,
    eliminating both compilation and process startup overhead.

    Args:
        node_set: The AgentGraphNodeSet containing all available nodes
        desired_outputs: List of desired outputs to produce

    Returns:
        List of node names in execution order, or None if planning failed
    """
    planner = get_global_planner()
    return planner.plan_path(node_set, desired_outputs)
