#!/usr/bin/env python3
"""
Proper Global Optimized Prolog vs Original TaiatManager Performance Comparison

This script compares the global optimized Prolog planner against the actual
TaiatBuilder.get_plan() method, which is what the original performance comparison tested.
"""

import sys
import os
import time
import statistics
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from taiat.base import AgentGraphNodeSet, AgentGraphNode, AgentData, TaiatQuery
from taiat.builder import TaiatBuilder
from prolog.taiat_path_planner import plan_taiat_path_global


class DummyLLM:
    def __call__(self, *args, **kwargs):
        return "dummy"


def create_performance_test_node_set(
    size: int, shuffle: bool = False
) -> AgentGraphNodeSet:
    """
    Create a node set of specified size for performance testing.

    Args:
        size: Number of nodes to create
        shuffle: Whether to shuffle the nodes to test loading impact

    Returns:
        AgentGraphNodeSet with the specified number of nodes
    """
    nodes = []

    # Create a dummy function for test nodes
    def dummy_function(state):
        return state

    # Create nodes with simpler dependencies to avoid stack overflow
    for i in range(size):
        # Determine inputs based on node position - simpler dependencies
        inputs = []
        if i > 0:
            # Each node depends on at most 1-2 previous nodes (reduced from 1-3)
            num_dependencies = min(random.randint(1, 2), i)
            for j in range(num_dependencies):
                dep_index = random.randint(
                    max(0, i - 3), i - 1
                )  # Only depend on recent nodes
                input_name = f"output_{dep_index}"
                inputs.append(
                    AgentData(
                        name=input_name,
                        parameters={},
                        description=f"Output from node {dep_index}",
                        data=None,
                    )
                )

        # Each node produces exactly 1 output (reduced from 1-2)
        outputs = []
        outputs.append(
            AgentData(
                name=f"output_{i}",
                parameters={},
                description=f"Output from node {i}",
                data=None,
            )
        )

        node = AgentGraphNode(
            name=f"node_{i}",
            description=f"Test node {i}",
            function=dummy_function,  # Add dummy function
            inputs=inputs,
            outputs=outputs,
        )
        nodes.append(node)

    # Shuffle nodes if requested to test loading impact
    if shuffle:
        random.shuffle(nodes)

    return AgentGraphNodeSet(nodes=nodes)


def test_original_taiat_builder(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData], num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test the original TaiatBuilder.get_plan() method.

    Args:
        node_set: The node set to test
        desired_outputs: Desired outputs
        num_runs: Number of test runs

    Returns:
        Dictionary with performance results
    """
    print(
        f"  Testing Original TaiatBuilder.get_plan() ({len(node_set.nodes)} nodes)..."
    )

    # Compute input nodes (nodes with no inputs)
    input_node_names = [node.name for node in node_set.nodes if not node.inputs]
    # Compute terminal nodes (nodes that produce any desired output)
    desired_output_names = set(output.name for output in desired_outputs)
    terminal_node_names = [
        node.name
        for node in node_set.nodes
        if any(output.name in desired_output_names for output in node.outputs)
    ]

    # Convert input node names to AgentData objects as expected by TaiatBuilder.build()
    input_agent_data = []
    for node_name in input_node_names:
        # Find the node and use its first output as the input data
        node = next(node for node in node_set.nodes if node.name == node_name)
        if node.outputs:
            input_agent_data.append(node.outputs[0])
        else:
            # Create a dummy AgentData if node has no outputs
            input_agent_data.append(
                AgentData(
                    name=f"input_{node_name}",
                    parameters={},
                    description=f"Input from {node_name}",
                    data=None,
                )
            )

    times = []
    success_count = 0

    for run in range(num_runs):
        try:
            # Time the path planning
            start_time = time.time()

            builder = TaiatBuilder(llm=DummyLLM())
            builder.build(
                node_set, inputs=input_agent_data, terminal_nodes=terminal_node_names
            )
            query = TaiatQuery(query="Performance test query")
            graph, status, error = builder.get_plan(query, desired_outputs)

            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            if status == "success":
                success_count += 1
                print(f"    Run {run + 1}: {execution_time:.4f}s - Success")
            else:
                print(f"    Run {run + 1}: {execution_time:.4f}s - Failed: {error}")

        except Exception as e:
            print(f"    Run {run + 1}: Error - {e}")
            times.append(float("inf"))

    if times:
        avg_time = statistics.mean([t for t in times if t != float("inf")])
        min_time = min([t for t in times if t != float("inf")])
        max_time = max([t for t in times if t != float("inf")])
    else:
        avg_time = min_time = max_time = float("inf")

    reliability = (success_count / num_runs) * 100

    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "reliability": reliability,
        "success_count": success_count,
        "total_runs": num_runs,
    }


def test_global_optimized_prolog(
    node_set: AgentGraphNodeSet, desired_outputs: List[AgentData], num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test the global optimized Prolog planner.

    Args:
        node_set: The node set to test
        desired_outputs: Desired outputs
        num_runs: Number of test runs

    Returns:
        Dictionary with performance results
    """
    print(f"  Testing Global Optimized Prolog ({len(node_set.nodes)} nodes)...")

    times = []
    success_count = 0

    for run in range(num_runs):
        try:
            # Time the path planning (reuses the same planner instance)
            start_time = time.time()
            execution_path = plan_taiat_path_global(node_set, desired_outputs)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            if execution_path:
                success_count += 1
                print(f"    Run {run + 1}: {execution_time:.4f}s - Success")
            else:
                print(f"    Run {run + 1}: {execution_time:.4f}s - Failed")

        except Exception as e:
            print(f"    Run {run + 1}: Error - {e}")
            times.append(float("inf"))

    if times:
        avg_time = statistics.mean([t for t in times if t != float("inf")])
        min_time = min([t for t in times if t != float("inf")])
        max_time = max([t for t in times if t != float("inf")])
    else:
        avg_time = min_time = max_time = float("inf")

    reliability = (success_count / num_runs) * 100

    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "reliability": reliability,
        "success_count": success_count,
        "total_runs": num_runs,
    }


def run_performance_comparison(node_sizes: List[int] = None, num_runs: int = 5):
    """
    Run the performance comparison between original TaiatBuilder and global optimized Prolog.

    Args:
        node_sizes: List of node sizes to test
        num_runs: Number of runs per test
    """
    if node_sizes is None:
        node_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("=" * 80)
    print("GLOBAL OPTIMIZED PROLOG vs ORIGINAL TAITBUILDER PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Testing node sizes: {node_sizes}")
    print(f"Runs per test: {num_runs}")
    print()

    # Set random seed for reproducible results
    random.seed(42)

    results = {
        "node_sizes": [],
        "original_taiat_builder": [],
        "global_optimized_prolog": [],
        "speedups": [],
    }

    for num_nodes in node_sizes:
        print(f"Testing {num_nodes} nodes:")
        print("-" * 40)

        # Create test data
        node_set = create_performance_test_node_set(num_nodes)

        # Create desired outputs (subset of outputs, like the original test)
        num_desired_outputs = min(3, num_nodes // 5 + 1)
        desired_outputs = []
        for i in range(num_desired_outputs):
            output_index = num_nodes - 1 - i
            if output_index >= 0:
                desired_outputs.append(
                    AgentData(
                        name=f"output_{output_index}",
                        parameters={},
                        description=f"Output from node {output_index}",
                        data=None,
                    )
                )

        # Test original TaiatBuilder
        original_results = test_original_taiat_builder(
            node_set, desired_outputs, num_runs
        )

        # Test global optimized Prolog
        prolog_results = test_global_optimized_prolog(
            node_set, desired_outputs, num_runs
        )

        # Calculate speedup
        if original_results["avg_time"] != float("inf") and prolog_results[
            "avg_time"
        ] != float("inf"):
            speedup = original_results["avg_time"] / prolog_results["avg_time"]
        else:
            speedup = float("inf")

        # Store results
        results["node_sizes"].append(num_nodes)
        results["original_taiat_builder"].append(original_results)
        results["global_optimized_prolog"].append(prolog_results)
        results["speedups"].append(speedup)

        # Print summary
        print(f"  Summary for {num_nodes} nodes:")
        print(
            f"    Original TaiatBuilder: {original_results['avg_time']:.4f}s avg, {original_results['reliability']:.1f}% reliable"
        )
        print(
            f"    Global Optimized Prolog: {prolog_results['avg_time']:.4f}s avg, {prolog_results['reliability']:.1f}% reliable"
        )
        if speedup != float("inf"):
            print(f"    Speedup: {speedup:.2f}x")
            if speedup > 1:
                print(f"    ✅ Prolog is {speedup:.2f}x faster")
            else:
                print(f"    ❌ TaiatBuilder is {1 / speedup:.2f}x faster")
        else:
            print(f"    Speedup: N/A (one or both failed)")
        print()

    # Print final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    successful_tests = [
        (i, speedup)
        for i, speedup in enumerate(results["speedups"])
        if speedup != float("inf")
    ]

    if successful_tests:
        avg_speedup = statistics.mean([speedup for _, speedup in successful_tests])
        max_speedup = max([speedup for _, speedup in successful_tests])
        min_speedup = min([speedup for _, speedup in successful_tests])

        print(f"Average speedup across all successful tests: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Minimum speedup: {min_speedup:.2f}x")

        # Find crossover point
        crossover_found = False
        for i, speedup in enumerate(results["speedups"]):
            if speedup != float("inf") and speedup > 1:
                print(f"✅ Crossover point found at {results['node_sizes'][i]} nodes")
                print(
                    f"   Prolog becomes faster than TaiatBuilder at {results['node_sizes'][i]} nodes"
                )
                crossover_found = True
                break

        if not crossover_found:
            print(
                "❌ No crossover point found - TaiatBuilder remains faster across all tested sizes"
            )

    # Print detailed results table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    print(
        f"{'Nodes':<6} {'TaiatBuilder (s)':<15} {'Prolog (s)':<12} {'Speedup':<10} {'Winner':<10}"
    )
    print("-" * 80)

    for i, num_nodes in enumerate(results["node_sizes"]):
        taiat_time = results["original_taiat_builder"][i]["avg_time"]
        prolog_time = results["global_optimized_prolog"][i]["avg_time"]
        speedup = results["speedups"][i]

        if taiat_time == float("inf"):
            taiat_str = "FAIL"
        else:
            taiat_str = f"{taiat_time:.4f}"

        if prolog_time == float("inf"):
            prolog_str = "FAIL"
        else:
            prolog_str = f"{prolog_time:.4f}"

        if speedup == float("inf"):
            speedup_str = "N/A"
            winner = "N/A"
        elif speedup > 1:
            speedup_str = f"{speedup:.2f}x"
            winner = "Prolog"
        else:
            speedup_str = f"{1 / speedup:.2f}x"
            winner = "Taiat"

        print(
            f"{num_nodes:<6} {taiat_str:<15} {prolog_str:<12} {speedup_str:<10} {winner:<10}"
        )

    print("=" * 80)

    return results


def run_performance_comparison_with_shuffle(
    node_sizes: List[int] = None, num_runs: int = 5
):
    """
    Run the performance comparison with shuffled nodes to test loading impact.

    Args:
        node_sizes: List of node sizes to test
        num_runs: Number of runs per test
    """
    if node_sizes is None:
        node_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("=" * 80)
    print(
        "GLOBAL OPTIMIZED PROLOG vs ORIGINAL TAITBUILDER PERFORMANCE COMPARISON (SHUFFLED)"
    )
    print("=" * 80)
    print(f"Testing node sizes: {node_sizes}")
    print(f"Runs per test: {num_runs}")
    print("Nodes are shuffled to test loading impact")
    print()

    # Set random seed for reproducible results
    random.seed(42)

    results = {
        "node_sizes": [],
        "original_taiat_builder": [],
        "global_optimized_prolog": [],
        "speedups": [],
    }

    for num_nodes in node_sizes:
        print(f"Testing {num_nodes} nodes (shuffled):")
        print("-" * 40)

        # Create test data with shuffled nodes
        node_set = create_performance_test_node_set(num_nodes, shuffle=True)

        # Create desired outputs (subset of outputs, like the original test)
        num_desired_outputs = min(3, num_nodes // 5 + 1)
        desired_outputs = []
        for i in range(num_desired_outputs):
            output_index = num_nodes - 1 - i
            if output_index >= 0:
                desired_outputs.append(
                    AgentData(
                        name=f"output_{output_index}",
                        parameters={},
                        description=f"Output from node {output_index}",
                        data=None,
                    )
                )

        # Test original TaiatBuilder
        original_results = test_original_taiat_builder(
            node_set, desired_outputs, num_runs
        )

        # Test global optimized Prolog
        prolog_results = test_global_optimized_prolog(
            node_set, desired_outputs, num_runs
        )

        # Calculate speedup
        if original_results["avg_time"] != float("inf") and prolog_results[
            "avg_time"
        ] != float("inf"):
            speedup = original_results["avg_time"] / prolog_results["avg_time"]
        else:
            speedup = float("inf")

        # Store results
        results["node_sizes"].append(num_nodes)
        results["original_taiat_builder"].append(original_results)
        results["global_optimized_prolog"].append(prolog_results)
        results["speedups"].append(speedup)

        # Print summary
        print(f"  Summary for {num_nodes} nodes (shuffled):")
        print(
            f"    Original TaiatBuilder: {original_results['avg_time']:.4f}s avg, {original_results['reliability']:.1f}% reliable"
        )
        print(
            f"    Global Optimized Prolog: {prolog_results['avg_time']:.4f}s avg, {prolog_results['reliability']:.1f}% reliable"
        )
        if speedup != float("inf"):
            print(f"    Speedup: {speedup:.2f}x")
            if speedup > 1:
                print(f"    ✅ Prolog is {speedup:.2f}x faster")
            else:
                print(f"    ❌ TaiatBuilder is {1 / speedup:.2f}x faster")
        else:
            print(f"    Speedup: N/A (one or both failed)")
        print()

    # Print final summary
    print("=" * 80)
    print("FINAL SUMMARY (SHUFFLED NODES)")
    print("=" * 80)

    successful_tests = [
        (i, speedup)
        for i, speedup in enumerate(results["speedups"])
        if speedup != float("inf")
    ]

    if successful_tests:
        avg_speedup = statistics.mean([speedup for _, speedup in successful_tests])
        max_speedup = max([speedup for _, speedup in successful_tests])
        min_speedup = min([speedup for _, speedup in successful_tests])

        print(f"Average speedup across all successful tests: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Minimum speedup: {min_speedup:.2f}x")

        # Find crossover point
        crossover_found = False
        for i, speedup in enumerate(results["speedups"]):
            if speedup != float("inf") and speedup > 1:
                print(f"✅ Crossover point found at {results['node_sizes'][i]} nodes")
                print(
                    f"   Prolog becomes faster than TaiatBuilder at {results['node_sizes'][i]} nodes"
                )
                crossover_found = True
                break

        if not crossover_found:
            print(
                "❌ No crossover point found - TaiatBuilder remains faster across all tested sizes"
            )

    # Print detailed results table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE (SHUFFLED NODES)")
    print("=" * 80)
    print(
        f"{'Nodes':<6} {'TaiatBuilder (s)':<15} {'Prolog (s)':<12} {'Speedup':<10} {'Winner':<10}"
    )
    print("-" * 80)

    for i, num_nodes in enumerate(results["node_sizes"]):
        taiat_time = results["original_taiat_builder"][i]["avg_time"]
        prolog_time = results["global_optimized_prolog"][i]["avg_time"]
        speedup = results["speedups"][i]

        if taiat_time == float("inf"):
            taiat_str = "FAIL"
        else:
            taiat_str = f"{taiat_time:.4f}"

        if prolog_time == float("inf"):
            prolog_str = "FAIL"
        else:
            prolog_str = f"{prolog_time:.4f}"

        if speedup == float("inf"):
            speedup_str = "N/A"
            winner = "N/A"
        elif speedup > 1:
            speedup_str = f"{speedup:.2f}x"
            winner = "Prolog"
        else:
            speedup_str = f"{1 / speedup:.2f}x"
            winner = "Taiat"

        print(
            f"{num_nodes:<6} {taiat_str:<15} {prolog_str:<12} {speedup_str:<10} {winner:<10}"
        )

    print("=" * 80)


if __name__ == "__main__":
    # Run the original comparison
    run_performance_comparison()

    print("\n" + "=" * 80)
    print("NOW TESTING WITH SHUFFLED NODES TO CHECK LOADING IMPACT")
    print("=" * 80)
    # Run the shuffled comparison
    run_performance_comparison_with_shuffle()
