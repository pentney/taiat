#!/usr/bin/env python3
"""
Final Performance Comparison: Haskell vs Prolog Path Planners

This script compares the performance of the Haskell and Prolog implementations
of the Taiat path planner, staying within Prolog's node limits for fair comparison.
"""

import time
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import taiat modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from haskell.haskell_interface import HaskellPathPlanner
from prolog.optimized_prolog_interface import OptimizedPrologPathPlanner

# Mock data structures for testing
class AgentData:
    def __init__(self, name: str, parameters: Dict[str, str] = None, description: str = "", data: Any = None):
        self.name = name
        self.parameters = parameters or {}
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


def create_simple_test_case() -> tuple:
    """Create a simple linear pipeline test case."""
    input_a = AgentData("input_a", {}, "Input A")
    output_b = AgentData("output_b", {}, "Output B")
    output_c = AgentData("output_c", {}, "Output C")
    final_output = AgentData("final_output", {}, "Final Output")
    
    node_a = AgentGraphNode("node_a", "Process A", [input_a], [output_b])
    node_b = AgentGraphNode("node_b", "Process B", [output_b], [output_c])
    node_c = AgentGraphNode("node_c", "Process C", [output_c], [final_output])
    
    node_set = AgentGraphNodeSet([node_a, node_b, node_c])
    desired_outputs = [final_output]
    
    return node_set, desired_outputs


def create_complex_test_case() -> tuple:
    """Create a complex multi-path dependency test case."""
    input_data = AgentData("input_data", {}, "Input Data")
    processed_data = AgentData("processed_data", {"format": "json"}, "Processed Data")
    analyzed_data = AgentData("analyzed_data", {"format": "json", "analysis": "basic"}, "Analyzed Data")
    report = AgentData("report", {"format": "pdf"}, "Report")
    summary = AgentData("summary", {"format": "text"}, "Summary")
    
    processor = AgentGraphNode("processor", "Data Processor", [input_data], [processed_data])
    analyzer = AgentGraphNode("analyzer", "Data Analyzer", [processed_data], [analyzed_data])
    reporter = AgentGraphNode("reporter", "Report Generator", [analyzed_data], [report])
    summarizer = AgentGraphNode("summarizer", "Summary Generator", [analyzed_data], [summary])
    
    node_set = AgentGraphNodeSet([processor, analyzer, reporter, summarizer])
    desired_outputs = [report, summary]
    
    return node_set, desired_outputs


def create_medium_test_case(node_count: int = 20) -> tuple:
    """Create a medium-sized test case within Prolog's limits."""
    nodes = []
    desired_outputs = []
    
    # Create input node
    input_data = AgentData("input_data", {}, "Input Data")
    nodes.append(AgentGraphNode("input_node", "Input Node", [], [input_data]))
    
    # Create processing nodes
    for i in range(node_count):
        input_name = f"data_{i-1}" if i > 0 else "input_data"
        output_name = f"data_{i}"
        
        input_agent = AgentData(input_name, {}, f"Data {i-1}" if i > 0 else "Input Data")
        output_agent = AgentData(output_name, {}, f"Data {i}")
        
        node = AgentGraphNode(f"node_{i}", f"Process {i}", [input_agent], [output_agent])
        nodes.append(node)
        
        if i == node_count - 1:
            desired_outputs.append(output_agent)
    
    # Add some parallel processing nodes
    for i in range(node_count // 5):
        base_data = AgentData(f"data_{i*5}", {}, f"Base Data {i}")
        parallel_output = AgentData(f"parallel_output_{i}", {}, f"Parallel Output {i}")
        
        parallel_node = AgentGraphNode(f"parallel_node_{i}", f"Parallel Process {i}", [base_data], [parallel_output])
        nodes.append(parallel_node)
        desired_outputs.append(parallel_output)
    
    node_set = AgentGraphNodeSet(nodes)
    return node_set, desired_outputs


def create_large_test_case(node_count: int = 100) -> tuple:
    """Create a large test case within Prolog's limits."""
    nodes = []
    desired_outputs = []
    
    # Create input node
    input_data = AgentData("input_data", {}, "Input Data")
    nodes.append(AgentGraphNode("input_node", "Input Node", [], [input_data]))
    
    # Create processing nodes
    for i in range(node_count):
        input_name = f"data_{i-1}" if i > 0 else "input_data"
        output_name = f"data_{i}"
        
        input_agent = AgentData(input_name, {}, f"Data {i-1}" if i > 0 else "Input Data")
        output_agent = AgentData(output_name, {}, f"Data {i}")
        
        node = AgentGraphNode(f"node_{i}", f"Process {i}", [input_agent], [output_agent])
        nodes.append(node)
        
        if i == node_count - 1:
            desired_outputs.append(output_agent)
    
    # Add some parallel processing nodes
    for i in range(node_count // 10):
        base_data = AgentData(f"data_{i*10}", {}, f"Base Data {i}")
        parallel_output = AgentData(f"parallel_output_{i}", {}, f"Parallel Output {i}")
        
        parallel_node = AgentGraphNode(f"parallel_node_{i}", f"Parallel Process {i}", [base_data], [parallel_output])
        nodes.append(parallel_node)
        desired_outputs.append(parallel_output)
    
    node_set = AgentGraphNodeSet(nodes)
    return node_set, desired_outputs


def run_single_test(test_name: str, node_set: AgentGraphNodeSet, desired_outputs: List[AgentData], iterations: int = 5) -> Dict[str, Any]:
    """Run a single performance test with multiple iterations."""
    print(f"\nRunning {test_name}...")
    print(f"Nodes: {len(node_set.nodes)}, Desired outputs: {len(desired_outputs)}, Iterations: {iterations}")
    
    try:
        # Initialize planners
        haskell_planner = HaskellPathPlanner()
        prolog_planner = OptimizedPrologPathPlanner()
        
        # Test Haskell
        haskell_times = []
        haskell_result = None
        haskell_error = None
        
        if hasattr(haskell_planner, 'available') and haskell_planner.available:
            try:
                for i in range(iterations):
                    start_time = time.time()
                    result = haskell_planner.plan_path(node_set, desired_outputs)
                    end_time = time.time()
                    haskell_times.append(end_time - start_time)
                    if i == 0:  # Keep first result for comparison
                        haskell_result = result
                
                avg_haskell_time = sum(haskell_times) / len(haskell_times)
                print(f"  Haskell: {avg_haskell_time:.6f}s avg ({len(haskell_times)} runs) - {len(haskell_result)} nodes")
            except Exception as e:
                haskell_error = str(e)
                print(f"  Haskell: ERROR - {haskell_error}")
        else:
            print("  Haskell: Not available")
        
        # Test Prolog
        prolog_times = []
        prolog_result = None
        prolog_error = None
        
        if hasattr(prolog_planner, 'compiled_program') and prolog_planner.compiled_program is not None:
            try:
                for i in range(iterations):
                    start_time = time.time()
                    result = prolog_planner.plan_path(node_set, desired_outputs)
                    end_time = time.time()
                    prolog_times.append(end_time - start_time)
                    if i == 0:  # Keep first result for comparison
                        prolog_result = result
                
                avg_prolog_time = sum(prolog_times) / len(prolog_times)
                print(f"  Prolog:  {avg_prolog_time:.6f}s avg ({len(prolog_times)} runs) - {len(prolog_result)} nodes")
            except Exception as e:
                prolog_error = str(e)
                print(f"  Prolog:  ERROR - {prolog_error}")
        else:
            print("  Prolog: Not available")
        
        # Calculate improvement
        improvement = None
        if haskell_times and prolog_times:
            avg_haskell = sum(haskell_times) / len(haskell_times)
            avg_prolog = sum(prolog_times) / len(prolog_times)
            improvement = ((avg_prolog - avg_haskell) / avg_prolog) * 100
            print(f"  Improvement: {improvement:.2f}%")
        
        # Check if results match
        results_match = False
        if haskell_result and prolog_result:
            results_match = haskell_result == prolog_result
            print(f"  Results match: {results_match}")
        
        return {
            "test_name": test_name,
            "node_count": len(node_set.nodes),
            "output_count": len(desired_outputs),
            "iterations": iterations,
            "haskell": {
                "available": hasattr(haskell_planner, 'available') and haskell_planner.available,
                "times": haskell_times,
                "avg_time": sum(haskell_times) / len(haskell_times) if haskell_times else None,
                "result_length": len(haskell_result) if haskell_result else None,
                "error": haskell_error
            },
            "prolog": {
                "available": hasattr(prolog_planner, 'compiled_program') and prolog_planner.compiled_program is not None,
                "times": prolog_times,
                "avg_time": sum(prolog_times) / len(prolog_times) if prolog_times else None,
                "result_length": len(prolog_result) if prolog_result else None,
                "error": prolog_error
            },
            "improvement": improvement,
            "results_match": results_match
        }
        
    except Exception as e:
        print(f"  Test failed: {e}")
        return {
            "test_name": test_name,
            "error": str(e)
        }


def run_performance_suite() -> List[Dict[str, Any]]:
    """Run the complete performance test suite."""
    print("Taiat Path Planner Performance Comparison")
    print("=========================================")
    print("Comparing Haskell vs Prolog implementations")
    print("(Staying within Prolog's 200-node limit for fair comparison)")
    
    results = []
    
    # Test 1: Simple linear pipeline
    node_set, desired_outputs = create_simple_test_case()
    results.append(run_single_test("Simple Linear Pipeline", node_set, desired_outputs, iterations=10))
    
    # Test 2: Complex multi-path dependencies
    node_set, desired_outputs = create_complex_test_case()
    results.append(run_single_test("Complex Multi-Path Dependencies", node_set, desired_outputs, iterations=10))
    
    # Test 3: Medium test case
    node_set, desired_outputs = create_medium_test_case(20)
    results.append(run_single_test("Medium Test Case (20 nodes)", node_set, desired_outputs, iterations=5))
    
    # Test 4: Large test case (within Prolog limits)
    node_set, desired_outputs = create_large_test_case(100)
    results.append(run_single_test("Large Test Case (100 nodes)", node_set, desired_outputs, iterations=3))
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of the performance test results."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    successful_tests = [r for r in results if "error" not in r and r.get("haskell", {}).get("avg_time") and r.get("prolog", {}).get("avg_time")]
    
    if not successful_tests:
        print("No successful performance comparisons available.")
        return
    
    print(f"Successful comparisons: {len(successful_tests)}/{len(results)}")
    print()
    
    # Calculate average improvements
    improvements = [r["improvement"] for r in successful_tests if r["improvement"] is not None]
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"Average performance improvement: {avg_improvement:.2f}%")
        print(f"Best improvement: {max(improvements):.2f}%")
        print(f"Worst improvement: {min(improvements):.2f}%")
        
        # Count wins for each implementation
        haskell_wins = sum(1 for imp in improvements if imp > 0)
        prolog_wins = sum(1 for imp in improvements if imp < 0)
        ties = sum(1 for imp in improvements if imp == 0)
        
        print(f"\nWins: Haskell {haskell_wins}, Prolog {prolog_wins}, Ties {ties}")
    
    print()
    print("Detailed Results:")
    print("-" * 80)
    
    for result in results:
        if "error" in result:
            print(f"{result['test_name']}: ERROR - {result['error']}")
            continue
        
        haskell_info = result["haskell"]
        prolog_info = result["prolog"]
        
        print(f"{result['test_name']}:")
        print(f"  Nodes: {result['node_count']}, Outputs: {result['output_count']}, Iterations: {result['iterations']}")
        
        if haskell_info["available"] and haskell_info["avg_time"]:
            print(f"  Haskell: {haskell_info['avg_time']:.6f}s avg - {haskell_info['result_length']} nodes")
        
        if prolog_info["available"] and prolog_info["avg_time"]:
            print(f"  Prolog:  {prolog_info['avg_time']:.6f}s avg - {prolog_info['result_length']} nodes")
        
        if result["improvement"] is not None:
            print(f"  Improvement: {result['improvement']:.2f}%")
        
        if result["results_match"] is not None:
            print(f"  Results match: {result['results_match']}")
        
        print()


def save_results(results: List[Dict[str, Any]], filename: str = "final_performance_results.json"):
    """Save test results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")


def main():
    """Main function to run the performance comparison."""
    try:
        results = run_performance_suite()
        print_summary(results)
        save_results(results)
        
        # Exit with error code if any tests failed
        failed_tests = [r for r in results if "error" in r]
        if failed_tests:
            print(f"\nWARNING: {len(failed_tests)} tests failed!")
            sys.exit(1)
        else:
            print("\nAll tests completed successfully!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 