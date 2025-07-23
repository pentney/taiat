#!/usr/bin/env python3
"""
Test script for TaiatGraphArchitect

This script demonstrates how to use the TaiatGraphArchitect to build an AgentGraphNodeSet
from a textual description of a workflow.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from taiat.graph_architect import TaiatGraphArchitect, AgentRegistry
from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet
from langchain_openai import ChatOpenAI


def create_sample_functions():
    """
    Create sample functions for the agent registry.
    These are placeholder functions that would normally contain actual implementation.
    """
    def load_dataset(state):
        """Load a UCI dataset from Kaggle."""
        print("Loading dataset...")
        # Simulate loading dataset
        state["data"]["dataset"] = {"name": "diabetes", "features": 8, "samples": 768}
        return state
    
    def logistic_regression(state):
        """Train a logistic regression model."""
        print("Training logistic regression model...")
        # Simulate training
        state["data"]["model"] = {"type": "logistic_regression", "accuracy": 0.85}
        return state
    
    def random_forest(state):
        """Train a random forest model."""
        print("Training random forest model...")
        # Simulate training
        state["data"]["model"] = {"type": "random_forest", "accuracy": 0.88}
        return state
    
    def nearest_neighbor(state):
        """Train a nearest neighbor model."""
        print("Training nearest neighbor model...")
        # Simulate training
        state["data"]["model"] = {"type": "nearest_neighbor", "accuracy": 0.82}
        return state
    
    def clustering(state):
        """Perform clustering analysis."""
        print("Performing clustering analysis...")
        # Simulate clustering
        state["data"]["clusters"] = {"n_clusters": 3, "silhouette_score": 0.65}
        return state
    
    def generate_report(state):
        """Generate a report about training the ML model."""
        print("Generating report...")
        # Simulate report generation
        state["data"]["report"] = {"summary": "Model training completed successfully"}
        return state
    
    def results_analysis(state):
        """Analyze the results of training."""
        print("Analyzing results...")
        # Simulate analysis
        state["data"]["analysis"] = {"insights": ["Model performs well", "Good feature importance"]}
        return state
    
    return {
        "load_dataset": load_dataset,
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "nearest_neighbor": nearest_neighbor,
        "clustering": clustering,
        "generate_report": generate_report,
        "results_analysis": results_analysis,
    }


def print_agent_data(agent_data_list):
    """Print AgentData objects in a readable format."""
    print("\n" + "="*60)
    print("AGENT DATA OBJECTS")
    print("="*60)
    
    for i, agent_data in enumerate(agent_data_list, 1):
        print(f"\n{i}. {agent_data.name}")
        print(f"   Description: {agent_data.description}")
        print(f"   Parameters: {agent_data.parameters}")
        print(f"   ID: {agent_data.id}")


def print_agent_graph_nodes(agent_graph_nodes):
    """Print AgentGraphNode objects in a readable format."""
    print("\n" + "="*60)
    print("AGENT GRAPH NODES")
    print("="*60)
    
    for i, node in enumerate(agent_graph_nodes, 1):
        print(f"\n{i}. {node.name}")
        print(f"   Description: {node.description}")
        print(f"   Function: {node.function.__name__ if callable(node.function) else node.function}")
        
        print(f"   Inputs:")
        for j, input_data in enumerate(node.inputs, 1):
            print(f"     {j}. {input_data.name} (params: {input_data.parameters})")
        
        print(f"   Outputs:")
        for j, output_data in enumerate(node.outputs, 1):
            print(f"     {j}. {output_data.name} (params: {output_data.parameters})")


def print_agent_graph_node_set(node_set):
    """Print the complete AgentGraphNodeSet."""
    print("\n" + "="*60)
    print("COMPLETE AGENT GRAPH NODE SET")
    print("="*60)
    
    print(f"\nTotal nodes: {len(node_set.nodes)}")
    print(f"Node names: {[node.name for node in node_set.nodes]}")
    
    # Show the JSON representation
    print(f"\nJSON representation:")
    print(json.dumps(node_set.model_dump(), indent=2, default=str))


def main():
    """Main test function."""
    print("Testing TaiatGraphArchitect")
    print("="*60)
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run this test")
        return
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = ChatOpenAI(model="gpt-4")
    
    # Create agent registry and register functions
    print("Setting up agent registry...")
    registry = AgentRegistry()
    sample_functions = create_sample_functions()
    
    for name, func in sample_functions.items():
        registry.register(name, func)
    
    print(f"Registered functions: {registry.list_registered()}")
    
    # Create the graph architect
    print("Creating TaiatGraphArchitect...")
    architect = TaiatGraphArchitect(
        llm=llm,
        agent_registry=registry,
        verbose=True,
        llm_explanation=True
    )
    
    # Read the workflow description
    print("Reading workflow description...")
    workflow_file = Path(__file__).parent / "src" / "examples" / "ml_workflow_description.md"
    
    if not workflow_file.exists():
        print(f"Error: Workflow description file not found at {workflow_file}")
        return
    
    with open(workflow_file, 'r') as f:
        workflow_description = f.read()
    
    print(f"Workflow description length: {len(workflow_description)} characters")
    print(f"Workflow description:\n{workflow_description}")
    
    # Build the AgentGraphNodeSet
    print("\nBuilding AgentGraphNodeSet...")
    try:
        node_set = architect.build(workflow_description)
        
        # Print the results
        print("\n" + "="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        
        # Extract agent data and nodes for display
        agent_data_list = []
        agent_graph_nodes = []
        
        # Collect all unique AgentData objects from all nodes
        seen_data = set()
        for node in node_set.nodes:
            for data in node.inputs + node.outputs:
                if data.id not in seen_data:
                    agent_data_list.append(data)
                    seen_data.add(data.id)
            agent_graph_nodes.append(node)
        
        # Print results
        print_agent_data(agent_data_list)
        print_agent_graph_nodes(agent_graph_nodes)
        print_agent_graph_node_set(node_set)
        
        # Test the node set with the path planner
        print("\n" + "="*60)
        print("TESTING WITH PATH PLANNER")
        print("="*60)
        
        try:
            from prolog.taiat_path_planner import plan_taiat_path_global
            
            # Test with a simple query
            test_outputs = [
                AgentData(name="model", parameters={"type": "logistic_regression"}, description="Trained model")
            ]
            
            print(f"Testing path planning for outputs: {[f'{o.name}({o.parameters})' for o in test_outputs]}")
            
            execution_path = plan_taiat_path_global(node_set, test_outputs)
            
            if execution_path:
                print(f"Execution path: {execution_path}")
            else:
                print("No valid execution path found")
                
        except ImportError as e:
            print(f"Path planner not available: {e}")
        except Exception as e:
            print(f"Path planning failed: {e}")
        
    except Exception as e:
        print(f"\n" + "="*60)
        print("BUILD FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 