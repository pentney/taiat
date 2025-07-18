"""
Debug script for parameter matching in Haskell engine.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
from haskell.haskell_interface import HaskellPathPlanner


def debug_parameter_matching():
    """Debug the parameter matching issue."""
    print("Debugging parameter matching...")
    
    # Create the test case from the failing test
    node = AgentGraphNode(
        name="model_node",
        description="Provides a model",
        inputs=[],
        outputs=[
            AgentData(
                name="model",
                parameters={"type": "logistic_regression", "version": "v1"},
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="model",
            parameters={"type": "logistic_regression"},
            description="",
            data=None,
        )
    ]
    
    print(f"Node outputs: {node.outputs}")
    print(f"Desired outputs: {desired_outputs}")
    print(f"Node output parameters: {node.outputs[0].parameters}")
    print(f"Desired output parameters: {desired_outputs[0].parameters}")
    
    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return
        
        print("✅ Haskell planner available")
        
        # Test the plan_path function
        path = planner.plan_path(node_set, desired_outputs)
        print(f"Path result: {path}")
        
        # Let's also test the validateOutputs function
        validation_result = planner.validate_outputs(node_set, desired_outputs)
        print(f"Validation result: {validation_result}")
        
        # Test available outputs
        available = planner.get_available_outputs(node_set)
        print(f"Available outputs: {available}")
        
        # Test circular dependencies
        has_circular = planner.has_circular_dependencies(node_set)
        print(f"Has circular dependencies: {has_circular}")
        
        # Let's also test with the Prolog planner for comparison
        try:
            from prolog.taiat_path_planner import plan_taiat_path_global
            prolog_path = plan_taiat_path_global(node_set, desired_outputs)
            print(f"Prolog path result: {prolog_path}")
        except Exception as e:
            print(f"Prolog test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_parameter_matching() 