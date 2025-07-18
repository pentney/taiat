"""
Minimal test for path planning without parameter constraints.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
from haskell.haskell_interface import HaskellPathPlanner


def minimal_test():
    """Test basic path planning without parameter constraints."""
    print("Testing minimal path planning...")
    
    # Create a simple node with no parameter constraints
    node = AgentGraphNode(
        name="simple_node",
        description="Simple node",
        inputs=[],
        outputs=[
            AgentData(
                name="output",
                parameters={},  # No parameters
                description="",
                data=None,
            )
        ],
    )
    node_set = AgentGraphNodeSet(nodes=[node])
    desired_outputs = [
        AgentData(
            name="output",
            parameters={},  # No parameters
            description="",
            data=None,
        )
    ]
    
    print(f"Node: {node.name}")
    print(f"Node outputs: {node.outputs}")
    print(f"Desired outputs: {desired_outputs}")
    
    try:
        planner = HaskellPathPlanner()
        if not planner.available:
            print("❌ Haskell planner not available")
            return
        
        print("✅ Haskell planner available")
        
        # Test the plan_path function
        path = planner.plan_path(node_set, desired_outputs)
        print(f"Haskell path result: {path}")
        
        # Test validation
        validation_result = planner.validate_outputs(node_set, desired_outputs)
        print(f"Haskell validation result: {validation_result}")
        
        # Test with Prolog for comparison
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
    minimal_test() 