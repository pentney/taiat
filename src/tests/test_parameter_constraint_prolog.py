"""
Test to verify parameter constraint implementation in Prolog engine.
"""

import sys
import os
from pathlib import Path

# Add the taiat package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from taiat.base import AgentGraphNode, AgentGraphNodeSet, AgentData
from prolog.taiat_path_planner import plan_taiat_path_global


def test_parameter_constraint_prolog():
    """Test that input parameter constraints are properly enforced in Prolog."""
    print("Testing parameter constraint enforcement in Prolog...")
    
    # Create a producer that outputs X with parameters {"A": "B", "C": "D"}
    producer = AgentGraphNode(
        name="producer",
        description="Produces output X with specific parameters",
        inputs=[],
        outputs=[
            AgentData(
                name="X",
                parameters={"A": "B", "C": "D"},
                description="Output X with parameters A:B and C:D",
                data=None,
            )
        ],
    )
    
    # Create a consumer that requires input X with parameter {"A": "B"}
    # This should match because the output has A:B
    consumer_correct = AgentGraphNode(
        name="consumer_correct",
        description="Consumes input X with parameter A:B (should match)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},
                description="Input X requiring parameter A:B",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )
    
    # Create a consumer that requires input X with parameter {"A": "C"}
    # This should NOT match because the output has A:B, not A:C
    consumer_incorrect = AgentGraphNode(
        name="consumer_incorrect",
        description="Consumes input X with parameter A:C (should NOT match)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "C"},
                description="Input X requiring parameter A:C",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )
    
    # Test case 1: Correct parameter constraint (should succeed)
    print("\n--- Test Case 1: Correct Parameter Constraint ---")
    node_set_1 = AgentGraphNodeSet(nodes=[producer, consumer_correct])
    desired_outputs_1 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]
    
    try:
        path_1 = plan_taiat_path_global(node_set_1, desired_outputs_1)
        print(f"Path result: {path_1}")
        
        if path_1 == ["producer", "consumer_correct"]:
            print("✅ Correct parameter constraint test PASSED")
        else:
            print(f"❌ Correct parameter constraint test FAILED: {path_1}")
            
    except Exception as e:
        print(f"❌ Error in correct parameter test: {e}")
    
    # Test case 2: Incorrect parameter constraint (should fail)
    print("\n--- Test Case 2: Incorrect Parameter Constraint ---")
    node_set_2 = AgentGraphNodeSet(nodes=[producer, consumer_incorrect])
    desired_outputs_2 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]
    
    try:
        path_2 = plan_taiat_path_global(node_set_2, desired_outputs_2)
        print(f"Path result: {path_2}")
        
        if path_2 is None or path_2 == []:
            print("✅ Incorrect parameter constraint test PASSED (correctly rejected)")
        else:
            print(f"❌ Incorrect parameter constraint test FAILED: {path_2}")
            
    except Exception as e:
        print(f"❌ Error in incorrect parameter test: {e}")
    
    # Test case 3: Additional parameters in output (should succeed)
    print("\n--- Test Case 3: Additional Parameters in Output ---")
    # Create a consumer that requires input X with parameter {"A": "B"}
    # The producer outputs {"A": "B", "C": "D"}, so this should match
    consumer_additional = AgentGraphNode(
        name="consumer_additional",
        description="Consumes input X with parameter A:B (output has additional C:D)",
        inputs=[
            AgentData(
                name="X",
                parameters={"A": "B"},
                description="Input X requiring parameter A:B",
                data=None,
            )
        ],
        outputs=[
            AgentData(
                name="final_output",
                parameters={},
                description="Final output",
                data=None,
            )
        ],
    )
    
    node_set_3 = AgentGraphNodeSet(nodes=[producer, consumer_additional])
    desired_outputs_3 = [
        AgentData(
            name="final_output",
            parameters={},
            description="Final output",
            data=None,
        )
    ]
    
    try:
        path_3 = plan_taiat_path_global(node_set_3, desired_outputs_3)
        print(f"Path result: {path_3}")
        
        if path_3 == ["producer", "consumer_additional"]:
            print("✅ Additional parameters test PASSED")
        else:
            print(f"❌ Additional parameters test FAILED: {path_3}")
            
    except Exception as e:
        print(f"❌ Error in additional parameters test: {e}")


if __name__ == "__main__":
    test_parameter_constraint_prolog() 