"""
Test for TaiatExecutor - custom execution engine without langgraph dependency.
"""

import pytest
from typing import Dict, Any

from taiat.executor import TaiatExecutor
from taiat.base import (
    State,
    AgentGraphNode,
    AgentData,
    TaiatQuery,
)


def test_simple_execution():
    """Test basic execution of a single node."""

    # Create a simple test node
    def test_agent(state: State) -> State:
        state["data"]["test_output"] = "test_value"
        return state

    test_node = AgentGraphNode(
        name="test_agent",
        description="A test agent",
        function=test_agent,
        inputs=[],
        outputs=[AgentData(name="test_output", parameters={})],
    )

    # Create initial state
    query = TaiatQuery(query="test query")
    state = State(query=query, data={})

    # Execute
    executor = TaiatExecutor(verbose=True)
    result_state = executor.execute([test_node], state)

    # Verify execution
    assert result_state["data"]["test_output"] == "test_value"
    assert result_state["query"].status == "success"
    assert len(executor.get_execution_path()) == 1
    assert executor.get_execution_path()[0].name == "test_agent"


def test_dependency_execution():
    """Test execution with dependencies between nodes."""

    # Create nodes with dependencies
    def producer_agent(state: State) -> State:
        state["data"]["intermediate_data"] = "produced_value"
        return state

    def consumer_agent(state: State) -> State:
        input_data = state["data"].get("intermediate_data")
        state["data"]["final_output"] = f"processed_{input_data}"
        return state

    producer_node = AgentGraphNode(
        name="producer",
        description="Produces intermediate data",
        function=producer_agent,
        inputs=[],
        outputs=[AgentData(name="intermediate_data", parameters={})],
    )

    consumer_node = AgentGraphNode(
        name="consumer",
        description="Consumes intermediate data",
        function=consumer_agent,
        inputs=[AgentData(name="intermediate_data", parameters={})],
        outputs=[AgentData(name="final_output", parameters={})],
    )

    # Create initial state
    query = TaiatQuery(query="test dependency execution")
    state = State(query=query, data={})

    # Execute
    executor = TaiatExecutor(verbose=True)
    result_state = executor.execute([producer_node, consumer_node], state)

    # Verify execution
    assert result_state["data"]["intermediate_data"] == "produced_value"
    assert result_state["data"]["final_output"] == "processed_produced_value"
    assert result_state["query"].status == "success"

    # Verify execution order
    execution_path = executor.get_execution_path()
    assert len(execution_path) == 2
    assert execution_path[0].name == "producer"
    assert execution_path[1].name == "consumer"


def test_error_handling():
    """Test error handling during execution."""

    def failing_agent(state: State) -> State:
        raise ValueError("Test error")

    failing_node = AgentGraphNode(
        name="failing_agent",
        description="An agent that fails",
        function=failing_agent,
        inputs=[],
        outputs=[AgentData(name="output", parameters={})],
    )

    # Create initial state
    query = TaiatQuery(query="test error handling")
    state = State(query=query, data={})

    # Execute
    executor = TaiatExecutor(verbose=True)
    result_state = executor.execute([failing_node], state)

    # Verify error handling
    assert result_state["query"].status == "error"
    assert "Error executing node failing_agent" in result_state["query"].error
    assert len(executor.get_execution_stats()["errors"]) == 1


def test_empty_nodes():
    """Test execution with empty node list."""

    query = TaiatQuery(query="test empty nodes")
    state = State(query=query, data={"initial": "value"})

    executor = TaiatExecutor(verbose=True)
    result_state = executor.execute([], state)

    # Should return state unchanged
    assert result_state["data"]["initial"] == "value"
    assert len(executor.get_execution_path()) == 0


def test_execution_stats():
    """Test that execution statistics are properly tracked."""

    def slow_agent(state: State) -> State:
        import time

        time.sleep(0.1)  # Simulate some work
        state["data"]["result"] = "done"
        return state

    test_node = AgentGraphNode(
        name="slow_agent",
        description="A slow agent",
        function=slow_agent,
        inputs=[],
        outputs=[AgentData(name="result", parameters={})],
    )

    query = TaiatQuery(query="test stats")
    state = State(query=query, data={})

    executor = TaiatExecutor(verbose=True)
    result_state = executor.execute([test_node], state)

    stats = executor.get_execution_stats()
    assert stats["nodes_executed"] == 1
    assert stats["dependencies_resolved"] == 1
    assert stats["execution_time"] > 0
    assert len(stats["errors"]) == 0


if __name__ == "__main__":
    # Run tests
    test_simple_execution()
    test_dependency_execution()
    test_error_handling()
    test_empty_nodes()
    test_execution_stats()
    print("All tests passed!")
