"""
Simple test for the AgentRegistry functionality.
"""

import unittest
from unittest.mock import Mock
from graph_architect import AgentRegistry, TaiatGraphArchitect
from base import AgentData, AgentGraphNode


class TestAgentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()

    def test_register_and_get(self):
        """Test registering and retrieving functions."""

        def test_function():
            return "test"

        self.registry.register("test_func", test_function)
        retrieved_func = self.registry.get("test_func")

        self.assertEqual(retrieved_func, test_function)
        self.assertEqual(retrieved_func(), "test")

    def test_get_nonexistent(self):
        """Test getting a function that doesn't exist."""
        func = self.registry.get("nonexistent")
        self.assertIsNone(func)

    def test_list_registered(self):
        """Test listing registered functions."""

        def func1():
            pass

        def func2():
            pass

        self.registry.register("func1", func1)
        self.registry.register("func2", func2)

        registered = self.registry.list_registered()
        self.assertIn("func1", registered)
        self.assertIn("func2", registered)
        self.assertEqual(len(registered), 2)

    def test_clear(self):
        """Test clearing the registry."""

        def test_func():
            pass

        self.registry.register("test", test_func)
        self.assertEqual(len(self.registry.list_registered()), 1)

        self.registry.clear()
        self.assertEqual(len(self.registry.list_registered()), 0)


class TestTaiatGraphArchitect(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.mock_llm = Mock()
        self.architect = TaiatGraphArchitect(
            llm=self.mock_llm, agent_registry=self.registry, verbose=False
        )

    def test_resolve_function_names(self):
        """Test resolving function names to actual functions."""

        def test_agent(state):
            return state

        self.registry.register("test_agent", test_agent)

        # Create a mock node with function name as string
        node = AgentGraphNode(
            name="test_node",
            description="Test node",
            function="test_agent",
            inputs=[],
            outputs=[],
        )

        resolved_nodes = self.architect._resolve_function_names([node])

        self.assertEqual(len(resolved_nodes), 1)
        self.assertEqual(resolved_nodes[0].function, test_agent)

    def test_resolve_function_names_missing(self):
        """Test error when function name is not found."""
        node = AgentGraphNode(
            name="test_node",
            description="Test node",
            function="missing_function",
            inputs=[],
            outputs=[],
        )

        with self.assertRaises(ValueError) as context:
            self.architect._resolve_function_names([node])

        self.assertIn("missing_function", str(context.exception))
        self.assertIn("Available functions", str(context.exception))


if __name__ == "__main__":
    unittest.main()
