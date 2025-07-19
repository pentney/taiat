"""
Tests for TaiatQuery parameters and state_info functionality.
"""

import unittest
from taiat.base import TaiatQuery


class TestTaiatQueryParameters(unittest.TestCase):
    """Test cases for TaiatQuery parameters and state_info functionality."""
    
    def test_parameters_basic(self):
        """Test basic parameter functionality."""
        query = TaiatQuery(
            query="test query",
            parameters={"key1": "value1", "key2": 42}
        )
        
        self.assertEqual(query.get_parameter("key1"), "value1")
        self.assertEqual(query.get_parameter("key2"), 42)
        self.assertEqual(query.get_parameter("nonexistent", "default"), "default")
        self.assertTrue(query.has_parameter("key1"))
        self.assertFalse(query.has_parameter("nonexistent"))
    
    def test_state_info_basic(self):
        """Test basic state_info functionality."""
        query = TaiatQuery(
            query="test query",
            state_info={"user_id": "user123", "session_id": "session456"}
        )
        
        self.assertEqual(query.get_state_info("user_id"), "user123")
        self.assertEqual(query.get_state_info("session_id"), "session456")
        self.assertEqual(query.get_state_info("nonexistent", "default"), "default")
        self.assertTrue(query.has_state_info("user_id"))
        self.assertFalse(query.has_state_info("nonexistent"))
    
    def test_get_all_methods(self):
        """Test get_all_parameters and get_all_state_info methods."""
        parameters = {"param1": "value1", "param2": "value2"}
        state_info = {"state1": "info1", "state2": "info2"}
        
        query = TaiatQuery(
            query="test query",
            parameters=parameters,
            state_info=state_info
        )
        
        # Test that we get copies, not references
        all_params = query.get_all_parameters()
        all_state = query.get_all_state_info()
        
        self.assertEqual(all_params, parameters)
        self.assertEqual(all_state, state_info)
        
        # Modify the returned dicts to ensure they're copies
        all_params["new_param"] = "new_value"
        all_state["new_state"] = "new_info"
        
        # Original query should be unchanged
        self.assertNotIn("new_param", query.parameters)
        self.assertNotIn("new_state", query.state_info)
    
    def test_db_serialization(self):
        """Test that parameters and state_info are properly serialized for database."""
        query = TaiatQuery(
            query="test query",
            parameters={"db_param": "db_value"},
            state_info={"db_state": "db_info"}
        )
        
        db_dict = query.as_db_dict()
        
        self.assertIn("parameters", db_dict)
        self.assertIn("state_info", db_dict)
        self.assertEqual(db_dict["parameters"], {"db_param": "db_value"})
        self.assertEqual(db_dict["state_info"], {"db_state": "db_info"})
    
    def test_db_deserialization(self):
        """Test that parameters and state_info are properly deserialized from database."""
        db_dict = {
            "query": "test query",
            "inferred_goal_output": None,
            "status": None,
            "error": "",
            "path": [],
            "visualize_graph": False,
            "parameters": {"restored_param": "restored_value"},
            "state_info": {"restored_state": "restored_info"}
        }
        
        query = TaiatQuery.from_db_dict(db_dict)
        
        self.assertEqual(query.get_parameter("restored_param"), "restored_value")
        self.assertEqual(query.get_state_info("restored_state"), "restored_info")
    
    def test_default_values(self):
        """Test that default values work correctly."""
        query = TaiatQuery(query="test query")
        
        # Should have empty dicts by default
        self.assertEqual(query.parameters, {})
        self.assertEqual(query.state_info, {})
        
        # Should return default values for missing keys
        self.assertEqual(query.get_parameter("missing", "default"), "default")
        self.assertEqual(query.get_state_info("missing", "default"), "default")
        
        # Should return None for missing keys with no default
        self.assertIsNone(query.get_parameter("missing"))
        self.assertIsNone(query.get_state_info("missing"))


if __name__ == "__main__":
    unittest.main() 