{-# LANGUAGE OverloadedStrings #-}

module PathPlannerTest where

import PathPlanner
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Test.HUnit
import Data.Maybe (fromJust)

-- Simple test data
testNodeSet :: AgentGraphNodeSet
testNodeSet = AgentGraphNodeSet [
    Node "data_loader" "Load data" [] [AgentData "raw_data" Map.empty "Raw data" Nothing],
    Node "preprocessor" "Preprocess data" [AgentData "raw_data" Map.empty "Raw data" Nothing] [AgentData "processed_data" Map.empty "Processed data" Nothing],
    Node "analyzer" "Analyze data" [AgentData "processed_data" Map.empty "Processed data" Nothing] [AgentData "analysis_results" Map.empty "Results" Nothing]
    ]

testDesiredOutputs :: [AgentData]
testDesiredOutputs = [AgentData "analysis_results" Map.empty "Results" Nothing]

-- Basic test cases
testAgentDataName :: Test
testAgentDataName = TestCase $ do
    let agentData = AgentData "test" Map.empty "test" Nothing
    assertEqual "agent_data_name should return correct name" "test" (agentDataName' agentData)

testAgentDataMatch :: Test
testAgentDataMatch = TestCase $ do
    let data1 = AgentData "test" Map.empty "test" Nothing
        data2 = AgentData "test" Map.empty "test" Nothing
    assertBool "agent_data_match should match identical data" (agentDataMatch data1 data2)

testAgentDataMatchParamsSubset :: Test
testAgentDataMatchParamsSubset = TestCase $ do
    let inputData = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "logistic_regression"), ("version", "v1")]) "output" Nothing
    assertBool "agent_data_match should match parameter subset" (agentDataMatch inputData outputData)

testAgentDataMatchParamsConflict :: Test
testAgentDataMatchParamsConflict = TestCase $ do
    let inputData = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "neural_network"), ("version", "v1")]) "output" Nothing
    assertBool "agent_data_match should not match conflicting parameters" (not (agentDataMatch inputData outputData))

testPlanExecutionPath :: Test
testPlanExecutionPath = TestCase $ do
    let executionPath = planExecutionPath testNodeSet testDesiredOutputs
    assertEqual "execution path should match expected order" ["data_loader", "preprocessor", "analyzer"] executionPath

testValidateOutputs :: Test
testValidateOutputs = TestCase $ do
    let valid = validateOutputs testNodeSet testDesiredOutputs
    assertBool "should validate existing outputs" valid

testInvalidOutput :: Test
testInvalidOutput = TestCase $ do
    let invalidOutputs = [AgentData "nonexistent" Map.empty "Nonexistent output" Nothing]
        valid = validateOutputs testNodeSet invalidOutputs
    assertBool "should not validate nonexistent outputs" (not valid)

testRemoveDuplicates :: Test
testRemoveDuplicates = TestCase $ do
    let input = [1, 2, 2, 3, 1, 4]
        expected = [1, 2, 3, 4]
        result = removeDuplicates input
    assertEqual "should remove duplicates while preserving order" expected result

-- Test suite
tests :: Test
tests = TestList [
    TestLabel "test_agent_data_name" testAgentDataName,
    TestLabel "test_agent_data_match" testAgentDataMatch,
    TestLabel "test_agent_data_match_params_subset" testAgentDataMatchParamsSubset,
    TestLabel "test_agent_data_match_params_conflict" testAgentDataMatchParamsConflict,
    TestLabel "test_plan_execution_path" testPlanExecutionPath,
    TestLabel "test_validate_outputs" testValidateOutputs,
    TestLabel "test_invalid_output" testInvalidOutput,
    TestLabel "test_remove_duplicates" testRemoveDuplicates
    ]

-- Main function to run tests
main :: IO Counts
main = runTestTT tests 