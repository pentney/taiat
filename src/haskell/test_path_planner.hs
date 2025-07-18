{-# LANGUAGE OverloadedStrings #-}

module TestPathPlanner where

import PathPlanner
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Test.HUnit
import Data.Maybe (fromJust, isJust, isNothing)

-- ============================================================================
-- TEST DATA
-- ============================================================================

-- Simple test data
testNodeSet :: AgentGraphNodeSet
testNodeSet = AgentGraphNodeSet [
    Node "data_loader" "Load data from source" [] 
        [AgentData "raw_data" Map.empty "Raw data from source" Nothing],
    Node "preprocessor" "Preprocess the data" 
        [AgentData "raw_data" Map.empty "Raw data from source" Nothing] 
        [AgentData "processed_data" Map.empty "Preprocessed data" Nothing],
    Node "analyzer" "Analyze the data" 
        [AgentData "processed_data" Map.empty "Preprocessed data" Nothing] 
        [AgentData "analysis_results" Map.empty "Analysis results" Nothing],
    Node "visualizer" "Create visualizations" 
        [AgentData "analysis_results" Map.empty "Analysis results" Nothing] 
        [AgentData "visualizations" Map.empty "Generated visualizations" Nothing]
    ]

testDesiredOutputs :: [AgentData]
testDesiredOutputs = [AgentData "visualizations" Map.empty "Generated visualizations" Nothing]

-- Complex test data with multiple paths
testComplexNodeSet :: AgentGraphNodeSet
testComplexNodeSet = AgentGraphNodeSet [
    Node "csv_loader" "Load CSV data" [] 
        [AgentData "csv_data" Map.empty "CSV data" Nothing],
    Node "json_loader" "Load JSON data" [] 
        [AgentData "json_data" Map.empty "JSON data" Nothing],
    Node "csv_processor" "Process CSV data" 
        [AgentData "csv_data" Map.empty "CSV data" Nothing] 
        [AgentData "processed_csv" Map.empty "Processed CSV data" Nothing],
    Node "json_processor" "Process JSON data" 
        [AgentData "json_data" Map.empty "JSON data" Nothing] 
        [AgentData "processed_json" Map.empty "Processed JSON data" Nothing],
    Node "data_merger" "Merge processed data" 
        [AgentData "processed_csv" Map.empty "Processed CSV data" Nothing,
         AgentData "processed_json" Map.empty "Processed JSON data" Nothing] 
        [AgentData "merged_data" Map.empty "Merged data" Nothing],
    Node "analyzer" "Analyze merged data" 
        [AgentData "merged_data" Map.empty "Merged data" Nothing] 
        [AgentData "analysis_results" Map.empty "Analysis results" Nothing]
    ]

testComplexDesiredOutputs :: [AgentData]
testComplexDesiredOutputs = [AgentData "analysis_results" Map.empty "Analysis results" Nothing]

-- ============================================================================
-- SIMPLE TESTS (Start with these)
-- ============================================================================

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

testAgentDataMatchParamsEmpty :: Test
testAgentDataMatchParamsEmpty = TestCase $ do
    let inputData = AgentData "model" Map.empty "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "neural_network")]) "output" Nothing
    assertBool "agent_data_match should match when input params are empty" (agentDataMatch inputData outputData)

testAgentDataMatchNameMismatch :: Test
testAgentDataMatchNameMismatch = TestCase $ do
    let inputData = AgentData "modelA" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "modelB" (Map.fromList [("type", "logistic_regression")]) "output" Nothing
    assertBool "agent_data_match should not match different names" (not (agentDataMatch inputData outputData))

testRemoveDuplicates :: Test
testRemoveDuplicates = TestCase $ do
    let input = [1, 2, 2, 3, 1, 4]
        expected = [1, 2, 3, 4]
        result = removeDuplicates input
    assertEqual "should remove duplicates while preserving order" expected result

testRemoveDuplicatesEmpty :: Test
testRemoveDuplicatesEmpty = TestCase $ do
    let result = removeDuplicates ([] :: [Int])
    assertEqual "should handle empty list" [] result

testRemoveDuplicatesSingle :: Test
testRemoveDuplicatesSingle = TestCase $ do
    let result = removeDuplicates [1]
    assertEqual "should handle single element" [1] result

-- ============================================================================
-- NODE OPERATION TESTS
-- ============================================================================

testNodeProducesOutput :: Test
testNodeProducesOutput = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        dataLoader = head $ filter (\n -> nodeName' n == "data_loader") nodes
        rawData = AgentData "raw_data" Map.empty "Raw data from source" Nothing
        producers = nodesProducingOutput nodes rawData
    assertEqual "should find data_loader as producer" 1 (length producers)
    assertEqual "should be data_loader" "data_loader" (nodeName' (head producers))

testNodesProducingOutputName :: Test
testNodesProducingOutputName = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        producers = nodesProducingOutputName nodes "raw_data"
    assertEqual "should find one producer for raw_data" 1 (length producers)
    assertEqual "should be data_loader" "data_loader" (nodeName' (head producers))

testNodeDependencies :: Test
testNodeDependencies = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        dependencies = nodeDependencies nodes preprocessor
    assertEqual "should have one dependency" 1 (length dependencies)
    assertEqual "dependency should be data_loader" "data_loader" (nodeName' (head dependencies))

testNodeReady :: Test
testNodeReady = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        dataLoader = head $ filter (\n -> nodeName' n == "data_loader") nodes
        ready = nodeReady nodes preprocessor [dataLoader]
    assertBool "preprocessor should be ready when data_loader is executed" ready

testNodeNotReady :: Test
testNodeNotReady = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        ready = nodeReady nodes preprocessor []
    assertBool "preprocessor should not be ready when no dependencies executed" (not ready)

-- ============================================================================
-- PATH PLANNING TESTS
-- ============================================================================

testRequiredNodes :: Test
testRequiredNodes = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        required = requiredNodes nodes testDesiredOutputs
    assertBool "should find required nodes" (isJust required)
    let requiredNodes' = fromJust required
    assertEqual "should find 4 required nodes" 4 (length requiredNodes')
    let nodeNames = map nodeName' requiredNodes'
    assertBool "should include data_loader" ("data_loader" `elem` nodeNames)
    assertBool "should include preprocessor" ("preprocessor" `elem` nodeNames)
    assertBool "should include analyzer" ("analyzer" `elem` nodeNames)
    assertBool "should include visualizer" ("visualizer" `elem` nodeNames)

testTopologicalSort :: Test
testTopologicalSort = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        sorted = topologicalSort nodes
    assertEqual "should sort all 4 nodes" 4 (length sorted)
    let nodeNames = map nodeName' sorted
    assertEqual "first should be data_loader" "data_loader" (head nodeNames)
    assertEqual "second should be preprocessor" "preprocessor" (nodeNames !! 1)
    assertEqual "third should be analyzer" "analyzer" (nodeNames !! 2)
    assertEqual "fourth should be visualizer" "visualizer" (nodeNames !! 3)

testPlanExecutionPath :: Test
testPlanExecutionPath = TestCase $ do
    let executionPath = planExecutionPath testNodeSet testDesiredOutputs
    assertEqual "should have 4 steps" 4 (length executionPath)
    assertEqual "first step should be data_loader" "data_loader" (head executionPath)
    assertEqual "second step should be preprocessor" "preprocessor" (executionPath !! 1)
    assertEqual "third step should be analyzer" "analyzer" (executionPath !! 2)
    assertEqual "fourth step should be visualizer" "visualizer" (executionPath !! 3)

testComplexPathPlanning :: Test
testComplexPathPlanning = TestCase $ do
    let executionPath = planExecutionPath testComplexNodeSet testComplexDesiredOutputs
    assertEqual "should have 6 steps" 6 (length executionPath)
    assertBool "should include csv_loader" ("csv_loader" `elem` executionPath)
    assertBool "should include json_loader" ("json_loader" `elem` executionPath)
    assertBool "should include csv_processor" ("csv_processor" `elem` executionPath)
    assertBool "should include json_processor" ("json_processor" `elem` executionPath)
    assertBool "should include data_merger" ("data_merger" `elem` executionPath)
    assertBool "should include analyzer" ("analyzer" `elem` executionPath)

-- ============================================================================
-- VALIDATION TESTS
-- ============================================================================

testValidateOutputs :: Test
testValidateOutputs = TestCase $ do
    let valid = validateOutputs testNodeSet testDesiredOutputs
    assertBool "should validate existing outputs" valid

testInvalidOutput :: Test
testInvalidOutput = TestCase $ do
    let invalidOutputs = [AgentData "nonexistent" Map.empty "Nonexistent output" Nothing]
        valid = validateOutputs testNodeSet invalidOutputs
    assertBool "should not validate nonexistent outputs" (not valid)

testAvailableOutputs :: Test
testAvailableOutputs = TestCase $ do
    let available = availableOutputs testNodeSet
        outputNames = map agentDataName' available
    assertBool "should include raw_data" ("raw_data" `elem` outputNames)
    assertBool "should include processed_data" ("processed_data" `elem` outputNames)
    assertBool "should include analysis_results" ("analysis_results" `elem` outputNames)
    assertBool "should include visualizations" ("visualizations" `elem` outputNames)

-- ============================================================================
-- EDGE CASE TESTS
-- ============================================================================

testEmptyNodeSet :: Test
testEmptyNodeSet = TestCase $ do
    let emptySet = AgentGraphNodeSet []
        executionPath = planExecutionPath emptySet testDesiredOutputs
    assertEqual "should return empty path for empty node set" [] executionPath

testNoRequiredNodes :: Test
testNoRequiredNodes = TestCase $ do
    let nodes = agentGraphNodeSetNodes testNodeSet
        impossibleOutputs = [AgentData "impossible" Map.empty "Impossible output" Nothing]
        required = requiredNodes nodes impossibleOutputs
    assertBool "should return Nothing for impossible outputs" (isNothing required)

testCircularDependencies :: Test
testCircularDependencies = TestCase $ do
    let circularSet = AgentGraphNodeSet [
            Node "A" "Node A" [AgentData "output_B" Map.empty "Output B" Nothing] 
                [AgentData "output_A" Map.empty "Output A" Nothing],
            Node "B" "Node B" [AgentData "output_A" Map.empty "Output A" Nothing] 
                [AgentData "output_B" Map.empty "Output B" Nothing]
            ]
        hasCircular = hasCircularDependencies circularSet
    assertBool "should detect circular dependencies" hasCircular

testNoCircularDependencies :: Test
testNoCircularDependencies = TestCase $ do
    let hasCircular = hasCircularDependencies testNodeSet
    assertBool "should not detect circular dependencies in valid graph" (not hasCircular)

-- ============================================================================
-- PARAMETER MATCHING TESTS
-- ============================================================================

testFlexibleParameterMatching :: Test
testFlexibleParameterMatching = TestCase $ do
    let data1 = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        data2 = AgentData "model" (Map.fromList [("type", "logistic_regression"), ("version", "v1")]) "output" Nothing
        data3 = AgentData "model" (Map.fromList [("version", "v1")]) "input2" Nothing
    assertBool "should match subset in forward direction" (agentDataMatchFlexible data1 data2)
    assertBool "should match subset in reverse direction" (agentDataMatchFlexible data2 data3)
    assertBool "should match identical parameters" (agentDataMatchFlexible data1 data1)

testSpecificityScoring :: Test
testSpecificityScoring = TestCase $ do
    let desiredParams = Map.fromList [("type", "logistic_regression"), ("version", "v1")]
        producedParams1 = Map.fromList [("type", "logistic_regression"), ("version", "v1"), ("optimizer", "adam")]
        producedParams2 = Map.fromList [("type", "logistic_regression")]
        score1 = calculateSpecificityScore desiredParams producedParams1
        score2 = calculateSpecificityScore desiredParams producedParams2
    assertEqual "should score exact match higher" 2 score1
    assertEqual "should score partial match lower" 1 score2

-- ============================================================================
-- TEST SUITES
-- ============================================================================

-- Simple tests that should pass first
simpleTests :: Test
simpleTests = TestList [
    TestLabel "test_agent_data_name" testAgentDataName,
    TestLabel "test_agent_data_match" testAgentDataMatch,
    TestLabel "test_agent_data_match_params_subset" testAgentDataMatchParamsSubset,
    TestLabel "test_agent_data_match_params_conflict" testAgentDataMatchParamsConflict,
    TestLabel "test_agent_data_match_params_empty" testAgentDataMatchParamsEmpty,
    TestLabel "test_agent_data_match_name_mismatch" testAgentDataMatchNameMismatch,
    TestLabel "test_remove_duplicates" testRemoveDuplicates,
    TestLabel "test_remove_duplicates_empty" testRemoveDuplicatesEmpty,
    TestLabel "test_remove_duplicates_single" testRemoveDuplicatesSingle
    ]

-- Node operation tests
nodeTests :: Test
nodeTests = TestList [
    TestLabel "test_node_produces_output" testNodeProducesOutput,
    TestLabel "test_nodes_producing_output_name" testNodesProducingOutputName,
    TestLabel "test_node_dependencies" testNodeDependencies,
    TestLabel "test_node_ready" testNodeReady,
    TestLabel "test_node_not_ready" testNodeNotReady
    ]

-- Path planning tests
pathPlanningTests :: Test
pathPlanningTests = TestList [
    TestLabel "test_required_nodes" testRequiredNodes,
    TestLabel "test_topological_sort" testTopologicalSort,
    TestLabel "test_plan_execution_path" testPlanExecutionPath,
    TestLabel "test_complex_path_planning" testComplexPathPlanning
    ]

-- Validation tests
validationTests :: Test
validationTests = TestList [
    TestLabel "test_validate_outputs" testValidateOutputs,
    TestLabel "test_invalid_output" testInvalidOutput,
    TestLabel "test_available_outputs" testAvailableOutputs
    ]

-- Edge case tests
edgeCaseTests :: Test
edgeCaseTests = TestList [
    TestLabel "test_empty_node_set" testEmptyNodeSet,
    TestLabel "test_no_required_nodes" testNoRequiredNodes,
    TestLabel "test_circular_dependencies" testCircularDependencies,
    TestLabel "test_no_circular_dependencies" testNoCircularDependencies
    ]

-- Parameter matching tests
parameterTests :: Test
parameterTests = TestList [
    TestLabel "test_flexible_parameter_matching" testFlexibleParameterMatching,
    TestLabel "test_specificity_scoring" testSpecificityScoring
    ]

-- All tests
allTests :: Test
allTests = TestList [
    TestLabel "Simple Tests" simpleTests,
    TestLabel "Node Tests" nodeTests,
    TestLabel "Path Planning Tests" pathPlanningTests,
    TestLabel "Validation Tests" validationTests,
    TestLabel "Edge Case Tests" edgeCaseTests,
    TestLabel "Parameter Tests" parameterTests
    ]

-- Main function to run tests
main :: IO Counts
main = runTestTT allTests 