{-# LANGUAGE OverloadedStrings #-}

module TestSimpleRunner where

import PathPlanner
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
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

-- Test data with parameter matching
testParameterNodeSet :: AgentGraphNodeSet
testParameterNodeSet = AgentGraphNodeSet [
    Node "model_trainer_lr" "Train logistic regression model" 
        [AgentData "training_data" Map.empty "Training data" Nothing] 
        [AgentData "model" (Map.fromList [("type", "logistic_regression"), ("version", "v1")]) "Trained model" Nothing],
    Node "model_trainer_nn" "Train neural network model" 
        [AgentData "training_data" Map.empty "Training data" Nothing] 
        [AgentData "model" (Map.fromList [("type", "neural_network"), ("version", "v2")]) "Trained model" Nothing],
    Node "model_evaluator" "Evaluate model" 
        [AgentData "model" (Map.fromList [("type", "logistic_regression")]) "Model to evaluate" Nothing] 
        [AgentData "evaluation_results" Map.empty "Evaluation results" Nothing]
    ]

-- ============================================================================
-- SIMPLE TESTS (Start with these)
-- ============================================================================

testAgentDataName :: IO Bool
testAgentDataName = do
    let agentData = AgentData "test" Map.empty "test" Nothing
        result = agentDataName' agentData == "test"
    putStrLn $ "test_agent_data_name: " ++ if result then "PASS" else "FAIL"
    return result

testAgentDataMatch :: IO Bool
testAgentDataMatch = do
    let data1 = AgentData "test" Map.empty "test" Nothing
        data2 = AgentData "test" Map.empty "test" Nothing
        result = agentDataMatch data1 data2
    putStrLn $ "test_agent_data_match: " ++ if result then "PASS" else "FAIL"
    return result

testAgentDataMatchParamsSubset :: IO Bool
testAgentDataMatchParamsSubset = do
    let inputData = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "logistic_regression"), ("version", "v1")]) "output" Nothing
        result = agentDataMatch inputData outputData
    putStrLn $ "test_agent_data_match_params_subset: " ++ if result then "PASS" else "FAIL"
    return result

testAgentDataMatchParamsConflict :: IO Bool
testAgentDataMatchParamsConflict = do
    let inputData = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "neural_network"), ("version", "v1")]) "output" Nothing
        result = not (agentDataMatch inputData outputData)
    putStrLn $ "test_agent_data_match_params_conflict: " ++ if result then "PASS" else "FAIL"
    return result

testAgentDataMatchParamsEmpty :: IO Bool
testAgentDataMatchParamsEmpty = do
    let inputData = AgentData "model" Map.empty "input" Nothing
        outputData = AgentData "model" (Map.fromList [("type", "neural_network")]) "output" Nothing
        result = agentDataMatch inputData outputData
    putStrLn $ "test_agent_data_match_params_empty: " ++ if result then "PASS" else "FAIL"
    return result

testAgentDataMatchNameMismatch :: IO Bool
testAgentDataMatchNameMismatch = do
    let inputData = AgentData "modelA" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        outputData = AgentData "modelB" (Map.fromList [("type", "logistic_regression")]) "output" Nothing
        result = not (agentDataMatch inputData outputData)
    putStrLn $ "test_agent_data_match_name_mismatch: " ++ if result then "PASS" else "FAIL"
    return result

testRemoveDuplicates :: IO Bool
testRemoveDuplicates = do
    let input = [1, 2, 2, 3, 1, 4]
        expected = [1, 2, 3, 4]
        result = removeDuplicates input
        passed = result == expected
    putStrLn $ "test_remove_duplicates: " ++ if passed then "PASS" else "FAIL"
    return passed

testRemoveDuplicatesEmpty :: IO Bool
testRemoveDuplicatesEmpty = do
    let result = removeDuplicates ([] :: [Int])
        passed = result == []
    putStrLn $ "test_remove_duplicates_empty: " ++ if passed then "PASS" else "FAIL"
    return passed

testRemoveDuplicatesSingle :: IO Bool
testRemoveDuplicatesSingle = do
    let result = removeDuplicates [1]
        passed = result == [1]
    putStrLn $ "test_remove_duplicates_single: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- NODE OPERATION TESTS
-- ============================================================================

testNodeProducesOutput :: IO Bool
testNodeProducesOutput = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        rawData = AgentData "raw_data" Map.empty "Raw data from source" Nothing
        producers = nodesProducingOutput nodes rawData
        passed = length producers == 1 && nodeName' (head producers) == "data_loader"
    putStrLn $ "test_node_produces_output: " ++ if passed then "PASS" else "FAIL"
    return passed

testNodesProducingOutputName :: IO Bool
testNodesProducingOutputName = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        producers = nodesProducingOutputName nodes "raw_data"
        passed = length producers == 1 && nodeName' (head producers) == "data_loader"
    putStrLn $ "test_nodes_producing_output_name: " ++ if passed then "PASS" else "FAIL"
    return passed

testNodeDependencies :: IO Bool
testNodeDependencies = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        dependencies = nodeDependencies nodes preprocessor
        passed = length dependencies == 1 && nodeName' (head dependencies) == "data_loader"
    putStrLn $ "test_node_dependencies: " ++ if passed then "PASS" else "FAIL"
    return passed

testNodeReady :: IO Bool
testNodeReady = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        dataLoader = head $ filter (\n -> nodeName' n == "data_loader") nodes
        ready = nodeReady nodes preprocessor [dataLoader]
    putStrLn $ "test_node_ready: " ++ if ready then "PASS" else "FAIL"
    return ready

testNodeNotReady :: IO Bool
testNodeNotReady = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        preprocessor = head $ filter (\n -> nodeName' n == "preprocessor") nodes
        ready = nodeReady nodes preprocessor []
        passed = not ready
    putStrLn $ "test_node_not_ready: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- PATH PLANNING TESTS
-- ============================================================================

testRequiredNodes :: IO Bool
testRequiredNodes = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        required = requiredNodes nodes testDesiredOutputs
        passed = case required of
            Just requiredNodes' -> 
                let nodeNames = map nodeName' requiredNodes'
                in length requiredNodes' == 4 &&
                   "data_loader" `elem` nodeNames &&
                   "preprocessor" `elem` nodeNames &&
                   "analyzer" `elem` nodeNames &&
                   "visualizer" `elem` nodeNames
            Nothing -> False
    putStrLn $ "test_required_nodes: " ++ if passed then "PASS" else "FAIL"
    return passed

testTopologicalSort :: IO Bool
testTopologicalSort = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        sorted = topologicalSort nodes
        nodeNames = map nodeName' sorted
        passed = length sorted == 4 &&
                head nodeNames == "data_loader" &&
                nodeNames !! 1 == "preprocessor" &&
                nodeNames !! 2 == "analyzer" &&
                nodeNames !! 3 == "visualizer"
    putStrLn $ "test_topological_sort: " ++ if passed then "PASS" else "FAIL"
    return passed

testPlanExecutionPath :: IO Bool
testPlanExecutionPath = do
    let executionPath = planExecutionPath testNodeSet testDesiredOutputs
        passed = length executionPath == 4 &&
                head executionPath == "data_loader" &&
                executionPath !! 1 == "preprocessor" &&
                executionPath !! 2 == "analyzer" &&
                executionPath !! 3 == "visualizer"
    putStrLn $ "test_plan_execution_path: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- VALIDATION TESTS
-- ============================================================================

testValidateOutputs :: IO Bool
testValidateOutputs = do
    let valid = validateOutputs testNodeSet testDesiredOutputs
    putStrLn $ "test_validate_outputs: " ++ if valid then "PASS" else "FAIL"
    return valid

testInvalidOutput :: IO Bool
testInvalidOutput = do
    let invalidOutputs = [AgentData "nonexistent" Map.empty "Nonexistent output" Nothing]
        valid = validateOutputs testNodeSet invalidOutputs
        passed = not valid
    putStrLn $ "test_invalid_output: " ++ if passed then "PASS" else "FAIL"
    return passed

testAvailableOutputs :: IO Bool
testAvailableOutputs = do
    let available = availableOutputs testNodeSet
        outputNames = map agentDataName' available
        passed = "raw_data" `elem` outputNames &&
                "processed_data" `elem` outputNames &&
                "analysis_results" `elem` outputNames &&
                "visualizations" `elem` outputNames
    putStrLn $ "test_available_outputs: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- EDGE CASE TESTS
-- ============================================================================

testEmptyNodeSet :: IO Bool
testEmptyNodeSet = do
    let emptySet = AgentGraphNodeSet []
        executionPath = planExecutionPath emptySet testDesiredOutputs
        passed = executionPath == []
    putStrLn $ "test_empty_node_set: " ++ if passed then "PASS" else "FAIL"
    return passed

testNoRequiredNodes :: IO Bool
testNoRequiredNodes = do
    let nodes = agentGraphNodeSetNodes testNodeSet
        impossibleOutputs = [AgentData "impossible" Map.empty "Impossible output" Nothing]
        required = requiredNodes nodes impossibleOutputs
        passed = isNothing required
    putStrLn $ "test_no_required_nodes: " ++ if passed then "PASS" else "FAIL"
    return passed

testCircularDependencies :: IO Bool
testCircularDependencies = do
    let circularSet = AgentGraphNodeSet [
            Node "A" "Node A" [AgentData "output_B" Map.empty "Output B" Nothing] 
                [AgentData "output_A" Map.empty "Output A" Nothing],
            Node "B" "Node B" [AgentData "output_A" Map.empty "Output A" Nothing] 
                [AgentData "output_B" Map.empty "Output B" Nothing]
            ]
        hasCircular = hasCircularDependencies circularSet
    putStrLn $ "test_circular_dependencies: " ++ if hasCircular then "PASS" else "FAIL"
    return hasCircular

testNoCircularDependencies :: IO Bool
testNoCircularDependencies = do
    let hasCircular = hasCircularDependencies testNodeSet
        passed = not hasCircular
    putStrLn $ "test_no_circular_dependencies: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- PARAMETER MATCHING TESTS
-- ============================================================================

testFlexibleParameterMatching :: IO Bool
testFlexibleParameterMatching = do
    let data1 = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "input" Nothing
        data2 = AgentData "model" (Map.fromList [("type", "logistic_regression"), ("version", "v1")]) "output" Nothing
        data3 = AgentData "model" (Map.fromList [("version", "v1")]) "input2" Nothing
        passed = agentDataMatchFlexible data1 data2 &&
                agentDataMatchFlexible data2 data3 &&
                agentDataMatchFlexible data1 data1
    putStrLn $ "test_flexible_parameter_matching: " ++ if passed then "PASS" else "FAIL"
    return passed

testSpecificityScoring :: IO Bool
testSpecificityScoring = do
    let desiredParams = Map.fromList [("type", "logistic_regression"), ("version", "v1")]
        producedParams1 = Map.fromList [("type", "logistic_regression"), ("version", "v1"), ("optimizer", "adam")]
        producedParams2 = Map.fromList [("type", "logistic_regression")]
        score1 = calculateSpecificityScore desiredParams producedParams1
        score2 = calculateSpecificityScore desiredParams producedParams2
        passed = score1 == 2 && score2 == 1
    putStrLn $ "test_specificity_scoring: " ++ if passed then "PASS" else "FAIL"
    return passed

testParameterMatching :: IO Bool
testParameterMatching = do
    let nodes = agentGraphNodeSetNodes testParameterNodeSet
        desiredModel = AgentData "model" (Map.fromList [("type", "logistic_regression")]) "Desired model" Nothing
        producers = nodesProducingOutput nodes desiredModel
        passed = length producers == 1 && nodeName' (head producers) == "model_trainer_lr"
    putStrLn $ "test_parameter_matching: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- COMPLEX PATH PLANNING TESTS
-- ============================================================================

testComplexPathPlanning :: IO Bool
testComplexPathPlanning = do
    let executionPath = planExecutionPath testComplexNodeSet testComplexDesiredOutputs
        passed = length executionPath == 6 &&
                "csv_loader" `elem` executionPath &&
                "json_loader" `elem` executionPath &&
                "csv_processor" `elem` executionPath &&
                "json_processor" `elem` executionPath &&
                "data_merger" `elem` executionPath &&
                "analyzer" `elem` executionPath
    putStrLn $ "test_complex_path_planning: " ++ if passed then "PASS" else "FAIL"
    return passed

testMultipleOutputs :: IO Bool
testMultipleOutputs = do
    let multipleOutputs = [
            AgentData "analysis_results" Map.empty "Analysis results" Nothing,
            AgentData "visualizations" Map.empty "Generated visualizations" Nothing
            ]
        executionPath = planExecutionPath testNodeSet multipleOutputs
        passed = length executionPath == 4 &&
                "data_loader" `elem` executionPath &&
                "preprocessor" `elem` executionPath &&
                "analyzer" `elem` executionPath &&
                "visualizer" `elem` executionPath
    putStrLn $ "test_multiple_outputs: " ++ if passed then "PASS" else "FAIL"
    return passed

-- ============================================================================
-- RUN ALL TESTS
-- ============================================================================

runSimpleTests :: IO ()
runSimpleTests = do
    putStrLn "Running Simple Tests..."
    putStrLn "========================"
    
    results <- sequence [
        testAgentDataName,
        testAgentDataMatch,
        testAgentDataMatchParamsSubset,
        testAgentDataMatchParamsConflict,
        testAgentDataMatchParamsEmpty,
        testAgentDataMatchNameMismatch,
        testRemoveDuplicates,
        testRemoveDuplicatesEmpty,
        testRemoveDuplicatesSingle
        ]
    
    let passed = length $ filter id results
        total = length results
    putStrLn $ "\nSimple Tests: " ++ show passed ++ "/" ++ show total ++ " passed"
    
    if passed == total
        then do
            putStrLn "\nAll simple tests passed! Running node operation tests..."
            putStrLn "=================================================="
            
            nodeResults <- sequence [
                testNodeProducesOutput,
                testNodesProducingOutputName,
                testNodeDependencies,
                testNodeReady,
                testNodeNotReady
                ]
            
            let nodePassed = length $ filter id nodeResults
                nodeTotal = length nodeResults
            putStrLn $ "\nNode Tests: " ++ show nodePassed ++ "/" ++ show nodeTotal ++ " passed"
            
            if nodePassed == nodeTotal
                then do
                    putStrLn "\nAll node tests passed! Running path planning tests..."
                    putStrLn "=================================================="
                    
                    pathResults <- sequence [
                        testRequiredNodes,
                        testTopologicalSort,
                        testPlanExecutionPath
                        ]
                    
                    let pathPassed = length $ filter id pathResults
                        pathTotal = length pathResults
                    putStrLn $ "\nPath Planning Tests: " ++ show pathPassed ++ "/" ++ show pathTotal ++ " passed"
                    
                    if pathPassed == pathTotal
                        then do
                            putStrLn "\nAll path planning tests passed! Running validation tests..."
                            putStrLn "=================================================="
                            
                            validationResults <- sequence [
                                testValidateOutputs,
                                testInvalidOutput,
                                testAvailableOutputs
                                ]
                            
                            let validationPassed = length $ filter id validationResults
                                validationTotal = length validationResults
                            putStrLn $ "\nValidation Tests: " ++ show validationPassed ++ "/" ++ show validationTotal ++ " passed"
                            
                            if validationPassed == validationTotal
                                then do
                                    putStrLn "\nAll validation tests passed! Running edge case tests..."
                                    putStrLn "=================================================="
                                    
                                    edgeResults <- sequence [
                                        testEmptyNodeSet,
                                        testNoRequiredNodes,
                                        testCircularDependencies,
                                        testNoCircularDependencies
                                        ]
                                    
                                    let edgePassed = length $ filter id edgeResults
                                        edgeTotal = length edgeResults
                                    putStrLn $ "\nEdge Case Tests: " ++ show edgePassed ++ "/" ++ show edgeTotal ++ " passed"
                                    
                                    if edgePassed == edgeTotal
                                        then do
                                            putStrLn "\nAll edge case tests passed! Running parameter matching tests..."
                                            putStrLn "=================================================="
                                            
                                            paramResults <- sequence [
                                                testFlexibleParameterMatching,
                                                testSpecificityScoring,
                                                testParameterMatching
                                                ]
                                            
                                            let paramPassed = length $ filter id paramResults
                                                paramTotal = length paramResults
                                            putStrLn $ "\nParameter Tests: " ++ show paramPassed ++ "/" ++ show paramTotal ++ " passed"
                                            
                                            if paramPassed == paramTotal
                                                then do
                                                    putStrLn "\nAll parameter tests passed! Running complex path planning tests..."
                                                    putStrLn "=================================================="
                                                    
                                                    complexResults <- sequence [
                                                        testComplexPathPlanning,
                                                        testMultipleOutputs
                                                        ]
                                                    
                                                    let complexPassed = length $ filter id complexResults
                                                        complexTotal = length complexResults
                                                    putStrLn $ "\nComplex Path Planning Tests: " ++ show complexPassed ++ "/" ++ show complexTotal ++ " passed"
                                                    
                                                    let allPassed = passed + nodePassed + pathPassed + validationPassed + edgePassed + paramPassed + complexPassed
                                                        allTotal = total + nodeTotal + pathTotal + validationTotal + edgeTotal + paramTotal + complexTotal
                                                    putStrLn $ "\n=== FINAL RESULTS ==="
                                                    putStrLn $ "All Tests: " ++ show allPassed ++ "/" ++ show allTotal ++ " passed"
                                                    
                                                    if allPassed == allTotal
                                                        then putStrLn "🎉 ALL TESTS PASSED! 🎉"
                                                        else putStrLn "❌ Some tests failed"
                                                else putStrLn "❌ Some parameter tests failed"
                                        else putStrLn "❌ Some edge case tests failed"
                                else putStrLn "❌ Some validation tests failed"
                        else putStrLn "❌ Some path planning tests failed"
                else putStrLn "❌ Some node tests failed"
        else putStrLn "❌ Some simple tests failed"

main :: IO ()
main = runSimpleTests 