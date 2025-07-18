{-# LANGUAGE OverloadedStrings #-}

module Main where

import PathPlanner
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Aeson (encode, decode, Value(..), object, (.=))
import qualified Data.ByteString.Lazy as BS
import System.IO (putStrLn)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import System.Environment (getArgs)
import qualified Data.HashMap.Strict as HM

-- Test data creation helpers
createAgentData :: Text -> Map Text Text -> Text -> AgentData
createAgentData name params desc = AgentData name params desc Nothing

createNode :: Text -> Text -> [AgentData] -> [AgentData] -> Node
createNode name desc inputs outputs = Node name desc inputs outputs

-- Create a simple test case
createSimpleTestCase :: (AgentGraphNodeSet, [AgentData])
createSimpleTestCase =
    let -- Create agent data
        inputA = createAgentData "input_a" Map.empty "Input A"
        outputB = createAgentData "output_b" Map.empty "Output B"
        outputC = createAgentData "output_c" Map.empty "Output C"
        finalOutput = createAgentData "final_output" Map.empty "Final Output"
        
        -- Create nodes
        nodeA = createNode "node_a" "Process A" [inputA] [outputB]
        nodeB = createNode "node_b" "Process B" [outputB] [outputC]
        nodeC = createNode "node_c" "Process C" [outputC] [finalOutput]
        
        -- Create node set
        nodeSet = AgentGraphNodeSet [nodeA, nodeB, nodeC]
        
        -- Desired outputs
        desiredOutputs = [finalOutput]
    in (nodeSet, desiredOutputs)

-- Create a more complex test case
createComplexTestCase :: (AgentGraphNodeSet, [AgentData])
createComplexTestCase =
    let -- Create agent data with parameters
        inputData = createAgentData "input_data" Map.empty "Input Data"
        processedData = createAgentData "processed_data" (Map.fromList [("format", "json")]) "Processed Data"
        analyzedData = createAgentData "analyzed_data" (Map.fromList [("format", "json"), ("analysis", "basic")]) "Analyzed Data"
        report = createAgentData "report" (Map.fromList [("format", "pdf")]) "Report"
        summary = createAgentData "summary" (Map.fromList [("format", "text")]) "Summary"
        
        -- Create nodes
        processor = createNode "processor" "Data Processor" [inputData] [processedData]
        analyzer = createNode "analyzer" "Data Analyzer" [processedData] [analyzedData]
        reporter = createNode "reporter" "Report Generator" [analyzedData] [report]
        summarizer = createNode "summarizer" "Summary Generator" [analyzedData] [summary]
        
        -- Create node set
        nodeSet = AgentGraphNodeSet [processor, analyzer, reporter, summarizer]
        
        -- Desired outputs
        desiredOutputs = [report, summary]
    in (nodeSet, desiredOutputs)

-- Performance test function
runPerformanceTest :: (AgentGraphNodeSet, [AgentData]) -> Text -> IO ()
runPerformanceTest (nodeSet, desiredOutputs) testName = do
    putStrLn $ "Running " ++ T.unpack testName ++ "..."
    
    start <- getCurrentTime
    let result = planExecutionPath nodeSet desiredOutputs
    end <- getCurrentTime
    
    let duration = diffUTCTime end start
    putStrLn $ "Result: " ++ show result
    putStrLn $ "Duration: " ++ show duration ++ " seconds"
    putStrLn ""

-- Main function
main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> runPerformanceTests
        [inputFile] -> handleJsonInput inputFile
        _ -> putStrLn "Usage: taiat-path-planner [input-file.json]"

-- Handle JSON input from Python interface
handleJsonInput :: String -> IO ()
handleJsonInput inputFile = do
    input <- BS.readFile inputFile
    case decode input of
        Just request -> processRequest request
        Nothing -> putStrLn "Error: Invalid JSON input"

-- Process JSON request
processRequest :: Value -> IO ()
processRequest (Object obj) =
    case (HM.lookup "function" obj, HM.lookup "input" obj) of
        (Just (String funcName), Just input) -> do
            result <- case funcName of
                "planExecutionPath" -> handlePlanExecutionPath input
                "validateOutputs" -> handleValidateOutputs input
                "availableOutputs" -> handleAvailableOutputs input
                "hasCircularDependencies" -> handleHasCircularDependencies input
                _ -> return $ object ["error" .= ("Unknown function: " ++ T.unpack funcName)]
            BS.putStrLn $ encode result
        _ -> BS.putStrLn $ encode $ object ["error" .= ("Invalid request format" :: String)]
processRequest _ = BS.putStrLn $ encode $ object ["error" .= ("Invalid request format" :: String)]

-- Handle planExecutionPath function
handlePlanExecutionPath :: Value -> IO Value
handlePlanExecutionPath (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "desiredOutputs" inputObj) of
        (Just nodeSetJson, Just desiredOutputsJson) ->
            case (decode $ encode nodeSetJson, decode $ encode desiredOutputsJson) of
                (Just nodeSet, Just desiredOutputs) ->
                    let externalInputs = case HM.lookup "externalInputs" inputObj of
                            Just externalInputsJson -> 
                                case decode $ encode externalInputsJson of
                                    Just inputs -> inputs
                                    Nothing -> []
                            Nothing -> []
                        -- Add external inputs as virtual nodes that produce them
                        externalNodes = map (\input -> Node (agentDataName' input <> "_external") 
                                                           ("External input: " <> agentDataName' input) 
                                                           [] 
                                                           [input]) externalInputs
                        extendedNodeSet = AgentGraphNodeSet (externalNodes ++ agentGraphNodeSetNodes nodeSet)
                    in do
                        let result = planExecutionPath extendedNodeSet desiredOutputs
                        return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handlePlanExecutionPath _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle validateOutputs function
handleValidateOutputs :: Value -> IO Value
handleValidateOutputs (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "desiredOutputs" inputObj) of
        (Just nodeSetJson, Just desiredOutputsJson) ->
            case (decode $ encode nodeSetJson, decode $ encode desiredOutputsJson) of
                (Just nodeSet, Just desiredOutputs) ->
                    let result = validateOutputs nodeSet desiredOutputs
                    in return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleValidateOutputs _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle availableOutputs function
handleAvailableOutputs :: Value -> IO Value
handleAvailableOutputs (Object inputObj) =
    case HM.lookup "nodeSet" inputObj of
        Just nodeSetJson ->
            case decode $ encode nodeSetJson of
                Just nodeSet ->
                    let result = availableOutputs nodeSet
                    in return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleAvailableOutputs _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle hasCircularDependencies function
handleHasCircularDependencies :: Value -> IO Value
handleHasCircularDependencies (Object inputObj) =
    case HM.lookup "nodeSet" inputObj of
        Just nodeSetJson ->
            case decode $ encode nodeSetJson of
                Just nodeSet ->
                    let result = hasCircularDependencies nodeSet
                    in return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleHasCircularDependencies _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Run performance tests (original functionality)
runPerformanceTests :: IO ()
runPerformanceTests = do
    putStrLn "Taiat Haskell Path Planner Performance Test"
    putStrLn "==========================================="
    putStrLn ""
    
    -- Test simple case
    let simpleCase = createSimpleTestCase
    runPerformanceTest simpleCase "Simple Linear Pipeline"
    
    -- Test complex case
    let complexCase = createComplexTestCase
    runPerformanceTest complexCase "Complex Multi-Path Dependencies"
    
    -- Test validation
    let (nodeSet, desiredOutputs) = complexCase
    putStrLn "Testing validation functions..."
    putStrLn $ "Outputs can be produced: " ++ show (validateOutputs nodeSet desiredOutputs)
    putStrLn $ "Has circular dependencies: " ++ show (hasCircularDependencies nodeSet)
    
    -- Test available outputs
    let available = availableOutputs nodeSet
    putStrLn $ "Available outputs: " ++ show (length available)
    
    putStrLn ""
    putStrLn "Performance test completed!" 