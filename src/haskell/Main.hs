{-# LANGUAGE OverloadedStrings #-}

module Main where

import PathPlanner
import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Aeson (encode, decode, Value(..), object, (.=))
import qualified Data.ByteString.Lazy as BS
import System.IO (putStrLn, hFlush, stdout, stdin)
import System.Environment (getArgs)
import qualified Data.HashMap.Strict as HM
import Control.Monad (forever)
import Data.Maybe (fromMaybe)

-- Main function
main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> putStrLn "Usage: taiat-path-planner --daemon [input-file.json]"
        ["--daemon"] -> runDaemonMode
        [inputFile] -> handleJsonInput inputFile
        _ -> putStrLn "Usage: taiat-path-planner [--daemon] [input-file.json]"

-- Run in daemon mode - continuously read from stdin and write to stdout
runDaemonMode :: IO ()
runDaemonMode = do
    -- Set line buffering for stdout
    hFlush stdout
    forever $ do
        line <- getLine
        case decode (BS.pack (map (fromIntegral . fromEnum) line)) of
            Just request -> do
                result <- processDaemonRequest request
                BS.putStrLn $ encode result
                hFlush stdout
            Nothing -> do
                BS.putStrLn $ encode $ object ["error" .= ("Invalid JSON input" :: String)]
                hFlush stdout

-- Process request in daemon mode
processDaemonRequest :: Value -> IO Value
processDaemonRequest (Object obj) =
    case (HM.lookup "request_id" obj, HM.lookup "function" obj, HM.lookup "input" obj) of
        (Just requestId, Just (String funcName), Just input) -> do
            result <- case funcName of
                "planExecutionPath" -> handlePlanExecutionPath input
                "planAlternativeExecutionPath" -> handlePlanAlternativeExecutionPath input
                "planMultipleAlternativePaths" -> handlePlanMultipleAlternativePaths input
                "validateOutputs" -> handleValidateOutputs input
                "validateOutputsWithFailedNodes" -> handleValidateOutputsWithFailedNodes input
                "availableOutputs" -> handleAvailableOutputs input
                "availableOutputsWithFailedNodes" -> handleAvailableOutputsWithFailedNodes input
                "hasCircularDependencies" -> handleHasCircularDependencies input
                _ -> return $ object ["error" .= ("Unknown function: " ++ T.unpack funcName)]
            return $ object ["request_id" .= requestId, "result" .= result]
        _ -> return $ object ["error" .= ("Invalid request format" :: String)]
processDaemonRequest _ = return $ object ["error" .= ("Invalid request format" :: String)]

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
                (Just nodeSet, Just desiredOutputs) -> do
                    let result = validateOutputs nodeSet desiredOutputs
                    return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleValidateOutputs _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle planAlternativeExecutionPath function
handlePlanAlternativeExecutionPath :: Value -> IO Value
handlePlanAlternativeExecutionPath (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "desiredOutputs" inputObj, HM.lookup "failedNodeNames" inputObj) of
        (Just nodeSetJson, Just desiredOutputsJson, Just failedNodeNamesJson) ->
            case (decode $ encode nodeSetJson, decode $ encode desiredOutputsJson, decode $ encode failedNodeNamesJson) of
                (Just nodeSet, Just desiredOutputs, Just failedNodeNames) ->
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
                        let result = planAlternativeExecutionPath extendedNodeSet desiredOutputs failedNodeNames
                        return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handlePlanAlternativeExecutionPath _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle planMultipleAlternativePaths function
handlePlanMultipleAlternativePaths :: Value -> IO Value
handlePlanMultipleAlternativePaths (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "desiredOutputs" inputObj, HM.lookup "failedNodeNames" inputObj) of
        (Just nodeSetJson, Just desiredOutputsJson, Just failedNodeNamesJson) ->
            case (decode $ encode nodeSetJson, decode $ encode desiredOutputsJson, decode $ encode failedNodeNamesJson) of
                (Just nodeSet, Just desiredOutputs, Just failedNodeNames) ->
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
                        let result = planMultipleAlternativePaths extendedNodeSet desiredOutputs failedNodeNames
                        return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handlePlanMultipleAlternativePaths _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle validateOutputsWithFailedNodes function
handleValidateOutputsWithFailedNodes :: Value -> IO Value
handleValidateOutputsWithFailedNodes (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "desiredOutputs" inputObj, HM.lookup "failedNodeNames" inputObj) of
        (Just nodeSetJson, Just desiredOutputsJson, Just failedNodeNamesJson) ->
            case (decode $ encode nodeSetJson, decode $ encode desiredOutputsJson, decode $ encode failedNodeNamesJson) of
                (Just nodeSet, Just desiredOutputs, Just failedNodeNames) -> do
                    let result = validateOutputsWithFailedNodes nodeSet desiredOutputs failedNodeNames
                    return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleValidateOutputsWithFailedNodes _ = return $ object ["error" .= ("Invalid input format" :: String)]

-- Handle availableOutputsWithFailedNodes function
handleAvailableOutputsWithFailedNodes :: Value -> IO Value
handleAvailableOutputsWithFailedNodes (Object inputObj) =
    case (HM.lookup "nodeSet" inputObj, HM.lookup "failedNodeNames" inputObj) of
        (Just nodeSetJson, Just failedNodeNamesJson) ->
            case (decode $ encode nodeSetJson, decode $ encode failedNodeNamesJson) of
                (Just nodeSet, Just failedNodeNames) -> do
                    let result = availableOutputsWithFailedNodes nodeSet failedNodeNames
                    return $ object ["result" .= result]
                _ -> return $ object ["error" .= ("Invalid data format" :: String)]
        _ -> return $ object ["error" .= ("Missing required fields" :: String)]
handleAvailableOutputsWithFailedNodes _ = return $ object ["error" .= ("Invalid input format" :: String)]

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