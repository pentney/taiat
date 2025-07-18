{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module PathPlanner where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.List (find, nub, sortBy)
import Data.Maybe (fromMaybe, catMaybes)
import Data.Ord (comparing)
import GHC.Generics (Generic)
import Data.Aeson (ToJSON, FromJSON, encode, decode)
import qualified Data.ByteString.Lazy as BS
import Data.Time.Clock (getCurrentTime, diffUTCTime)

-- Data structures matching the Prolog implementation
data AgentData = AgentData
    { agentDataName :: Text
    , agentDataParameters :: Map Text Text
    , agentDataDescription :: Text
    , agentDataData :: Maybe Text
    } deriving (Show, Eq, Ord, Generic)

instance ToJSON AgentData
instance FromJSON AgentData

data Node = Node
    { nodeName :: Text
    , nodeDescription :: Text
    , nodeInputs :: [AgentData]
    , nodeOutputs :: [AgentData]
    } deriving (Show, Eq, Ord, Generic)

instance ToJSON Node
instance FromJSON Node

data AgentGraphNodeSet = AgentGraphNodeSet
    { agentGraphNodeSetNodes :: [Node]
    } deriving (Show, Eq, Generic)

instance ToJSON AgentGraphNodeSet
instance FromJSON AgentGraphNodeSet

-- Helper functions for working with agent data
agentDataName' :: AgentData -> Text
agentDataName' = agentDataName

agentDataParameters' :: AgentData -> Map Text Text
agentDataParameters' = agentDataParameters

agentDataDescription' :: AgentData -> Text
agentDataDescription' = agentDataDescription

-- Helper functions for working with nodes
nodeName' :: Node -> Text
nodeName' = nodeName

nodeDescription' :: Node -> Text
nodeDescription' = nodeDescription

nodeInputs' :: Node -> [AgentData]
nodeInputs' = nodeInputs

nodeOutputs' :: Node -> [AgentData]
nodeOutputs' = nodeOutputs

-- Check if parameters1 is a subset of parameters2
parametersSubset :: Map Text Text -> Map Text Text -> Bool
parametersSubset params1 params2 = all (\(k, v) -> Map.lookup k params2 == Just v) (Map.toList params1)

-- Check if two agent data items match (name and parameters)
-- This allows for more flexible matching where desired parameters can be a subset
agentDataMatch :: AgentData -> AgentData -> Bool
agentDataMatch (AgentData name1 params1 _ _) (AgentData name2 params2 _ _) =
    name1 == name2 && parametersSubset params1 params2

-- More flexible parameter matching that allows for subset matching in both directions
agentDataMatchFlexible :: AgentData -> AgentData -> Bool
agentDataMatchFlexible (AgentData name1 params1 _ _) (AgentData name2 params2 _ _) =
    name1 == name2 && (parametersSubset params1 params2 || parametersSubset params2 params1 || params1 == params2)

-- Find all nodes that produce a specific output (with parameter matching)
nodesProducingOutput :: [Node] -> AgentData -> [Node]
nodesProducingOutput nodes output = filter producesOutput nodes
  where
    producesOutput node = any (agentDataMatch output) (nodeOutputs' node)

-- Calculate specificity score for parameter matching
calculateSpecificityScore :: Map Text Text -> Map Text Text -> Int
calculateSpecificityScore desiredParams producedParams =
    length $ filter (\(k, v) -> Map.lookup k producedParams == Just v) (Map.toList desiredParams)

-- Find the most specific producer among multiple candidates
findMostSpecificProducer :: [Node] -> AgentData -> [Node] -> [Node]
findMostSpecificProducer _ _ [] = []
findMostSpecificProducer _ _ [singleNode] = [singleNode]
findMostSpecificProducer nodes desiredOutput candidates =
    case findBestProducer candidates desiredOutput (-1) Nothing of
        Just bestProducer -> [bestProducer]
        Nothing -> []
  where
    findBestProducer [] _ _ bestProducer = bestProducer
    findBestProducer (node:rest) desiredOutput bestScore bestProducer =
        let score = calculateScore node desiredOutput
        in if score > bestScore
           then findBestProducer rest desiredOutput score (Just node)
           else findBestProducer rest desiredOutput bestScore bestProducer
    
    calculateScore node desiredOutput =
        let matchingOutputs = filter (agentDataMatch desiredOutput) (nodeOutputs' node)
        in case matchingOutputs of
            (output:_) -> calculateSpecificityScore (agentDataParameters' desiredOutput) (agentDataParameters' output)
            [] -> -1

-- Find all nodes that produce a specific output name (legacy - for backward compatibility)
nodesProducingOutputName :: [Node] -> Text -> [Node]
nodesProducingOutputName nodes outputName = filter producesOutputName nodes
  where
    producesOutputName node = any (\output -> agentDataName' output == outputName) (nodeOutputs' node)

-- Get all dependencies for a node (nodes that produce its inputs with parameter matching)
nodeDependencies :: [Node] -> Node -> [Node]
nodeDependencies nodes node = concatMap findProducers (nodeInputs' node)
  where
    findProducers input =
        let producers = nodesProducingOutput nodes input
        in case findMostSpecificProducer nodes input producers of
            [producer] -> [producer]
            [] -> []
            _ -> take 1 producers  -- Take first if multiple

-- Check if a node is ready to execute (all dependencies satisfied)
nodeReady :: [Node] -> Node -> [Node] -> Bool
nodeReady nodes node executedNodes =
    let dependencies = nodeDependencies nodes node
    in all (`elem` executedNodes) dependencies

-- Check if a node's inputs can be satisfied by any nodes in the graph
canNodeInputsBeSatisfied :: [Node] -> Node -> Bool
canNodeInputsBeSatisfied nodes node =
    all inputCanBeSatisfied (nodeInputs' node)
  where
    inputCanBeSatisfied input =
        let producers = nodesProducingOutput nodes input
        in if null producers
           then -- Check if this input is truly external
               let inputName = agentDataName' input
                   potentialProducers = filter (\n -> any (\o -> agentDataName' o == inputName) (nodeOutputs' n)) nodes
               in null potentialProducers  -- External if no potential producers
           else -- Check if any producer can actually satisfy this input
               any (\producer -> any (agentDataMatch input) (nodeOutputs' producer)) producers

-- Recursively collect required nodes from a list of agent_data outputs
requiredNodesFromOutputs :: [Node] -> [AgentData] -> [Node] -> Maybe [Node]
requiredNodesFromOutputs _ [] acc = Just acc
requiredNodesFromOutputs nodes (output:rest) acc =
    let producers = nodesProducingOutput nodes output
    in if null producers
       then Nothing  -- Return Nothing instead of error
       else case find (\p -> canNodeInputsBeSatisfied nodes p) producers of
            Just producer -> case requiredNodesFromNode nodes producer acc of
                Just accWithNode -> requiredNodesFromOutputs nodes rest accWithNode
                Nothing -> Nothing
            Nothing -> Nothing  -- Return Nothing instead of error

-- Recursively collect required nodes starting from a node
requiredNodesFromNode :: [Node] -> Node -> [Node] -> Maybe [Node]
requiredNodesFromNode nodes node acc
    | node `elem` acc = Just acc
    | otherwise =
        let inputs = nodeInputs' node
            validInputs = filter (\input -> 
                let producers = nodesProducingOutput nodes input
                in not (null producers) && any (\p -> any (agentDataMatch input) (nodeOutputs' p)) producers) inputs
        in case requiredNodesFromOutputs nodes validInputs acc of
            Just accWithDeps -> Just (accWithDeps ++ [node])
            Nothing -> Nothing

-- Find all nodes needed to produce the desired outputs
requiredNodes :: [Node] -> [AgentData] -> Maybe [Node]
requiredNodes nodes desiredOutputs = 
    case requiredNodesFromOutputs nodes desiredOutputs [] of
        Just nodes' -> Just $ removeDuplicates nodes'
        Nothing -> Nothing

-- Remove duplicates while preserving order
removeDuplicates :: Eq a => [a] -> [a]
removeDuplicates [] = []
removeDuplicates (h:t) = h : removeDuplicates (filter (/= h) t)

-- Topological sort to determine execution order
topologicalSort :: [Node] -> [Node]
topologicalSort nodes = topologicalSortRecursive nodes []

topologicalSortRecursive :: [Node] -> [Node] -> [Node]
topologicalSortRecursive [] sortedNodes = sortedNodes
topologicalSortRecursive nodes currentSorted =
    let readyNodes = filter (\node -> nodeReady nodes node currentSorted) nodes
    in if null readyNodes
       then currentSorted  -- No nodes ready - might indicate cycle or missing dependencies
       else let updatedSorted = currentSorted ++ readyNodes
                remainingNodes = filter (`notElem` readyNodes) nodes
            in topologicalSortRecursive remainingNodes updatedSorted

-- Helper predicate to extract node names from a list of nodes
extractNodeNames :: [Node] -> [Text]
extractNodeNames = map nodeName'

-- Check if a node is needed in the execution path
nodeIsNeeded :: [Node] -> Node -> [AgentData] -> [Node] -> Bool
nodeIsNeeded nodes node desiredOutputs requiredNodes =
    let nodeOutputs = nodeOutputs' node
        producesDesiredOutput = any (\desired -> any (agentDataMatch desired) nodeOutputs) desiredOutputs
        consumers = filter (\consumer -> 
            consumer /= node && 
            any (\input -> any (agentDataMatch input) nodeOutputs) (nodeInputs' consumer)) requiredNodes
    in producesDesiredOutput || not (null consumers)

-- Filter nodes to only include those that are actually needed
filterRequiredNodes :: [Node] -> [AgentData] -> [Node] -> [Node]
filterRequiredNodes nodes desiredOutputs allRequiredNodes =
    filter (\node -> nodeIsNeeded nodes node desiredOutputs allRequiredNodes) allRequiredNodes

-- Main predicate: plan execution path
planExecutionPath :: AgentGraphNodeSet -> [AgentData] -> [Text]
planExecutionPath nodeSet desiredOutputs =
    let nodes = agentGraphNodeSetNodes nodeSet
    in case requiredNodes nodes desiredOutputs of
        Just requiredNodes' ->
            let uniqueRequiredNodes = removeDuplicates requiredNodes'
                filteredNodes = filterRequiredNodes nodes desiredOutputs uniqueRequiredNodes
                sortedNodes = topologicalSort filteredNodes
            in extractNodeNames sortedNodes
        Nothing -> []  -- Return empty list instead of crashing

-- Predicate to validate that all desired outputs can be produced
validateOutputs :: AgentGraphNodeSet -> [AgentData] -> Bool
validateOutputs nodeSet desiredOutputs =
    let nodes = agentGraphNodeSetNodes nodeSet
    in all (\output -> 
        let outputName = agentDataName' output
            producingNodes = nodesProducingOutputName nodes outputName
        in not (null producingNodes)) desiredOutputs

-- Predicate to find all outputs that can be produced by the node set
availableOutputs :: AgentGraphNodeSet -> [AgentData]
availableOutputs nodeSet =
    let nodes = agentGraphNodeSetNodes nodeSet
        allOutputs = concatMap nodeOutputs' nodes
    in removeDuplicates allOutputs

-- Check for circular dependencies
hasCircularDependencies :: AgentGraphNodeSet -> Bool
hasCircularDependencies nodeSet =
    let nodes = agentGraphNodeSetNodes nodeSet
        sortedNodes = topologicalSort nodes
    in length sortedNodes /= length nodes

-- Performance measurement helper
measurePerformance :: IO a -> IO (a, Double)
measurePerformance action = do
    start <- getCurrentTime
    result <- action
    end <- getCurrentTime
    let duration = diffUTCTime end start
    return (result, realToFrac duration)

-- JSON serialization helpers
encodeToJSON :: ToJSON a => a -> BS.ByteString
encodeToJSON = encode

decodeFromJSON :: FromJSON a => BS.ByteString -> Maybe a
decodeFromJSON = decode 