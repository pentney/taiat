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
import Data.Aeson (ToJSON, FromJSON)

-- Data structures for the path planning implementation
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

-- Find all nodes that produce a specific output (with parameter matching)
nodesProducingOutput :: [Node] -> AgentData -> [Node]
nodesProducingOutput nodes output = filter producesOutput nodes
  where
    producesOutput node = any (agentDataMatch output) (nodeOutputs' node)

-- Calculate specificity score for parameter matching
calculateSpecificityScore :: Map Text Text -> Map Text Text -> Int
calculateSpecificityScore desiredParams producedParams =
    length $ filter (\(k, v) -> Map.lookup k producedParams == Just v) (Map.toList desiredParams)

-- Find the most specific producer with context from desired outputs
findMostSpecificProducerWithContext :: [Node] -> AgentData -> [Node] -> [AgentData] -> [Node]
findMostSpecificProducerWithContext _ _ [] _ = []
findMostSpecificProducerWithContext _ _ [singleNode] _ = [singleNode]
findMostSpecificProducerWithContext nodes desiredOutput candidates desiredOutputs =
    let validCandidates = filter (\p -> canNodeInputsBeSatisfied nodes p) candidates
    in case findBestProducerWithContext validCandidates desiredOutput desiredOutputs (-1) Nothing of
        Just bestProducer -> [bestProducer]
        Nothing -> []
  where
    findBestProducerWithContext [] _ _ _ bestProducer = bestProducer
    findBestProducerWithContext (node:rest) desiredOutput desiredOutputs bestScore bestProducer =
        let score = calculateScoreWithContext node desiredOutput desiredOutputs
        in if score > bestScore
           then findBestProducerWithContext rest desiredOutput desiredOutputs score (Just node)
           else findBestProducerWithContext rest desiredOutput desiredOutputs bestScore bestProducer
    
    calculateScoreWithContext node desiredOutput desiredOutputs =
        let matchingOutputs = filter (agentDataMatch desiredOutput) (nodeOutputs' node)
        in case matchingOutputs of
            (output:_) -> 
                let baseScore = calculateSpecificityScore (agentDataParameters' desiredOutput) (agentDataParameters' output)
                    -- Bonus score if this output matches a desired output
                    bonusScore = if any (\desired -> agentDataMatch desired output) desiredOutputs
                                then 1000  -- High bonus for exact matches
                                else 0
                in baseScore + bonusScore
            [] -> -1

-- Find all nodes that produce a specific output name (legacy - for backward compatibility)
nodesProducingOutputName :: [Node] -> Text -> [Node]
nodesProducingOutputName nodes outputName = filter producesOutputName nodes
  where
    producesOutputName node = any (\output -> agentDataName' output == outputName) (nodeOutputs' node)

-- Get all dependencies for a node (nodes that produce its inputs with parameter matching)
nodeDependencies :: [Node] -> Node -> [AgentData] -> [Node]
nodeDependencies nodes node desiredOutputs = concatMap findProducers (nodeInputs' node)
  where
    findProducers input =
        let producers = nodesProducingOutput nodes input
        in case findMostSpecificProducerWithContext nodes input producers desiredOutputs of
            [producer] -> [producer]
            [] -> []
            _ -> []  -- Don't include any if multiple and none is most specific

-- Check if a node is ready to execute (all dependencies satisfied)
nodeReady :: [Node] -> Node -> [Node] -> [AgentData] -> Bool
nodeReady nodes node executedNodes desiredOutputs =
    let dependencies = nodeDependencies nodes node desiredOutputs
    in all (`elem` executedNodes) dependencies

-- Check if a node's inputs can be satisfied by any nodes in the graph
canNodeInputsBeSatisfied :: [Node] -> Node -> Bool
canNodeInputsBeSatisfied nodes node =
    all inputCanBeSatisfied (nodeInputs' node)
  where
    inputCanBeSatisfied input =
        let producers = nodesProducingOutput nodes input
        in if null producers
           then -- Check if this input is truly external (no producers with matching parameters)
               let inputName = agentDataName' input
                   potentialProducers = filter (\n -> any (\o -> agentDataName' o == inputName) (nodeOutputs' n)) nodes
                   -- Only consider it external if no potential producers have matching parameters
                   matchingProducers = filter (\n -> any (agentDataMatch input) (nodeOutputs' n)) potentialProducers
               in null matchingProducers  -- External if no producers with matching parameters
           else -- Check if any producer can actually satisfy this input
               any (\producer -> any (agentDataMatch input) (nodeOutputs' producer)) producers

-- Recursively collect required nodes from a list of agent_data outputs
requiredNodesFromOutputs :: [Node] -> [AgentData] -> [AgentData] -> [Node] -> Maybe [Node]
requiredNodesFromOutputs _ [] _ acc = Just acc
requiredNodesFromOutputs nodes (output:rest) desiredOutputs acc =
    let producers = nodesProducingOutput nodes output
    in if null producers
       then Nothing  -- Return Nothing instead of error
       else case findMostSpecificProducerWithContext nodes output producers desiredOutputs of
            [producer] -> 
                -- Check if the producer can actually be executed (all inputs satisfied)
                if canNodeInputsBeSatisfied nodes producer
                   then case requiredNodesFromNode nodes producer desiredOutputs acc of
                        Just accWithNode -> requiredNodesFromOutputs nodes rest desiredOutputs accWithNode
                        Nothing -> Nothing
                   else Nothing  -- Producer cannot be executed
            [] -> Nothing  -- No suitable producer found
            _ -> Nothing  -- Multiple producers but none is most specific

-- Recursively collect required nodes starting from a node
requiredNodesFromNode :: [Node] -> Node -> [AgentData] -> [Node] -> Maybe [Node]
requiredNodesFromNode nodes node desiredOutputs acc
    | node `elem` acc = Just acc
    | otherwise =
        let inputs = nodeInputs' node
            -- Check if all inputs can be satisfied with matching parameters
            allInputsSatisfiable = all (\input -> 
                let producers = nodesProducingOutput nodes input
                in not (null producers)) inputs
        in if allInputsSatisfiable
           then case requiredNodesFromOutputs nodes inputs desiredOutputs acc of
                Just accWithDeps -> Just (accWithDeps ++ [node])
                Nothing -> Nothing
           else Nothing  -- Node has inputs that cannot be satisfied

-- Find all nodes needed to produce the desired outputs
requiredNodes :: [Node] -> [AgentData] -> Maybe [Node]
requiredNodes nodes desiredOutputs = 
    case requiredNodesFromOutputs nodes desiredOutputs desiredOutputs [] of
        Just nodes' -> Just $ removeDuplicates nodes'
        Nothing -> Nothing

-- Remove duplicates while preserving order
removeDuplicates :: Eq a => [a] -> [a]
removeDuplicates [] = []
removeDuplicates (h:t) = h : removeDuplicates (filter (/= h) t)

-- Topological sort to determine execution order
topologicalSort :: [Node] -> [AgentData] -> [Node]
topologicalSort nodes desiredOutputs = topologicalSortRecursive nodes [] desiredOutputs

topologicalSortRecursive :: [Node] -> [Node] -> [AgentData] -> [Node]
topologicalSortRecursive [] sortedNodes _ = sortedNodes
topologicalSortRecursive nodes currentSorted desiredOutputs =
    let readyNodes = filter (\node -> nodeReady nodes node currentSorted desiredOutputs) nodes
    in if null readyNodes
       then currentSorted  -- No nodes ready - might indicate cycle or missing dependencies
       else let updatedSorted = currentSorted ++ readyNodes
                remainingNodes = filter (`notElem` readyNodes) nodes
            in topologicalSortRecursive remainingNodes updatedSorted desiredOutputs

-- Helper predicate to extract node names from a list of nodes
extractNodeNames :: [Node] -> [Text]
extractNodeNames = map nodeName'

-- Check if a node is needed in the execution path
nodeIsNeeded :: [Node] -> Node -> [AgentData] -> [Node] -> Bool
nodeIsNeeded nodes node desiredOutputs requiredNodes =
    let nodeOutputs = nodeOutputs' node
        
        -- Check if this node produces any of the desired outputs
        producesDesiredOutput = any (\desired -> any (agentDataMatch desired) nodeOutputs) desiredOutputs
        
        -- Check if this node's outputs are consumed by other required nodes
        outputsConsumed = any (\output -> 
            any (\consumer -> 
                consumer /= node && 
                any (\input -> agentDataMatch input output) (nodeInputs' consumer)) requiredNodes) nodeOutputs
        
        -- For nodes with parameterized outputs, check if they match specific desired outputs
        -- or if they're needed by consumers that require specific parameters
        parameterizedOutputsNeeded = any (\output -> 
            let outputParams = agentDataParameters' output
            in not (Map.null outputParams) && 
               (any (\desired -> 
                    agentDataName' desired == agentDataName' output && 
                    agentDataParameters' desired == outputParams) desiredOutputs ||
                any (\consumer -> 
                    consumer /= node && 
                    any (\input -> 
                        agentDataName' input == agentDataName' output && 
                        parametersSubset (agentDataParameters' input) outputParams) (nodeInputs' consumer)) requiredNodes)) nodeOutputs
        
        -- For nodes with non-parameterized outputs, they're needed if they produce desired outputs
        -- or if they're consumed by other required nodes
        nonParameterizedOutputsNeeded = any (\output -> 
            Map.null (agentDataParameters' output) && 
            (any (\desired -> agentDataMatch desired output) desiredOutputs ||
             any (\consumer -> 
                 consumer /= node && 
                 any (\input -> agentDataMatch input output) (nodeInputs' consumer)) requiredNodes)) nodeOutputs
        
    in producesDesiredOutput || outputsConsumed || parameterizedOutputsNeeded || nonParameterizedOutputsNeeded

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
                sortedNodes = topologicalSort filteredNodes desiredOutputs
            in extractNodeNames sortedNodes
        Nothing -> []  -- Return empty list instead of crashing

-- Plan alternative execution path excluding failed nodes
planAlternativeExecutionPath :: AgentGraphNodeSet -> [AgentData] -> [Text] -> [Text]
planAlternativeExecutionPath nodeSet desiredOutputs failedNodeNames =
    let nodes = agentGraphNodeSetNodes nodeSet
        -- Filter out failed nodes
        availableNodes = filter (\node -> not (nodeName' node `elem` failedNodeNames)) nodes
        availableNodeSet = AgentGraphNodeSet availableNodes
    in planExecutionPath availableNodeSet desiredOutputs

-- Plan multiple alternative paths with different strategies
planMultipleAlternativePaths :: AgentGraphNodeSet -> [AgentData] -> [Text] -> [[Text]]
planMultipleAlternativePaths nodeSet desiredOutputs failedNodeNames =
    let nodes = agentGraphNodeSetNodes nodeSet
        -- Filter out failed nodes
        availableNodes = filter (\node -> not (nodeName' node `elem` failedNodeNames)) nodes
        availableNodeSet = AgentGraphNodeSet availableNodes
        
        -- Try different strategies for alternative paths
        strategies = [
            -- Strategy 1: Standard planning
            \nodeSet' outputs -> planExecutionPath nodeSet' outputs,
            -- Strategy 2: Planning with minimal dependencies (could be implemented)
            \nodeSet' outputs -> planExecutionPath nodeSet' outputs,
            -- Strategy 3: Planning with different node selection criteria (could be implemented)
            \nodeSet' outputs -> planExecutionPath nodeSet' outputs
        ]
        
        -- Apply each strategy
        paths = map (\strategy -> strategy availableNodeSet desiredOutputs) strategies
        
        -- Filter out empty paths and duplicates
        validPaths = filter (\path -> not (null path)) paths
        uniquePaths = removeDuplicates validPaths
    in take 3 uniquePaths  -- Return up to 3 alternative paths

-- Predicate to validate that all desired outputs can be produced
validateOutputs :: AgentGraphNodeSet -> [AgentData] -> Bool
validateOutputs nodeSet desiredOutputs =
    let nodes = agentGraphNodeSetNodes nodeSet
    in all (\output -> 
        let outputName = agentDataName' output
            producingNodes = nodesProducingOutputName nodes outputName
        in not (null producingNodes)) desiredOutputs

-- Validate outputs with failed nodes excluded
validateOutputsWithFailedNodes :: AgentGraphNodeSet -> [AgentData] -> [Text] -> Bool
validateOutputsWithFailedNodes nodeSet desiredOutputs failedNodeNames =
    let nodes = agentGraphNodeSetNodes nodeSet
        -- Filter out failed nodes
        availableNodes = filter (\node -> not (nodeName' node `elem` failedNodeNames)) nodes
        availableNodeSet = AgentGraphNodeSet availableNodes
    in validateOutputs availableNodeSet desiredOutputs

-- Predicate to find all outputs that can be produced by the node set
availableOutputs :: AgentGraphNodeSet -> [AgentData]
availableOutputs nodeSet =
    let nodes = agentGraphNodeSetNodes nodeSet
        allOutputs = concatMap nodeOutputs' nodes
    in removeDuplicates allOutputs

-- Get available outputs excluding failed nodes
availableOutputsWithFailedNodes :: AgentGraphNodeSet -> [Text] -> [AgentData]
availableOutputsWithFailedNodes nodeSet failedNodeNames =
    let nodes = agentGraphNodeSetNodes nodeSet
        -- Filter out failed nodes
        availableNodes = filter (\node -> not (nodeName' node `elem` failedNodeNames)) nodes
        availableNodeSet = AgentGraphNodeSet availableNodes
    in availableOutputs availableNodeSet

-- Check for circular dependencies
hasCircularDependencies :: AgentGraphNodeSet -> Bool
hasCircularDependencies nodeSet =
    let nodes = agentGraphNodeSetNodes nodeSet
        sortedNodes = topologicalSort nodes []  -- Empty desired outputs for circular dependency check
    in length sortedNodes /= length nodes 