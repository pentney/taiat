{-# LANGUAGE OverloadedStrings #-}
import PathPlanner
import qualified Data.Map as Map

main :: IO ()
main = do
    -- Create nodes matching the full ML workflow structure
    let loadDataset = Node "load_dataset" "Load the dataset" 
            [AgentData "dataset_name" Map.empty "" Nothing] 
            [AgentData "dataset" Map.empty "" Nothing]
    
    let randomForest = Node "random_forest" "Train a random forest model" 
            [AgentData "dataset" Map.empty "" Nothing] 
            [AgentData "model" (Map.fromList [("model_type", "random_forest")]) "" Nothing]
    
    let logisticRegression = Node "logistic_regression" "Train a logistic regression model" 
            [AgentData "dataset" Map.empty "" Nothing] 
            [AgentData "model" (Map.fromList [("model_type", "logistic_regression")]) "" Nothing, AgentData "model_params" Map.empty "" Nothing]
    
    let nearestNeighbors = Node "nearest_neighbors" "Train a nearest neighbors model" 
            [AgentData "dataset" Map.empty "" Nothing] 
            [AgentData "model" (Map.fromList [("model_type", "nearest_neighbors")]) "" Nothing]
    
    let clustering = Node "clustering" "Train a clustering model" 
            [AgentData "dataset" Map.empty "" Nothing] 
            [AgentData "model" (Map.fromList [("model_type", "clustering")]) "" Nothing]
    
    let predict = Node "predict_and_generate_report" "Make a prediction and generate a report" 
            [AgentData "model" Map.empty "" Nothing] 
            [AgentData "model_preds" Map.empty "" Nothing, AgentData "model_report" Map.empty "" Nothing]
    
    let resultsAnalysis = Node "results_analysis" "Analyze the results" 
            [AgentData "dataset_name" Map.empty "" Nothing, AgentData "model_report" Map.empty "" Nothing] 
            [AgentData "summary" Map.empty "" Nothing]
    
    let nodes = [loadDataset, randomForest, logisticRegression, nearestNeighbors, clustering, predict, resultsAnalysis]
    
    -- Add external input as a virtual node
    let externalNode = Node "dataset_name_external" "External input: dataset_name" [] 
            [AgentData "dataset_name" Map.empty "" Nothing]
    let extendedNodeSet = AgentGraphNodeSet (externalNode : nodes)
    
    let desiredOutputs = [AgentData "model" (Map.fromList [("model_type", "random_forest")]) "" Nothing,
                         AgentData "model_report" Map.empty "" Nothing]
    

    putStrLn "Testing full ML workflow..."
    let result = planExecutionPath extendedNodeSet desiredOutputs
    putStrLn $ "Result: " ++ show result
    
    -- Test individual components
    putStrLn "\nTesting individual components:"
    
    -- Test requiredNodes
    let required = requiredNodes nodes desiredOutputs
    putStrLn $ "Required nodes: " ++ show (fmap (map nodeName') required)
    
    -- Test what happens when we add each node to the required list
    putStrLn "\nTesting node addition:"
    case required of
        Just reqNodes -> do
            putStrLn $ "Initial required nodes: " ++ show (map nodeName' reqNodes)
            -- Test what happens when we add predict node
            let withPredict = reqNodes ++ [predict]
            putStrLn $ "With predict node: " ++ show (map nodeName' withPredict)
            -- Test dependencies of predict node
            let predictDeps = nodeDependencies nodes predict desiredOutputs
            putStrLn $ "Predict dependencies: " ++ show (map nodeName' predictDeps)
        Nothing -> putStrLn "No required nodes found" 