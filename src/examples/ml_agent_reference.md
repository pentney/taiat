# ML Agent Reference Documentation

This document provides a complete specification of all AgentGraphNodes in the ML Agent workflow, formatted for use by the TaiatGraphArchitect to rebuild the AgentGraphNodeSet.

## AgentGraphNodeSet Specification

### 1. load_dataset

**Name:** `load_dataset`  
**Description:** Load the dataset  
**Function:** `load_dataset`  
**Inputs:**
- `dataset_name` (string) - Name of the dataset to download from Kaggle

**Outputs:**
- `dataset` (pandas.DataFrame) - The loaded and preprocessed dataset
- `dataset_test` (pandas.DataFrame) - Test split of the dataset
- `label` (string) - Name of the target column
- `model_params` (dict) - Empty dictionary for model parameters

**Function Details:**
- Downloads dataset from Kaggle using the provided dataset name
- Applies one-hot encoding to categorical columns
- Splits data into training (80%) and test (20%) sets
- Automatically detects the target column (looks for 'class', 'target', 'diabetes', or uses last column)
- Returns preprocessed dataset ready for model training

### 2. logistic_regression

**Name:** `logistic_regression`  
**Description:** Train a logistic regression model  
**Function:** `logistic_regression`  
**Inputs:**
- `dataset` (pandas.DataFrame) - Training dataset with features and target
- `model_params` (dict) - Model hyperparameters (optional)

**Outputs:**
- `model` (LogisticRegression) - Trained logistic regression model with parameters:
  - `model_type`: "logistic_regression"

**Function Details:**
- Fits LogisticRegression from scikit-learn
- Uses all columns except the target column as features
- Returns trained model ready for prediction

### 3. random_forest

**Name:** `random_forest`  
**Description:** Train a random forest model  
**Function:** `random_forest`  
**Inputs:**
- `dataset` (pandas.DataFrame) - Training dataset with features and target
- `model_params` (dict) - Model hyperparameters (optional)

**Outputs:**
- `model` (RandomForestClassifier) - Trained random forest model with parameters:
  - `model_type`: "random_forest"

**Function Details:**
- Fits RandomForestClassifier from scikit-learn
- Uses all columns except the target column as features
- Returns trained model ready for prediction

### 4. nearest_neighbors

**Name:** `nearest_neighbors`  
**Description:** Train a nearest neighbors model  
**Function:** `nearest_neighbors`  
**Inputs:**
- `dataset` (pandas.DataFrame) - Training dataset with features and target
- `model_params` (dict) - Model hyperparameters (optional)

**Outputs:**
- `model` (KNeighborsClassifier) - Trained K-nearest neighbors model with parameters:
  - `model_type`: "nearest_neighbors"

**Function Details:**
- Fits KNeighborsClassifier from scikit-learn
- Uses all columns except the target column as features
- Returns trained model ready for prediction

### 5. clustering

**Name:** `clustering`  
**Description:** Train a clustering model  
**Function:** `clustering`  
**Inputs:**
- `dataset` (pandas.DataFrame) - Training dataset with features and target
- `model_params` (dict) - Model hyperparameters (optional)

**Outputs:**
- `model` (KMeans) - Trained K-means clustering model with parameters:
  - `model_type`: "clustering"

**Function Details:**
- Fits KMeans from scikit-learn
- Uses all columns except the target column as features
- Returns trained model ready for prediction

### 6. predict_and_generate_report

**Name:** `predict_and_generate_report`  
**Description:** Make a prediction and generate a report  
**Function:** `predict_and_generate_report`  
**Inputs:**
- `model` - Trained machine learning model
- `dataset_test` (pandas.DataFrame) - Test dataset (inherited from previous state)

**Outputs:**
- `model_preds` - Model predictions on test data
- `model_report` (string) - Classification report with precision, recall, f1-score

**Function Details:**
- Uses trained model to make predictions on test dataset
- Generates classification report using sklearn.metrics.classification_report
- Returns predictions and detailed performance metrics

### 7. results_analysis

**Name:** `results_analysis`  
**Description:** Analyze the results  
**Function:** `results_analysis`  
**Inputs:**
- `dataset_name` (string) - Name of the dataset used
- `model_report` (string) - Performance report from the model
- `model_name` (string) - Name/type of the model used (inherited from previous state)

**Outputs:**
- `summary` (string) - AI-generated summary of dataset, model, and performance

**Function Details:**
- Uses OpenAI GPT-4o-mini to analyze the results
- Generates human-readable summary of the entire workflow
- Combines dataset information, model details, and performance metrics

## State Management

The workflow uses `MLAgentState` which extends the base `State` class with the following structure:

```python
class MLAgentState(State):
    model: str                    # Trained model object
    model_name: str              # Name/type of the model
    model_params: dict           # Model hyperparameters
    model_results: dict          # Model results and metrics
    dataset: pd.DataFrame        # Training dataset
    dataset_test: pd.DataFrame   # Test dataset
    label: str                   # Target column name
    model_preds: array           # Model predictions
    model_report: str            # Performance report
    summary: str                 # AI-generated analysis
```

## Dependencies

The workflow requires the following Python packages:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models and metrics
- `kaggle` - Dataset download
- `langchain_openai` - AI-powered analysis
- `taiat.base` - Core Taiat framework

## Workflow Flow

1. **Data Loading**: `load_dataset` downloads and preprocesses the dataset
2. **Model Training**: One of the model training agents (`logistic_regression`, `random_forest`, `nearest_neighbors`, or `clustering`) trains a model
3. **Evaluation**: `predict_and_generate_report` evaluates the model on test data
4. **Analysis**: `results_analysis` provides AI-generated insights

## Notes for TaiatGraphArchitect

- The `load_dataset` agent must run first as it provides the foundational data
- Model training agents are mutually exclusive - only one should be executed per workflow
- The `predict_and_generate_report` agent requires both a trained model and test data
- The `results_analysis` agent depends on the performance report and dataset information
- All agents maintain state through the `MLAgentState` object
- The workflow automatically handles data preprocessing including one-hot encoding and train-test splitting 