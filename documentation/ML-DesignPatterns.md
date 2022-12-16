# Machine Learning Design Patterns

- Transfer Learning Design Pattern
- Cascade Design Pattern
- Embeddings Design Pattern
- Rebalancing Design Pattern - Imbalanced datasets 
- Distribution Strategy Design Pattern
- Model Versioning Design Pattern
- Checkpoints Design Pattern
- Transform Design Pattern
- Explainable Predictions Design Pattern
- Heuristic Benchmark Design Pattern
- Fairness Lens Design Pattern
- Reframing Design Pattern
- Ensembles Design Pattern

- Data representation
- Input data - Real world data
- Features - Transformed data - Input to model
- Do not throw away outliers

- Responsible AI

## Cascade Design Pattern
- Divide machine learning problem into number of small machine learning problems

## Rebalancing Design Pattern
- Imbalanced datasets 
  - Majority of samples for specific class 
  - Far fewer samples for other class or classes
- Examples - Disease detection, fraud detection
- Misleading accuracy - Solution - Appropriate evaluation metric - Confusion matrix, F1 score  
- Outlier samples 
- Downsampling and upsampling - Change balance of dataset - Data augmentation - Use Ensemble Design Pattern to combine results 
- Weighted classes - Change optimization funtion 
- Alternatives
  - Reframing Design Pattern

## Reframing Design Pattern
- Reframe regression problem as classification problem 
  - Normal distribution dataset - Equal importance to various classes or bins - Better performance
  - Multi modal distribution dataset - Inheritant classification - Better performance  
- Reframe classification problem as regression problem 
  - Predicted value vs predicted class
- Alternatives
  - Multitask learning - Classification + regression - Classification get bucket or class - Regression improve exact value
   
## Ensembles Design Pattern
- Combine multiple machine learning models
- Aggregate their predictions to provide single prediction
- Better performance
- Ensemble approaches
  - Bagging
  - Boosting
  - Stacking

### Bagging 
- Bootstrap aggregation
- Parallel ensembling
- Reduce high variance
- Same model and algorithm - Different datasets
- Bootstrap - Dataset - Random sampling with replacement
- Aggregation - Ensemble method - Majority vote for classification - Average for regression
- Out of bag dataset
- Train models with number of datasets
- Wisdom of Crowds
- Example
  - Random forest using decision trees
  - Dropout for neural network

### Boosting
- Reduce bias
- Number of weak learners to single strong learner
- Iteratively reduce prediction error and improve model

### Stacking
- Number of models - Same complete dataset - Number of predictions
- Meta-model - Input as predictions of models - Output learned single prediction
- Combine the best of both bagging and boosting
- Extension to simple model averaging

## Distribution Strategy Design Pattern
- Data parallelism and model parallelism
- Synchronous training and asynchronous training
 
## Transfer Learning Design Pattern
- Progressive fine tuning

## Hyperparameter Tuning
- Model parameters - Learned by training
- Hyperparameter - Set by user
- Learning rate, number of epochs, number of layers and so on
- Manual tuning, grid search, and random search
- Model architecture hyperparameters, and model training hyperparameters
- Non-linear optimization applied to non-differentiable problem
- Bayesian optimization
  - Black box function + Surrogate function
  - Objective function v/s Surrogate function
  - Surrogate function - Optimal hyperparameters - Before completing training run
 
