# AI Concepts

## Light Detection And Ranging
- LiDAR
- Light

## RADAR
- Radio Detection And Ranging
- Echo radar
- Doppler radar - Radio waves 

## Region Based Convolutional Neural Network 
- R-CNN
- Two stage
- Most accurate object detection
- Slowest approach
- Region proposal network 
- Region classification network

## Machine Learning - Iteration
- Does the model do well on the training dataset ?
  - If NO 
    - High bias
    - Use bigger network architecture
    - Use better network architecture
  - If YES
    - Does the model do well on the test dataset ?
      - If NO 
        - High variance
        - Collect more dataset
        - [Collect more better dataset](https://datacentricai.org/)
        - Use data augmentation methods
        - Use regularization methods
      - If YES - Use the model for deployment
     
## Open Neural Network Exchange
- ONNX
- Open format for machine learning models
- Interchange models between various machine learning frameworks
- Define common set of operators - Building blocks of machine learning and deep learning

## Tensor Processing Units
- Quantization - Reduce Precision from 32 bit float to 8 bit interger
- D = A * B + C in one cycle
- 12 times more throughput than GPU

## [Deep Double Descent](https://openai.com/blog/deep-double-descent/)
- Performance first improves - First Descent
- Then it gets worse
- Then it improves again - Deep Double Descent
- Observed for - Model size, dataset size, or training time
- Reduce impact using regularization
- Design or select model size, dataset size, and training time 
  - Overcome First Descent 
  - Operate beyond Deep Double Descent   

## Deep Double Descent - Model Size
- Label noise - More effect

## Deep Double Descent - Dataset Size
- 

## Deep Double Descent - Number Of Epochs
- 

## [LIME - Image Classification](https://youtu.be/ENa-w65P1xM)
- Explanations for Image classification
- Divide original image into small parts
- Create number of smaple images using small parts
- Compute predictions for sample images
- Compute distances between original image and sample images - Cosine similarity
- Compute weights or importance of each sample image
- Fit explainable or linear model using sample images, predictions and weights
- Compute top features - Explanations for Image classification

## [Explainable Machine Learning]()
- [Explainable model does not provide its own explanation](https://towardsdatascience.com/interperable-vs-explainable-machine-learning-1fa525e12f48) - Example Deep neural network 
- Provide explanations for decisions made
- House price prediction
  - Input variables - Number of bedrooms, area of house, locality of house and so on
  - Output variable - Price of house
  - Explainable ML for predicted Price of house
    - Number of bedrooms - 25% important 
    - Area of house - 30% important 
    - Locality of house - 40% important
- Explainable ML - Important - NOT always necessary

## [Interpretable Machine Learning]()
- [Interpretable model provides its own explanation](https://towardsdatascience.com/interperable-vs-explainable-machine-learning-1fa525e12f48) - Example Decision tree 
- Cause and effect can be determined
- House price prediction
  - Input variables - Number of bedrooms, area of house, locality of house and so on
  - Output variable - Price of house
  - Interpretable ML for predicted Price of house
    - Change Number of bedrooms by 1, then Change predicted Price by 10K
    - Change Area of house by 100 sqm, then Change predicted Price by 5K
    - Change Locality of house to Prime area, the Change predicted Price by 20K
- Interpretable ML - Important and always necessary 
- Compare model performance

## Multitask Learning
- Parameter sharing for number of objectives
- Number of objectives - Number of optimization functions
- Better performance
- Different types
  - Hard parameter sharing - Model layers shared between all objectives
  - Soft parameter sharing - Each objective different model layers with regularization 
- Examples
  - Object detection - Object class as classification and object bounding box as regression - Better performance 

## Machine Learning Error 
- Irreducible error
  - Noise in dataset, bad training samples
- Model bias - High bias - Oversimplify mapping function - Underfit 
- Model variance - High variance - Learned too much - Overfit
- Low bias and low variance - Better performance
- Bias vs variance trade-off 
  - Increase model complexity - Decrease bias - Increase variance
  - Decrease model complexity - Decrease variance - Increase bias

## Covariance v/s Correlation
- Covariance measures how two variables are related to each other and how one would vary with respect to changes in the other variable. 
  - If the value is positive it means there is a direct relationship between the variables and one would increase or decrease with an increase or decrease in the base variable respectively.
- Correlation quantifies the relationship between two random variables and has only three specific values, i.e., 1, 0, and -1.
  - 1 denotes a positive relationship
  - -1 denotes a negative relationship
  - 0 denotes that the two variables are independent of each other

## Confusion Matrix
- TP - True Positive - Real True and Prediction True
- TN - True Negative - Real False and Prediction False
- FP - False Positive - Real False and Prediction True - Type one error
- FN - False Negative - Real True and Prediction False - Type two error

## Confusion Matrix - Metrics 
- Accuracy - (TP + TN) / (TP + TN + FP + FN) 
- Precision - TP / (TP + FP) - Increase precision - Increase threshold - Decrease False Positive - Accuracy for minority class
- Recall - TP / (TP + FN) - Increase Recall - Decrease threshold - Decrease False Negative - Measure coverage of minority class
- F1 score - Harmonic mean of Precision and Recall - (2* Precision * Recall ) / (Precision + Recall)

## Sensitivity
- Ability of model to correctly identify patients with disease
- TP / (TP + FN) 

## Specificity
- Ability of model to correctly identify patients without disease
- TN / (TN + FP)

## Linear Transformation
- Uniform distribution
- Normal distribution  
- Min-max scaling
- Clipping
- Z-score normalization 
  - Mean and standard deviation
  - Zero mean 
  - Normalized - Standard deviation

## Nonlinear transformation
- Skewed dataset
- High dynamic range dataset
- Logarithm or power function followed by linear transformations

## How many images are needed for image classification, object detection and so on ?
 - 150-500 images per class - Base accuracy
 - After 500 images - Accuracy saturates  
 - I don’t know or It depends up on dataset.

## Optimal learning rate 
- Topology of loss function 
- Model architecture 
- Dataset
