# Topics

## Artificial Intelligence
- [What is AI ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [AI Classification](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## Machine Learning
- [What is ML ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [ML Classification](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## Supervised Learning
- [What is Supervised Learning ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [Supervised Learning Examples](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## Unsupervised Learning
- [What is Unsupervised Learning ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [Unsupervised Learning Examples](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## Reinforcement Learning
- [What is Reinforcement Learning ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [Reinforcement Learning Terms](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [Reinforcement Learning Examples](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## [Semi Supervised Learning](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [What is Semi supervised Learning ?](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)
- [Semi Supervised Learning Examples](https://github.com/look4pritam/ArtificialIntelligence/blob/master/ArtificialIntelligence/ArtificialIntelligence.pptx)

## References
- [Artificial Intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)
- [AI Classification - One](https://www.mygreatlearning.com/blog/what-is-artificial-intelligence/)
- [What is Unsupervised Learning ?](https://www.ibm.com/cloud/learn/unsupervised-learning)
- [Reinforcement Learning Terms](https://www.youtube.com/watch?v=JgvyzIkgxF0)
- [What is Semi supervised Learning ?](https://www.youtube.com/watch?v=tCaPH_bBoWM)

## What is Self supervised Learning ?
- [Automatic annotations](https://medium.com/analytics-vidhya/what-is-self-supervised-learning-in-computer-vision-a-simple-introduction-def3302d883d)
- Generate labeled dataset using large amount of dataset.
- Train model using labeled dataset.  

## Supervised Learning Types
- Classification model - Assign discrete label or labels from predefined set of categories to dataset inputs 
- Regression model - Assign continuous numerical values to dataset inputs

## Supervised Learning Examples
- House price prediction - Regression model
  - Input variables - Number of bedrooms, area of house, locality of house and so on
  - Output variable - Price of house  
  - Optimization function - Reduce error between predicted and actual price of house
- Face recognition - Classification model
  - Input variables - Input cropped images
  - Output variable - Identity of person
  - Optimization function - Reduce classification error

## Unsupervised Learning Types
- Clustering
- Feature Learning
- Association Rule Learning

## Unsupervised Learning Examples
- Association Rule Learning - Apriori Algorithm
  - If a person buys Bread then he also buys Jam. - Bread and Jam are bought together
  - If a person buys Laptop then he also buys Laptop bag. - Laptop and Laptop bag are bought together
- Customer segmentation - Clustering algorithm
  - Input variables - Number of size measurements for customers  
  - Number of shirt sizes small medium, large computed using clustering  
  - Optimization function - NO accuracy
    - Decrease distance between samples from same cluster 
    - Increase distance between samples from different clusters  
- Anomaly detection - Feature learning  
  - Use autoencoder for feature learning 
  - Detect anomaly using reconstruction error and threshold

## Reinforcement Learning Examples
- AlphaGo
- Super Mario game
- Autonomous drones - Stanford University

## Semi Supervised Learning Examples
- Vehicle number plate detection
  - Open dataset - Train model
  - Use trained model to label dataset for Indian context
  - Finetune trained model using Indian dataset
- Vehicle number plate recognition
  - Generate synthetic dataset - Indian context
  - Train model using synthetic dataset
  - Use trained model to label dataset for Indian context
  - Finetune trained model using Indian dataset

## Self Supervised Learning Examples
- [Gray images to color images](https://www.youtube.com/watch?v=tCaPH_bBoWM)
  - Use color images to generate corresponding gray images - Labeled dataset
  - Train model using labeled dataset
- [Predict next word - Language model](https://www.youtube.com/watch?v=tCaPH_bBoWM)
  - Collect text dataset using internet web pages
  - Generate labeled dataset for predicting next word using text dataset.
  - Train model using labeled dataset  

# Supervised Learning Classification
- Support vector machine
- Decision tree
- [Artificial neural network](https://playground.tensorflow.org/)

# Artificial Neural Network Classification
- Linear networks - One input and one output layer- Linear function
- Deep neural networks - More than one hidden layer
- Feed forward neural network
- Convolutional neural network
- Recurrent neural network 

# Unsupervised Learning Classification
- K-means Clustering
- Hierarchal Clustering
- Principle Component Analysis
- Autoencoder

## What is Deep Learning ?
- Input layer and Output layer
- More than one hidden layer

## Why Deep Learning ?
- Large amount of data - Internet of Things - IoT
- Faster hardware - GPU
- Better algorithms - Residual networks, ReLU activation  

## Machine Learning v/s Deep Learning
- Training time
  - Machine learning - Less time 
  - Deep learning - Large time - Large number of parameters
- Feature extraction
  - Machine learning - Domain expert - Feature extraction - Reduce data complexity
  - Deep learning - Learn high level features - No need of domain expert
- Interpretability
  - Machine learning - Inherent interpretable  
  - Deep learning - Black box - Explainable AI
- Accuracy
  - Machine learning - Less parameters - Less complexity - Less accuracy for large datasets
  - Deep learning - More layers - More parameters - More complexity - More accuracy for large datasets

## Artificial Neural Network v/s Deep Neural Network
- Artificial Neural Network - One hidden layer
- Deep Neural Network - More than one hidden layer
