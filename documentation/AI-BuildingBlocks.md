# AI Building Blocks

## Reinforcement Learning
- Markov decision processes
- Rewards
- Discount factor
- Bellman equation
- Solving Bellman equation
- Deterministic vs stochastic processes
- Value neural networks
- Policy neural networks
- Train policy neural network

## Convolutional Neural Networks
- Why ?
  - Input - Image - Large number of parameters (for Artificial Neural Network) 
  - Parameter sharing
  - Sparsity of connections
- Image classification - Face recogntion, 
- Object detection - Face detection, vehicle detection, number plate detection
- Image-based sequence recognition - Vehicle number plate recogntion
- Neural style transfer - Synthetic face generation

## Neural Style Transfer
- Content image
- Style image
- Generate content image using style image   
- Examples 
  - Generate art work
  - Generate synthetic faces

## Understanding Convolutional Neural Networks
- Initial layers - Edges 
- Intermediate layers - Combine edges - Vertical or horizontal lines - Circular shapes - Partial nose, eyes, ears 
- Deep layers - Complex shapes - Nose, eyes, ears
- Final layers - Complete face using nose, eyes, and ears  

## [Apriori Algorithm - Association Rule Learning](https://www.youtube.com/watch?v=guVvtZ7ZClw)
- Market basket analysis 
- Association between items
- If (...) then (...) relationship
- Examples
  - If a person buys Bread then he also buys Jam. - Bread and Jam are bought together
  - If a person buys Laptop then he also buys Laptop bag. - Laptop and Laptop bag are bought together
- Measure association
  - Support 
  - Confidence 
  - Lift
  - Conviction

## Apriori Algorithm - Support
- Frequency of item or Frequency of items
- Threshold for Support
- Filter less frequent items using Support

## Apriori Algorithm - Confidence
- Ratio of frequency of items together by frequency of base item 
- Support (X->Y) / Support(X) - Frequency or popularity of X  
- Threshold for Confidence
- Filter rules using Confidence

## [Apriori Algorithm - Lift](https://www.youtube.com/watch?v=WGlMlS_Yydk)
- Association v/s Randomness
- Support (X->Y) / ( Support(X) * Support(Y) ) - Frequency or popularity of both X and Y - Association
- Association - Increase Lift
- Randomness - Decrease Lift

## [Apriori Algorithm - Conviction](https://www.youtube.com/watch?v=WGlMlS_Yydk)
- (1 - Support(Y))  / (1 - Confidence(X -> Y))  

## [Principal Component Analysis](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- Reduce dimensions of data - High dimensions to low dimensions (2D) - Better visualization
- Compute importance of various features - Remove less important features
- Feature engineering - Reduce data dimensionality without losing much information

## Principal Component Analysis - Algorithm
- Normalize dataset
- Compute average or mean dataset values
- Center dataset using mean values
- Compute principal component along one direction
  - Start with random line passing through origin 
  - Project features on line
  - Compute feature distance from origin for projected features
  - Compute sum of square of feature distances
  - Fit line to maximize sum of squared distances OR Fit line to minimize sum of squared projected distances
  - Fit line - Principal Component along one direction 
    - Linear Combinations of features
    - Singular vectors or Eigen vectors
  - Normalize distance
  - Square root of sum of square distances - Eigen value - Singular value 
- Principal components - Perpendicular to each other
- Rank axes with in order of importance

## Principal Component Analysis - Scree Plots
- Importance of Principal Components
- Remove less important Principal Components
- Reduce data dimensionality without losing much information 

## [DBSCAN Clustering](https://www.youtube.com/watch?v=RDZUdRSDOok)
- Good for Nested Clusters
- Radius - similarity or distance measure 
- Core points - High neighbour count within Radius 
- Cluster Core points using similarity between other Core points - Core point extends Cluster
- Add non-core points using similarity between Core points - Non-core point added to Cluster  

## [Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
- Heatmaps - [Dendrogram](https://www.youtube.com/watch?v=ijUMKMC4f9I) - Features
- Merge samples using similarity (or distance) - Construct Cluster
- Merge sample to cluster using similarity (between cluster centroid or farest or closest sample) 

## [t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM)
- Input - High dimensional dataset 
- Output - Low dimensional graph - 2D or 3D graph - Easy visualization
- Compute similarity (or distance) between selected sample and other samples - Unscaled similarity
- Normalize similarity values - Sum == 1
- t distribution - Less tall than Normal distribution  

## [Graph Neural Networks](https://www.youtube.com/watch?v=fOctJB4kVlM)
- Graph data
  - Nodes + Edges + Adjacency matrix (connection between nodes)
- Node features and Edge features
  - Molecule - Node features == Atom - Edge features == Bond type
- Applications
  - 3D games, drug discovery, social networks,
- Node level prediction
- Edge level prediction
- Graph level prediction
- Representation learning
- Graph knowledge

## Transformer
- Attention mechanism - Infinite reference window
- RNN - Short reference window
- LSTM or GRU - Long reference window
- Self-attention 
  - Infinite reference window
  - Query, Key, and Value
- Input embedding
- Positional encoding
- Transformer encoder 
  - Feature extraction
  - Multi headed attention
- Transformer decoder 
  - Generate output sequence
  - Multi headed attention

## Self Attention
- Query
- Key
- Value

## BERT 
- Bidirectional Encoder Representations from Transformers

## [Decision Tree](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- Binary tree
- Decision nodes
- Root node 
- Leaf nodes
- Impure nodes
- Select decision with - More Information gain - Less impurity
- Gini impurity  
  - 1 - (probability of yes) ** 2 - (probability of no) ** 2 
  - Weighted average of decision nodes
- Less accurate - Overfitting - Solution - Random Forests

## [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- Bootstrap dataset
- Out of bag dataset  
- Number of decision trees
- Random subset of variables - Decorrelated decision trees
- Create number of random forests - Select best one

## Extremely Randomized Trees Classifier
- Use whole original dataset
- Similar to random forests

## [K Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- Unsupervised learning algorithm
- Divide data into k clusters
- Minimum variation or non change in cluster centers 
- Estimate k - Calculate variation for different k value - Elbow plot - Sharp reduction in variation
- Sensitive to outliers and initial cluster centers
- Normalize features 
- Silhouette score
  - Within cluster - Closer points
  - Distance between clusters
  - -1 to 1 - 1 best - -1 worst

## K Nearest Neighbors Algorithm
- Supervised learning 
- Voting - k nearest neighbors

## Autoencoder
- Encode input data - Feature engineering - Features
- Decode features 
- Reconstruct output data identical to input data
- Minimize reconstruction error
- Applications
  - [Anomaly detection](https://keras.io/examples/timeseries/timeseries_anomaly_detection/) 
  - Data denoising
  - Feature extraction 
  - Image colorization
  - Information retrieval 
  - Dimensionality reduction 

## Isolation Forest
- Unsupervised anomaly detection
- Based on decision trees - Isolation trees
- Anomaly score - Depth of tree for data point
- Deeper depth data point - Normal
- Shorter depth data point - Anomaly

## UNet
- Encoding path - Feature engineering 
- Decoding path
- Applications
  - Image segmentation - Medical, satellite
  - Semantic segmentation

## Surrogate Models
- Accelerate simulation based analysis
- Data driven approach
- Surrogate model 
  - Metamodel or emulator
  - Approximate simulation output - Save number of simulaions

## [Support Vector Classifier](https://www.youtube.com/watch?v=efR1C6CvhmE)
- Margin - Shortest distance between observations and threshold 
- Soft margin - Allow misclassification - High bias - Low variance  
- Soft Margin Classifier - Support Vector Classifier 
- Soft margins - Support vectors
- 1 dimensional data - 0 dimensional Support Vector Classifier - Single point
- 2 dimensional data - 1 dimensional line Support Vector Classifier
- 3 dimensional data - 2 dimensional plane Support Vector Classifier
- More dimensional data - Hyperplane Support Vector Classifier

## [Support Vector Machine](https://www.youtube.com/watch?v=efR1C6CvhmE)
- SVM
- Kernel function + Support Vector Classifier
- Kernel function 
  - Move low dimensional data to high dimension
  - Compute relationship in high dimension
- [Polynomial kernel](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [Radial Basis Function kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns)

## Naive Bayes
-
 
## Term Frequency Inverse Document Frequency 
-

## Passive Aggressive Algorithm
- Online learning algorithm
- Passive - If prediction is correct, keep model and do not make any changes
- Aggressive - If prediction is incorrect, make changes to model
- [Online learning algorithm will get a training example, update the classifier, and then throw away the example.](https://www.youtube.com/watch?v=TJU8NfDdqNQ)
- Fake news detection

## Recurrent Neural Network
-

## Bidirectional Recurrent Neural Network
-

## GAN
- Generative Adverserial Network
- Generator - Discriminator 
- Unsupervised learning

## DCGAN
- Deep Convolutional GAN

## Cycle GAN
- Image-to-image translation
- Unpaired image dataset
- Examples
  - Zebra to Horse images
  - Synthetic to real vehicle number plate images
  
## pix2pix GAN
- Image-to-image translation
- Paired image dataset

## SSIM
- Structural Similarity
-

## Sub Pixel Convolution
-

## VAE
- Variational Autoencoders

## [Learning Rate](https://www.jeremyjordan.me/nn-learning-rate/)
- Start 
  - Small learning rate 
  - Better accuracy 
  - Less oscillations
  - Less number of epochs - Avoid local minima    
- Middle 
  - Large learning rate 
  - Fast convergence
  - Avoid local minima 
  - Large number of epochs 
  - Decrease learning rate at saturation   
- End 
  - Small learning rate
  - Better accuracy 
  - Less oscillations


## [Triplet Loss](https://omoindrot.github.io/triplet-loss)
- What ?
  - Learn image features
- Why ?
  - Variable number of classes - Softmax fix number of classes
  - Image similarity 
  - Image verification
- How ?
  - Siamese network - Shared weights
  - Anchor image - Reference image
  - Positive image - Same class as that of anchor image
  - Negative image - Different class as that of anchor image
- Loss
  - Reduce distance between anchor image and positive image
  - Increase distance between anchor image and negative image
  - Margin - Expected distance between Positive and Negative   
  - L(Anchor, Positive, Negative) = max(0, Distance(Anchor, Positive) — Distance(Anchor, Negative) + Margin)
- Challenges
  - Easy triplets - Zero loss - Large number - No effective learning 
  - Hard triplets - Negative closer to anchor than positive
  - Semi-hard triplets - Positive loss - Negative not closer to anchor than positive 
  - Offline triplet mining - Precompute triplets offline
  - Online triplet mining - Compute triplets online
 
## DropBlock
-

## Label Smoothing
-

## Universal Sentence Encoder
- 

## YOLOX
- Decoupled head
- Anchor box free detection
- Mosaic data augmentation
- Mix up data augmentation
- Multi positives
    - Select 3x3 positives instead of just 1 positive
    - Center sampling
- SimOTA - Advanced label assignment strategy
- End-to-end training
