# [Machine Learning Project Lifecycle](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/PhRTU/steps-of-an-ml-project)
- Project scoping
- Dataset collection
- Modelling
- Deployment
 
## [Project Scoping](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/Qe96S/scoping-process)
- Identify project
  - Problems - What ? - Domain experts - Identify problems 
  - Solutions - How ? - AI experts - Indentify solutions to problems
  - [Access project feasibility](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/vHkzC/diligence-on-feasibility-and-value) - Use benchmarks 
  - [Access project value](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/DQGCX/diligence-on-value)
  - Select valuable and feasible problems 
- Determine milestones
- Define metrics 
- Estimate resource requirement - Data, time, and people

## Dataset Collection  
- Machine learning dataset types
- Establish baseline
  - Estimate initial metrics for base line model
- Label dataset
  - Use either open dataset or collect in-house dataset
  - Clean dataset
- Validate dataset
- Organize dataset
  - Divide dataset into training, validation, and test datasets
- Feature engineering
  - Select features for model training
  - Preprocess dataset

## Modelling
- [Iterative process](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/OFPbm/key-challenges) - Model + hyperparameters + Data - Training - Error analysis
- [Establish baseline model performance](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/lecture/KYu4T/establish-a-baseline)
- Model-centric AI development 
- Data-centric AI development

## Deployment
- Post training optimization
- Deploy model
- Monitor model performance
- Maintain model performance

### Access Project Feasibility
- Human Level Performance (HLP) - Unstructured dataset
- Feature engineering - Structured dataset
- Previous project history - Estimate project progress

### Define Metrics (Search Engine)
- Machine learning metrics - Word level accuracy
- Software metrics
- Business metrics - Revenue
- Word level accuracy - Query level accuracy - Search result quality - Revenue
- Define initial metrics
- Improve performance towards business metrics
- Estimate of business metrics from machine learning metrics
   
### Machine Learning Dataset types
- Structured dataset - Spreadsheet dataset
- Unstructured dataset - Images, video, audio, and texts
   
### Validate Dataset
- Compute statistics on dataset
- Understand dataset schema
     - Define data type for each feature 
     - Identify samples where certain values are incorrect or missing
- Identify inconsistencies in training, validation, and testing dataset
- Identify dataset drift 
- Identify concept drift 
- Identify training-serving dataset skew

### Data Curation
- Manage data throughout lifecycle
- Model quality and performance
- Collect (select and curate) - Label - Train - Deploy 

### Organize Dataset
- Multiple parts with similar statistical properties
- Training dataset
- Validation dataset - Evaluate performance of model during training
  - Stop model training 
  - Select hyperparameters 
- Testing dataset - Evaluate performance of model after training
  - Performance of model on unseen dataset

### Feature Engineering
- Dataset preprocessing
- Dataset transformation
- Input - Single column in dataset before preprocessing
- Feature - Single column in dataset after preprocessing
- Scale numerical values to (0, 1) or (-1, 1) range
- Converting non numerical data values into numerical format

### Establish Baseline Model Performance
- Unstructured dataset 
  - Human Level Performance 
  - Previous research works
  - Previous system
- Structured dataset  
  - Previous research works
  - Previous system 
- Bayes error - Irreducible error
 
### Model-centric AI Development 
- Fix dataset - Improve model - Improve performance
- Train model
  - Overfit on training dataset - Low training error - Low bias 
- Evaluate trained model  
  - Evaluate on validation and test dataset - Low validation or testing error - Low variance
- Validate trained model
  - Evaluate model on multiple test datasets
  - Evaluate model on business metrics
- Perform error analysis
- Test validated model using test dataset
  
### Data-centric AI Development
- Fix model - Improve dataset - Improve performance
- Multiple test datasets - Compare performance with base-line model - Add more training data - Improve performance
- From big dataset to good dataset
- Data augmentation - Unstructured dataset
- Feature engineering - Structured dataset

### Deploy Model
- Prediction and inference
- Cloud AI
- Edge AI
- On premises
- Online prediction - Less latency
- Batch prediction - Offline prediction - Large dataset - Offline dataset

### Cloud AI
- Infrastructure support
- Scalability 
- Different services

### Edge AI
- Run AI algorithm locally at source 
- Hardware devide - Edge devices 
- Advantages - Real time decision making - No need for constant network connectivity - Reduce latency
- Example applications - Autonomous vehicles (low latency) - 
- Example devices - NVIDIA Jetson Nano, NVIDIA Jetson Xavier, Raspberry Pi 4, Intel Movidius

###  [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano)
- GPU	- NVIDIA Maxwell architecture - 128 NVIDIA CUDA cores
- CPU	- Quad core ARM Cortex
- Memory - 2 GB or 4 GB

### [NVIDIA Jetson Xavier](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit)
- GPU	- NVIDIA Volta architecture - 512 NVIDIA CUDA cores with Tensor Cores
- CPU	- 8 core ARM 
- Memory - 32 GB

### [NVIDIA Jetson Orin](https://developer.nvidia.com/embedded/jetson-agx-orin)
- GPU	- NVIDIA Ampere architecture - 2048 NVIDIA CUDA cores - 64 Tensor Core
- CPU	- 12 core Arm Cortex
- Memory - 32 GB

### Monitor Model Performance
- Concept drift
- Data drift
  
## Post Training Optimization  
- Hardware Level Optimization
- Software Level Optimization
- Algorithm Level Optimization

### Hardware Level Optimization
- Edge AI
- CPU, GPU, TPU

### Software Level Optimization
- Target Optimized Libraries - cuDNN, 
- Deep Learning Compilers - OpenCV-DNN, TensorRT

### [Algorithm Level Optimization](https://medium.com/aiguys/reducing-deep-learning-size-16bed87cccff)
- Pruning - Remove model parameters 
  - Remove 80% parameters 
  - 1.5% reduction in accuracy
  - Retraining to improve performance 
- Quantization - Convert weights to int16, int8 or float16 
