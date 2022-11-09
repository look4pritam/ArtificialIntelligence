# CNN Architectures
- VGGNet
  - 3x3 Convolutions
- ResNet
  - Shortcuts
- MobileNet-v1
  - Depthwise Separable Convolution
    - Network speed and power consumption - Proportional - Number of fused Multiplication and Addition operations
    - Depthwise convolution + pointwise convolution
  - Width multiplier 
    - Adjust the number of channels 
    - {1, 0.75, 0.5, 0.25}
  - Resolution multiplier 
    - Adjust the spatial dimensions of the input image and the feature maps
    - {224, 192, 160, 128}
- MobileNet-v2
  - Inverted residual block
   
# CNN examples

- [Class Activation Map examples](./ClassActivationMap/)
- [Saliency Map examples](./SaliencyMap/)
- [TripletLoss examples](./TripletLoss/)
