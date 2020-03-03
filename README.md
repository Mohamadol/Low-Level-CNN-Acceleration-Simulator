# A Low-Level CNN Acceleration Simulator in C
## Intro
This is a low-level c code which supports various layers involved in convolutional neural networks, Since the operations are written in low-level C, framework can be used for simulation purposes. Specifically, one can include counters at different layers to simulate effect of various acceleration techniques on and hardware availability.
## Layers
 Source code for layers are included under **layers** directory and are described below:
1. Convolution Layer
2. Fully Connected Layer
3. ReLu Layer
4. Max Pooling Layer
## Included Network
Weights for a pre-trained network are included under **data** directory along with a few sample input data. The network is trained for classification of CIFAR-10 data set. This network serves as an example which shows how the framework can be used.
## Test Functions
**Tests** directory contains simple test functions which were used to confirm functionality of each layer. The result was confirmed with the result of python scripts that performed the same computations on the same data.
