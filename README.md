# mini_deep_learning_framework

In this package, I have implemeented a basic framework which performs neural network functions just like PyTorch but with much fewer features. Note that, we only import torch library (for arithmetic operations) and not torch.nn (for neural networks).

Essentially for training your neural network, following modules are available:
1. Linear layer
2. Activation functions : ReLU, TanH
3. Sequential module
4. Optimizers: Adam and batch SGD

DL_framework.py: Implementation of the above-mentioned modules 
test.py: sample program to import modules from DL_framework and use them for training a neural network.

# To run
python test.py
