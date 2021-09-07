# MIT-6.036

The W7 Neural Networks II file uses the supporting codes from the MIT 6.036 to build up the Neural Network:

The modules include: 

1. Linear Modules: 
    -  Forward: to create W@Input
    -  Backward: to calculate dLdW, dLdW0, and dLdA.
    -  sgd_step: update weights through  dLdW, dLdW0 and learning rate

2. Activation Modules: Each class has forward and backward (input: dLdA / output: dLdZ)
    -  Tanh
    -  ReLU
    -  SoftMax (it also returns class indices to see for each column vector, which index has the highest probability)

3. Loss Modules:
    -  Forward: Determine the loss between Ypred and Y
    -  Backward: Provide the backward calculation (i.e. dLdA)
