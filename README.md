# MIT-6.036

Author: YungYu Chen (Michelle)

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

4. Neural Network Implementation:
    -  Class: Sequential
    -  Forward: Compute Ypred
    -  Backward: reversed list of modules (with recursion, i.e. delta = func(delta))
                 # Update dLdW and dLdW0
    -  sgd_step: Gradient descent step. Input - learning rate
    -  sgd***: Train the model.
               - Pick random set of X & Y
               - Generate Ypred through self.forward
               - dLoss: dL/dA through loss modules.backward()
               - Backprop: self.backward(dLoss) - only Linear Modules have sgd_step function to update weights, otherwise update will be passed.
               - self.sgd_step(lrate): update weights
