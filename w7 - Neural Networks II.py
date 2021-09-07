# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:23:25 2021

@author: Yung-Yu Chen (Michelle)
"""

import numpy as np
import modules_disp as disp
from expected_results import *
from code_for_hw7 import *
from math import log


class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights


# Linear modules
#
# Each linear module has a forward method that takes in a batch of
# activations A (from the previous layer) and returns
# a batch of pre-activations Z.
#
# Each linear module has a backward method that takes in dLdZ and
# returns dLdA. This module also computes and stores dLdW and dLdW0,
# the gradients with respect to the weights.
class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        return np.dot(self.W.T,self.A)+self.W0   # Your code (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW  = self.A@dLdZ.T  # Your code -- A (2x4); dLdZ.T (4 x3); 4 is sample size ; if X is (2x4) 2 is m means input dims, 4 is the number of samples
        self.dLdW0 = np.reshape(np.sum(dLdZ.T, axis = 0), (self.n, 1))  # Your code
        return np.dot(self.W,dLdZ)        # Your code: return dLdA (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
        self.W  = self.W - lrate*self.dLdW  # Your code
        self.W0 = self.W0 - lrate*self.dLdW0  # Your code
        
# Activation modules
#
# Each activation module has a forward method that takes in a batch of
# pre-activations Z and returns a batch of activations A.
#
# Each activation module has a backward method that takes in dLdA and
# returns dLdZ, with the exception of SoftMax, where we assume dLdZ is
# passed in.
class Tanh(Module):  # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):  # Uses stored self.A
        return dLdA*(1-self.A**2)  # Your code: return dLdZ (?, b)


class ReLU(Module):  # Layer activation
    def forward(self, Z):
        self.A = np.where(Z>0, Z, 0)  # Your code: (?, b)
        return self.A

    def backward(self, dLdA):  # uses stored self.A
        return dLdA*np.where(self.A>0, 1, 0)  # Your code: return dLdZ (?, b)


class SoftMax(Module):  # Output activation
    def forward(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z), axis=0)  # Your code: (?, b)
        #axis = 0 as Z will contain b sample sizes

    def backward(self, dLdZ):  # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred, axis=0)  # Your code: (1, b)
    
# Loss modules
#
# Each loss module has a forward method that takes in a batch of
# predictions Ypred (from the previous layer) and labels Y and returns
# a scalar loss value.
#
# The NLL module has a backward method that returns dLdZ, the gradient
# with respect to the preactivation to SoftMax (note: not the
# activation!), since we are always pairing SoftMax activation with
# NLL loss
class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return -np.sum(Y*np.log(Ypred))  # Your code: return loss (scalar)

    def backward(self):  # Use stored self.Ypred, self.Y
        return -(self.Y/self.Ypred - (1-self.Y)/(1-self.Ypred))*(self.Ypred*(1-self.Ypred))  # Your code (?, b)

    #self.Ypred - self.Y 
    
# Neural Network implementation
np.random.seed(0)
class Sequential:
    def __init__(self, modules, loss):  # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        sum_loss = 0
        for it in range(iters):
            rnd = np.random.randint(N) #New Code:pick random index for X & Y
            Xt, Yt =  X[:,rnd:rnd+1], Y[:,rnd:rnd+1]
            Ypred = self.forward(Xt) #New Code:generate Ypred
            #Loss = self.loss.forward(Ypred, Y[:, N:N+1]) #New Code:generate Loss
            sum_loss += self.loss.forward(Ypred, Yt)
            dloss = self.loss.backward() #New Code:generate dloss
            self.backward(dloss)
            self.sgd_step(lrate)
            

    def forward(self, Xt):  # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):  # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]:
            # Note that delta can refer to dLdA or dLdZ over the
            # course of the for loop, depending on the module m
            delta = m.backward(delta)

    def sgd_step(self, lrate):  # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '\tAcc =', acc, '\tLoss =', cur_loss, flush=True)



np.random.seed(0)

# data
X, Y = super_simple_separable()
# module
linear_1 = Linear(2, 3)
#hyperparameters
lrate = 0.005

# test case
# forward
z_1 = linear_1.forward(X)
exp_z_1 =  np.array([[10.41750064, 6.91122168, 20.73366505, 22.8912344],
                     [7.16872235, 3.48998746, 10.46996239, 9.9982611],
                     [-2.07105455, 0.69413716, 2.08241149, 4.84966811]])
unit_test("linear_forward", exp_z_1, z_1)

# backward
dL_dz1 = np.array([[1.69467553e-09, -1.33530535e-06, 0.00000000e+00, -0.00000000e+00],
                                     [-5.24547376e-07, 5.82459519e-04, -3.84805202e-10, 1.47943038e-09],
                                     [-3.47063705e-02, 2.55611604e-01, -1.83538094e-02, 1.11838432e-04]])
exp_dLdX = np.array([[-2.40194628e-02, 1.77064845e-01, -1.27021626e-02, 7.74006953e-05],
                                    [2.39827939e-02, -1.75870737e-01, 1.26832126e-02, -7.72828555e-05]])
dLdX = linear_1.backward(dL_dz1)
unit_test("linear_backward", exp_dLdX, dLdX)

# sgd step
linear_1.sgd_step(lrate)
exp_linear_1_W = np.array([[1.2473734,  0.28294514,  0.68940437],
                           [1.58455079, 1.32055711, -0.69218045]]),
unit_test("linear_sgd_step_W",  exp_linear_1_W,  linear_1.W)

exp_linear_1_W0 = np.array([[6.66805339e-09],
                            [-2.90968033e-06],
                            [-1.01331631e-03]]),
unit_test("linear_sgd_step_W0", exp_linear_1_W0, linear_1.W0)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)

def for_softmax(y):
    return np.vstack([1 - y, y])

######################################################################
# Tests
######################################################################

def unit_test(name, expected, actual):
    if actual is None:
        print(name + ": unimplemented")
    elif np.allclose(expected, actual):
        print(name + ": OK")
    else:
        print(name + ": FAILED")
        print("expected: " + str(expected))
        print("but was: " + str(actual))


def sgd_test(nn, test_values):
    """Run one step of SGD on a simple dataset with the specified
    network, and with batch size (b) = len(dataset)

    :param nn: A "Sequential" object representing a neural network

    :param test_values: A dictionary containing the expected values
    for the necessary unit tests

    """
    lrate = 0.005
    # data
    X, Y = super_simple_separable()

    # define the modules
    assert len(nn.modules) == 4
    (linear_1, f_1, linear_2, f_2) = nn.modules
    Loss = nn.loss

    unit_test('linear_1.W', test_values['linear_1.W'], linear_1.W)
    unit_test('linear_1.W0', test_values['linear_1.W0'], linear_1.W0)
    unit_test('linear_2.W', test_values['linear_2.W'], linear_2.W)
    unit_test('linear_2.W0', test_values['linear_2.W0'], linear_2.W0)

    z_1 = linear_1.forward(X)
    unit_test('z_1', test_values['z_1'], z_1)
    a_1 = f_1.forward(z_1)
    unit_test('a_1', test_values['a_1'], a_1)
    z_2 = linear_2.forward(a_1)
    unit_test('z_2', test_values['z_2'], z_2)
    a_2 = f_2.forward(z_2)
    unit_test('a_2', test_values['a_2'], a_2)

    Ypred = a_2
    loss = Loss.forward(Ypred, Y)
    unit_test('loss', test_values['loss'], loss)
    dloss = Loss.backward()
    unit_test('dloss', test_values['dloss'], dloss)

    dL_dz2 = f_2.backward(dloss) #n x b
    unit_test('dL_dz2', test_values['dL_dz2'], dL_dz2)
    dL_da1 = linear_2.backward(dL_dz2) #dim change to m x b ; dim m is the input size of weights 
    unit_test('dL_da1', test_values['dL_da1'], dL_da1)
    dL_dz1 = f_1.backward(dL_da1)
    unit_test('dL_dz1', test_values['dL_dz1'], dL_dz1)
    dL_dX = linear_1.backward(dL_dz1)
    unit_test('dL_dX', test_values['dL_dX'], dL_dX)

    linear_1.sgd_step(lrate)
    unit_test('updated_linear_1.W', test_values['updated_linear_1.W'], linear_1.W)
    unit_test('updated_linear_1.W0', test_values['updated_linear_1.W0'], linear_1.W0)
    linear_2.sgd_step(lrate)
    unit_test('updated_linear_2.W', test_values['updated_linear_2.W'], linear_2.W)
    unit_test('updated_linear_2.W0', test_values['updated_linear_2.W0'], linear_2.W0)

######################################################################

def nn_tanh_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), Tanh(), Linear(3, 2), SoftMax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=1, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]

def nn_relu_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), ReLU(), Linear(3, 2), SoftMax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=2, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]


def nn_pred_test():
    np.random.seed(0)
    nn = Sequential([Linear(2, 3), ReLU(), Linear(3, 2), SoftMax()], NLL())
    X, Y = super_simple_separable()
    nn.sgd(X, Y, iters=1, lrate=0.005)
    Ypred = nn.forward(X)
    return nn.modules[-1].class_fun(Ypred).tolist(), [nn.loss.forward(Ypred, Y)]

