# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 12:24:49 2021

@author: YungYu Chen (Michelle)
"""

import math as m 
import numpy as np


class Module:
    def step(self, lrate): pass  # For modules w/o weights

# Linear modules

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

    def step(self, lrate):  # Gradient descent step
        self.W  = self.W - lrate*self.dLdW  # Your code
        self.W0 = self.W0 - lrate*self.dLdW0  # Your code

# Activation modules
        
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

class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return -np.sum(Y*np.log(Ypred))  # Your code: return loss (scalar)

    def backward(self):  # Use stored self.Ypred, self.Y
        return -(self.Y/self.Ypred - (1-self.Y)/(1-self.Ypred))*(self.Ypred*(1-self.Ypred))  # Your code (?, b)
        


class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:, indices]  # Your code: reorder the columns based on shuffled indices
            Y = Y[:, indices]  # Your code

            for j in range(m.floor(N/K)):
                if num_updates >= iters: break

                # Implement the main part of mini_gd here
                # Your code
                Xt = X[:, K*j:K*(j+1)]
                Yt = Y[:, K*j:K*(j+1)]
                Ypred = self.forward(Xt)
                loss = self.loss.forward(Ypred, Yt) # Although loss is not used, the input params are needed to specify Y & Ypred
                dloss = self.loss.backward()
                self.backward(dloss)
                self.step(lrate)
                num_updates += 1


    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def step(self, lrate):    
        for m in self.modules: m.step(lrate)

######################################################################
# OPTIONAL: Problem 2B) - BatchNorm
######################################################################

class Module:
    def step(self, lrate): pass  # For modules w/o weights

class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(self.A, axis=1, keepdims = True)  # Your Code - It's necessary to keep dims to remain 3D or more array dimension
        self.vars = np.var(self.A, axis=1, keepdims = True)   # Your Code
        
        # Keep dims: if nd = (2, 2, 3) without keepdims the stats become (2, 3) - the middle dimension disappears
        # if keepdims = True, mean of nd(2, 2, 3) becmoes (2, 1, 3) - the middle dimension is kept
        

        # Normalize inputs using their mean and standard deviation
        self.norm = (self.A - self.mus)/np.sqrt(self.vars+self.eps)  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.norm*self.G+self.B  # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B = self.B - lrate*self.dLdB  # Your Code
        self.G = self.G - lrate*self.dLdG  # Your Code

######################################################################
# Tests
######################################################################
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)
  
def for_softmax(y):
    return np.vstack([1-y, y])

# For problem 1.1: builds a simple model and trains it for 3 iters on a simple dataset
# Verifies the final weights of the model
def mini_gd_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 3, lrate=0.005, K=1)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]

# For problem 1.2: builds a simple model with a BatchNorm layer
# Trains it for 1 iter on a simple dataset and verifies, for the BatchNorm module (in order): 
# The final shifts and scaling factors (self.B and self.G)
# The final running means and variances (self.mus_r and self.vars_r)
# The final 'self.norm' value
def batch_norm_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), BatchNorm(2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 1, lrate=0.005, K=2)
    return [np.vstack([nn.modules[3].B, nn.modules[3].G]).tolist(), \
    np.vstack([nn.modules[3].mus, nn.modules[3].vars]).tolist(), nn.modules[3].norm.tolist()]
