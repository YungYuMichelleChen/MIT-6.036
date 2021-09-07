# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file
"""

import numpy as np
import math

#-------------------W6 exercise------------------------------------------
X = np.array([[0, 1, 2],
              [0, 1, 2]])
Y = np.array([[0, 1, 0]]) 

weight = np.array([[-0.5, 1, 0],
              [1.5, -1, 0]])

X_w0 = np.vstack((np.full((1, 3), 1), X))

#-------------------W6 1A------------------------------------------
result_1A =weight@X_w0
print(list(np.where((result_1A>0), 1, 0)))

weight_2 = np.array([[1, 1, 1]])
np.linalg.norm(weight_2)

X = np.array([[2, 3, 9, 12],
              [5, 2, 6, 5]])
Y = np.array([[1, 0, 1, 0]])

#-------------------W6 Q2------------------------------------------
t = np.array([-1, 0, 1])
exp_array = np.vectorize(math.exp)
list(map(exp_array, t))
total = sum(list(map(exp_array, t)))
np.multiply(np.array(list(map(exp_array, t))), 1/total)
def SM(z):
  # implement softmax
  return np.exp(z)/np.sum(np.exp(z))

#-------------------W6 Q2C------------------------------------------
w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1, 1]]).T
y = np.array([[0, 1, 0]]).T
guess = SM(w.T@x)
NLL_grad = x@(guess-y).T
w_r1 = w-np.multiply(NLL_grad, 0.5)

