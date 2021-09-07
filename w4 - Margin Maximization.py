# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:54:50 2021

@author: yungchen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

x1 = [3, 2]
x2 = [1, 1]
x3 = [4, 2]

theta = np.array([[1, 1]])
th0 = -4

def margin(x, th , th0):
    x = np.array([x]).T
    th0 = np.full((1, 1), th0)
    return (th@x+th0)/np.linalg.norm(th) #this does not take into account of th0 for norm(theta)
#(th@x+th0)/np.linalg.norm(np.hstack((th, th0)))
data = [x1, x2, x3]
labels = [1, -1, -1]
#for i in range(3):
    #print(margin(data[i], theta, th0)*labels[i])
    
#-----------------------------Question3--------------------------------------
data3 = np.array([[1, 1, 3, 3],[3, 1, 4, 2]])
labels3 = np.array([[-1, -1, 1, 1]])
th3 = np.array([[0, 1]]).T
th03 = -3

(d3, n3 ) = data3.shape

def margin3(x, label, th,th0):
    return label*(th.T@x+th0)/np.linalg.norm(th)

list3 = []
print(margin3(data3, labels3, th3, th03))
    
def perceptron(data, label, th, th0):
    (d, n) = data.shape
    min_margin = np.amin(margin3(data, label, th, th0))
    th_max = th
    th0_max = th0
    for i in range(10000):
        for j in range(n):
            if label[0, j]*(th.T@data[:, j]+th0) <=0 :
                th = th + np.array([data3[:,j]*labels3[0, j]]).T
                th0 = th0 + label[0, j]
                if np.amin(margin3(data, label, th, th0))>= min_margin:
                    min_margin = np.amin(margin3(data, label, th, th0))
                    th_max = th
                    th0_max = th0
    return(th, th0, min_margin, th_max, th0_max)
    
#----------------------------Homework------------------------------------------
'----------------------------W4 Question1------------------------------------------'
data4 = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels4 = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5
red_margin = margin3(data4, labels4, red_th, red_th0)
blue_margin= margin3(data4, labels4, blue_th, blue_th0)
print(red_margin.sum(), np.amin(red_margin), np.amax(red_margin))
print(blue_margin.sum(), np.amin(blue_margin), np.amax(blue_margin))

'----------------------------W4 Question3 Simply inseparable------------------------------------'

data4_3 = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels4_3 = np.array([[1, -1, -1]])
th4_3 = np.array([[1, 1]]).T
th04_3 = -4
ref_margin = (2**0.5) /2
margin4_3 = margin3(data4_3, labels4_3, th4_3, th04_3)
'Solution 1: use map to iterate into every item in a list'
hinge_loss = lambda x: 1-x/ref_margin if x<ref_margin else 0 
list(map(hinge_loss, margin4_3[0])) #to iterate function over every element in margin4_3
'Solution 2: use np.vectorize so that the function is used as the numpy and could be applied to np directly'
test_vector = np.vectorize(hinge_loss)
print(test_vector(margin4_3))

'----------------------------W4 Question4 Hinges on the Loss------------------------------------'
Norm_thA = np.linalg.norm(np.array([[0.01280916, -1.42043497]]))

Norm_thB = np.linalg.norm(np.array([[0.45589866, -4.50220738]]))

Norm_thC = np.linalg.norm(np.array([[0.04828952, -4.13159675]]))

print(Norm_thA, Norm_thB, Norm_thC)

'When we increase \lambdaλ, we penalize larger values of \theta.θ. '
'In some cases, the may mean we incur non-zero or larger hinge loss '
'(where points are closers to the separator, or even where some points are misclassified).'
'Plot A has minimized \thetaθ to the degree that a point is misclassified; '
'this corresponds to the largest value of \lambdaλ, 0.3. (Note that for Plot A, ||\theta|| = 1.4205∣∣θ∣∣=1.4205, average hinge loss is 0.1861, and the margin is negative, -2.0735).'


'----------------------------W4 Question6 Implementing Gradient Descent------------------------------------'

def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)
def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def gd(f, df, x0, step_size_fn, max_iter):
    x_new = x0
    x_stack = np.array([x0])
    fs = []
    for i in range(max_iter):
        fs.append(f(x_new))
        x_old = x_new.copy()
        x_new = x_old- step_size_fn(0) * df(x_old)
        x_stack = np.vstack((x_stack, np.array([x_new])))
    
    return (x_new, fs, x_stack) #'Return the latest value of x'

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]


'Assume f is given, and to estimate the gradient of f at a particular x0'

def num_grad(f, delta=0.001):
    def df(x):
        d = x.shape[0]
        for i in range(d): #for each dimension in the X, 
            delta_v = np.zeros((d, 1))
            np.put(delta_v, i, delta) #put the delta in certain ith component 
            if i == 0:
                gd_col = (f(x+delta_v)-f(x-delta_v))/(2*delta)
            else:
                other_comp = (f(x+delta_v)-f(x-delta_v))/(2*delta)
                gd_col = np.vstack((np.array(gd_col), np.array(other_comp)))
        return np.reshape(gd_col, (d, 1))
        
    return df

def num_grad_MIT(f, delta=0.001):
    def df(x):
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            fxm = f(x)
            x[i,0] = xi + delta
            fxp = f(x)
            x[i,0] = xi
            g[i,0] = (fxp - fxm)/(2*delta)
        return g
    return df

'takes only a function f and uses this function and numerical gradient descent to return the local minimum.'
def minimize(f, x0, step_size_fn, max_iter):
    return gd(f, num_grad(f), x0, step_size_fn, max_iter) # as the num_grad will return function (i.e. df) 
#The derivative function is further used in the gd with input = updated x after every loop

'----------------------------W4 Question7 SVM------------------------------------'
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator

def hinge(v):
    return max(0, 1-v)

def hinge_loss(x, y, th, th0):
    ans = y*(th.T@x + th0)
    return np.where(ans<1, 1-ans, 0)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    hinge_array = hinge_loss(x, y, th, th0)
    return np.mean(hinge_array) + lam * (np.linalg.norm(th)**2)

'----------------------------W4 Question7.2 SVM------------------------------------'
X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

# Returns the gradient of hinge(v) with respect to v.

def d_hinge(v):
    return np.where(v>0, -1, 0)

def d_hinge_loss_th(x, y, th, th0):
    ans = hinge_loss(x, y, th, th0)
    d, n = ans.shape
    for i in range(n):
        if ans[0, i] == 0:
            new = np.zeros((2, 1))
        else:
            new = np.reshape(-x[:,i]*y[:, i], (2, 1))
        if i ==0:
            initial = new
        else:
            initial = np.hstack((initial, new))
    
    return initial #np.reshape(initial, (2, n))

def d_hinge_loss_th0(x, y, th, th0):
    ans = hinge_loss(x, y, th, th0)
    d, n = ans.shape
    for i in range(n):
        if ans[0, i] == 0:
            new = np.zeros((1, 1))
        else:
            new = np.reshape(-y[:,i], (1, 1))
        if i ==0:
            initial = new
        else:
            initial = np.hstack((initial, new))
    
    return initial 
        
def d_svm_obj_th(x, y, th, th0, lam):
    th_d = d_hinge_loss_th(x, y, th, th0)
    th_d_avg = np.reshape(th_d.mean(axis=1), (2, 1))
    reg = 2*lam*th
    return th_d_avg+reg

def d_svm_obj_th0(x, y, th, th0, lam):
    th0_d = d_hinge_loss_th0(x, y, th, th0)
    th0_d_avg = np.reshape(th0_d.mean(axis=1), (1, 1))
    
    return th0_d_avg

def svm_obj_grad(X, y, th, th0, lam):
    th_d = d_svm_obj_th(X, y, th, th0, lam)
    th0_d = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack((th_d, th0_d))

'----------------------------W4 Question7.3 SVM------------------------------------'

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

def gd(f, df, x0, step_size_fn, max_iter):
    x_new = x0
    x_stack = np.array([x0])
    fs = []
    for i in range(max_iter):
        fs.append(f(x_new))
        x_old = x_new.copy()
        x_new = x_old- step_size_fn(0) * df(x_old)
        x_stack = np.vstack((x_stack, np.array([x_new])))
    
    return (x_new, fs, x_stack) 
x_1,y_1=super_simple_separable()

'You will need to call gd, which is already defined for you as well; '
'your function batch_svm_min should return the values that gd does.'

def batch_svm_min(data, labels, lam):
    th_sep = sep_m_separator[0]
    th0_sep = sep_m_separator[1]
    th = np.zeros((2, 1))
    th0 = np.zeros((1, 1))
    th_stack = th
    th0_stack = th0
    fs = [svm_obj(data, labels, th, th0, lam)]
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
   
    for i in range(9):
        
        th_old = th.copy()
        th0_old = th0.copy()
        th = th_old - svm_min_step_size_fn(i)*d_svm_obj_th(data, labels, th_old, th0_old, lam)
        th0 = th0_old - svm_min_step_size_fn(i)*d_svm_obj_th0(data, labels, th_old, th0_old, lam)
        fs.append(svm_obj(data, labels, th, th0, lam))
        th_stack = np.hstack((th_stack, th))
        th0_stack = np.hstack((th0_stack, th0))
   
    return (np.vstack((th, th0)), fs, np.hstack((np.array([th_stack]), np.array([th0_stack]))).T)





        
        