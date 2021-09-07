# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 17:38:51 2021

@author: yungchen
"""
import numpy as np

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    """ Return a d x 1 np array.
        value_list is a python list of values of length d.

    >>> cv([1,2,3])
    array([[1],
           [2],
           [3]])
    """
    return np.transpose(rv(value_list))
def rv(value_list):
    """ Return a 1 x d np array.
        value_list is a python list of values of length d.

    >>> rv([1,2,3])
    array([[1, 2, 3]])
    """
    return np.array([value_list])
#--------------------------Question6C------------------------------------------
X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
Y = np.array([[1., 2.2, 2.8, 4.1]])
th = np.array([[1.0],[0.05]])
th0 = np.array([[0.]])

# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th

def lin_reg(x, th, th0):
    return np.dot(th.T, x) + th0
def d_lin_reg_th(x, th, th0):
    return x

# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th.
    
def square_loss(x, y, th, th0):
    return (y - lin_reg(x, th, th0))**2
def d_square_loss_th(x, y, th, th0):
    return 2*np.multiply(-d_lin_reg_th(x, th, th0),(y-lin_reg(x, th, th0)))

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses d_square_loss_th.
    
def mean_square_loss(x, y, th, th0):
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)
def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

#--------------------------Question6D------------------------------------------
# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th0. Hint: Think carefully about what the dimensions of the returned value should be!
def d_lin_reg_th0(x, th, th0):
    return np.full((1,x.shape[1]), 1) #Depends on the sample size n, will have column = n dimension
#th is dx1, th.T is 1xd, X may be dxn, then the derivetive of th0 is expected to have 1xn dims
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th0.
def d_square_loss_th0(x, y, th, th0):
    return -2*(y-lin_reg(x, th, th0))

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses d_square_loss_th0.
def d_mean_square_loss_th0(x, y, th, th0):
    return np.array([[np.mean(-2*(y-lin_reg(x, th, th0)))]])

#--------------------------Question7------------------------------------------
def ridge_obj(x, y, th, th0, lam):
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2

def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0)+ np.multiply(th, 2*lam) #should be the dimension of th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return np.array([[np.mean(-2*(y-lin_reg(x, th, th0)))]])

#--------------------------Question8 Stochastic Gradient Descent---------------
def downwards_line():
    X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
    y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
    return X, y

X8, y8 = downwards_line()  

def J(Xi, yi, w):
    # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
    return float(ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, yi, w):
    def f(w): return J(Xi, yi, w)
    return num_grad(f)(w)

def num_grad(f):
    def df(x):
        g = np.zeros(x.shape)
        delta = 0.001
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            xm = f(x)
            x[i,0] = xi + delta
            xp = f(x)
            x[i,0] = xi
            g[i,0] = (xp - xm)/(2*delta)
        return g
    return df

#w should be the value one gets after applying stochastic gradient descent to w0 for max_iter-1 iterations 
#(we call this the final step). 
#The first element of fs should be the value of J calculated with w0, and fs should have length max_iter; 
#similarly, the first element of ws should be w0, and ws should have length max_iter.
def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    n = X.shape[1]
    np.random.seed(0)
    fs = []; ws = []
    prev_w = w0
    for i in range(max_iter):
        j = np.random.randint(n)
        Xj = X[:, j:j+1]; yj = y[:, j:j+1]
        prev_f, prev_grad = J(Xj, yj, prev_w), dJ(Xj, yj, prev_w)
        fs.append(prev_f); ws.append(prev_w)
        if i == max_iter-1:
            return prev_w, fs, ws
        prev_w = prev_w - step_size_fn(i)*prev_grad

def sgd_MIT(X, y, J, dJ, w0, step_size_fn, max_iter):
    n = y.shape[1]
    prev_w = w0
    fs = []; ws = []
    np.random.seed(0)
    for i in range(max_iter):
        j = np.random.randint(n)
        Xj = X[:,j:j+1]; yj = y[:,j:j+1]
        prev_f, prev_grad = J(Xj, yj, prev_w), dJ(Xj, yj, prev_w)
        fs.append(prev_f); ws.append(prev_w)
        if i == max_iter - 1:
            return prev_w, fs, ws
        step = step_size_fn(i)
        prev_w = prev_w - step * prev_grad
        

#--------------------------Question9 Predicting mpg values---------------------
import code_for_hw5 as hw5
import itertools, functools, operator
        
def ridge_obj_grad(x, y, th, th0, lam):
    grad_th = d_ridge_obj_th(x, y, th, th0, lam)
    grad_th0 = d_ridge_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0]) 
   
def ridge_min(X, y, lam):
    """ Returns th, th0 that minimize the ridge regression objective
    
    Assumes that X is NOT 1-extended. Interfaces to our sgd by 1-extending
    and building corresponding initial weights.
    """
    def svm_min_step_size_fn(i):
        return 0.01/(i+1)**0.5

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    w_init = np.zeros((d+1, 1))

    def J(Xj, yj, th):
        return float(ridge_obj(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam))

    def dJ(Xj, yj, th):
        return ridge_obj_grad(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam)
    
    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w_init, svm_min_step_size_fn, 1000) 
    #w is the last updated weights
    #fs is the squared loss
    #ws is the whole list of weights during all updates
    return w[:-1,:], w[-1:,:]

def mul(seq):
    '''
    Given a list or numpy array of float or int elements, return the product 
    of all elements in the list/array.  
    '''
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    '''
    Transform raw features into polynomial features or order 'order'.
    If raw_features is a d by n numpy array, return a k by n numpy array 
    where k = sum_{i = 0}^order multichoose(d, i) (the number of all possible terms in the polynomial feature or order 'order')
    '''
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(1, order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

def eval_predictor(X_train, Y_train, X_test, Y_test, lam):
    th, th0 = ridge_min(X_train, Y_train, lam)
    return np.sqrt(mean_square_loss(X_test, Y_test, th, th0))

#Returns the mean RMSE from cross validation given a dataset (X, y), a value of lam,
#and number of folds, k
def xval_learning_alg(X, y, lam, k):
    '''
    Given a learning algorithm and data set, evaluate the learned classifier's score with k-fold
    cross validation. 
    
    learner is a learning algorithm, such as perceptron.
    data, labels = dataset and its labels.

    k: the "k" of k-fold cross validation
    '''
    _, n = X.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    X, y = X[:,idx], y[:,idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_predictor(X_train, y_train, X_test, y_test, lam)
    return score_sum/k


#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw5.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw5.standard and hw5.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

features2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = hw5.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = hw5.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = hw5.std_y(auto_values)
def raw(x):
    '''
    Make x into a nested list. Helper function to be used in auto_data_and_labels.
    >>> data = [1,2,3,4]
    >>> raw(data)
    [[1, 2, 3, 4]]
    '''
    return [x]
def std_vals(data, f):
    '''
    Helper function to be used inside auto_data_and_labels. Returns average and standard deviation of 
    data's f-th feature. 
    >>> data = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    >>> f=0
    >>> std_vals(data, f)
    (3.5, 2.5)
    >>> f=3
    >>> std_vals(data, f)
    (6.5, 2.5)
    '''
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)
def one_hot(v, entries):
    '''
    Outputs a one hot vector. Helper function to be used in auto_data_and_labels.
    v is the index of the "1" in the one-hot vector.
    entries is range(k) where k is the length of the desired onehot vector. 

    >>> one_hot(2, range(4))
    [0, 0, 1, 0]
    >>> one_hot(1, range(5))
    [0, 1, 0, 0, 0]
    '''
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

def auto_data_and_values(auto_data, features):
    features = [('mpg', raw)] + features 
    '''add mpg into the features set'''
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    return data_labels[:, 1:].T, data_labels[:, 0:1].T
#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     

# The function below is to give the min(RMSE) and corresponding polynomial order, and lambda
#x = auto_data[0] or auto_data[1]
#y = auto_values
def min_RMSE(x, y):
    for pol in range(1, 4): # pol will be 1, 2 or 3 with corresponding lambda range
        if pol ==3:
            lam = np.arange(0, 220, 20)
        else:
            lam = np.arange(0, 0.11, 0.01)
    
        RMSE_func = lambda z: xval_learning_alg(make_polynomial_feature_fun(pol)(x), y, z, 10)*sigma
        RMSE = list(map(RMSE_func, lam))
        if pol==1:
            min_RMSE = np.amin(RMSE)
            min_lam = np.argmin(lam)
            min_pol = pol
        else:
            if np.amin(RMSE)<min_RMSE:
                min_RMSE = np.amin(RMSE)
                min_lam = np.argmin(lam)
                min_pol = pol              
    return(min_pol, min_lam, min_RMSE)
                
'Solution 2: use np.vectorize so that the function is used as the numpy and could be applied to np directly'
#test_vector = np.vectorize(RMSE_pol1)
#Your code for cross-validation goes here
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset
