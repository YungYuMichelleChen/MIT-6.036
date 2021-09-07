# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:57:19 2021

@author: yungchen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def positive(x, th, th0):
   return np.sign(th.T@x + th0)
def score(data, labels, th, th0):
    A = positive(data, th, th0)== np.array(labels)
    return np.sum(A)

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    return ax

# Must either specify limits or existing ax
def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None,
                        ymin = None, ymax = None, res = 30):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)            
            
    ima = np.array([[predictor(x1i, x2i) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = cmap, norm = norm)

data = np.array(([[200, 800, 200, 800],
             [0.2,  0.2,  0.8,  0.8]]))
labels = np.array([[-1, -1, 1, 1]])

def perceptron(data, labels, params = {}, hook = None):
    T = params.get('T', 100)
    (d, n) = data.shape
    m = 0
    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                m += 1
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0

def margin(data, labels, factor):
    
    #(th, th0, mistake_count) = perceptron(data, labels)
    th = np.array([[0, 1]])
    th0 = -0.5
    (d, n) = data.shape
    data[0, :] = np.multiply(data[0, :],factor )
    gamma_list = []
    for i in range(n):
        gamma = labels[0, i]*(th@data[:, i]+th0)/np.linalg.norm(np.concatenate((th.T, np.array([[th0]]))))
        gamma_list.append(gamma)
    
    return np.amin(np.array([[gamma_list]]))

#-----------------------------Question2--------------------------------------
data2 =   np.array([[2, 3,  4,  5]])
labels2 = np.array([[1, 1, -1, -1]])
labels2_6 = np.array([[1, 1, -1, -1, 1, 1]])

#-----------------------------Question2D--------------------------------------
'''For example, one_hot(3,7) should return a column vector of length 7
with the entry at index 2 taking value 1 (indices start at 0) and other entries taking value 0.'''
def one_hot(x, k):
    vector = np.zeros(shape=(k, 1))
    vector[x-1, 0] = 1
    return vector

#create one hot encoding for the mobiles instead of data2
(d, n) = labels2_6.shape
mobile_encoding = one_hot(2, 6)
for i in range(3, 6):
    mobile_encoding = np.concatenate((mobile_encoding, one_hot(i, 6)), axis = 1)
    
for i in range(1, n+1):
    if i == 1:
        mobile6_hot = one_hot(1, 6)
    else:
        mobile6_hot = np.concatenate((mobile6_hot, one_hot(i, 6)), axis = 1)

#--------------------------Question3 Polynomial Features-----------------------
import functools
import operator
import itertools

def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def mul(seq):
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

#--------------------------Question3B------------------------------------------
def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

def test_linear_classifier_with_features(dataFun, learner, feature_fun,
                             draw = True, refresh = True, pause = True):
    raw_data, labels = dataFun()
    data = feature_fun(raw_data) if feature_fun else raw_data
    if draw:
        ax = plot_data(raw_data, labels)
        def hook(params):
            (th, th0) = params
            plot_nonlin_sep(
                lambda x1,x2: int(positive(feature_fun(cv([x1, x2])), th, th0)),
                ax = ax)
            plot_data(raw_data, labels, ax)
            plt.pause(0.05)
            print('th', th.T, 'th0', th0)
            if pause: input('press enter here to continue:')
    else:
        hook = None
    th, th0 = learner(data, labels, hook = hook)
    if hook: hook((th, th0))
    print("Final score", int(score(data, labels, th, th0)))
    print("Params", np.transpose(th), th0)

def test_with_features(dataFun, order = 2, draw=True, pause=True):
    test_linear_classifier_with_features(
        dataFun,                        # data
        perceptron,                     # learner
        make_polynomial_feature_fun(order), # feature maker
        draw=draw,
        pause=pause)

#--------------------------Part 2 Question4 Preparation------------------------------------------
import csv
import code_for_hw3_part2 as hw3

def load_auto_data(path_data):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

auto_data_all = hw3.load_auto_data('auto-mpg.tsv')   


features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]


features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features2)

#--------------------------Part 2 Question4.1------------------------------------------
#print(hw3.xval_learning_alg(hw3.perceptron,auto_data, auto_labels, 10 ), hw3.xval_learning_alg(hw3.averaged_perceptron,auto_data, auto_labels, 10 ))

#--------------------------Question5------------------------------------------

review_data = hw3.load_review_data('reviews.tsv')
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))
dictionary = hw3.bag_of_words(review_texts) #just to give unique index to each word appear in the reviews
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
#print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)
hw3.averaged_perceptron(review_bow_data, review_labels)
theta_avg, theta_0_avg = hw3.averaged_perceptron(review_bow_data, review_labels)
param_avg = np.concatenate((theta_avg.T, theta_0_avg), axis =1)
#to extract indices for theta which have most significant contribution towards positive/negative
param_avg[0].argsort()[-10:][::-1] 
param_avg[0].argsort()[:10][::-1] 
#to swap the key and values in the dictionary
reverse_dict = {v:k for k, v in dictionary.items()}
#[reverse_dict[count] for count in param_avg[0].argsort()[-10:][::-1].tolist()]
#[reverse_dict[count] for count in param_avg[0].argsort()[:10][::-1].tolist()]

#--------------------------Question6 MNIST------------------------------------------
mnist_data_all = hw3.load_mnist_data(range(10))
n0 = np.array(mnist_data_all[0]['labels']).shape[1]
n1 = np.array(mnist_data_all[1]['labels']).shape[1]
image0 = np.reshape(np.array(mnist_data_all[0]['images']).T, (28*28, n0))
image1 = np.reshape(np.array(mnist_data_all[1]['images']).T, (28*28, n1))
labels0 = np.full((1, n0), -1)
labels1 = np.full((1, n1), 1)
#np.full((2, 3), 7)
data01 = np.concatenate((image0, image1), axis=1)
labels01 = np.concatenate((labels0, labels1), axis=1)

#6.2) Feature evaluation
#n6 = np.array(mnist_data_all[6]['labels']).shape[1]
#n9 = np.array(mnist_data_all[9]['labels']).shape[1]
#image6 = np.reshape(np.array(mnist_data_all[6]['images']).T, (28*28, n6))
#image9 = np.reshape(np.array(mnist_data_all[9]['images']).T, (28*28, n9))
#labels6 = np.array(mnist_data_all[6]['labels'])
#labels9 = np.array(mnist_data_all[9]['labels'])
#data90 = np.concatenate((image9, image0), axis=1)
#labels90 = np.concatenate((labels9, labels0), axis=1)
#print(hw3.get_classification_accuracy(data90, labels90) )
print(hw3.get_classification_accuracy(data01, labels01))

#--------------------------Question6 MNIST------------------------------------------
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    #raise Exception("implement me!")
    return np.reshape(x, (28*28, x.shape[0]))

acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    return np.array([np.mean(x, axis = 1)]).T

def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.array([np.mean(x, axis=0)]).T

def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    (m, n) = x.shape
    top =  x[:round(m/2-0.1), :]
    bottom = x[round(m/2-0.1):, :]
    
    return np.array([[np.mean(top), np.mean(bottom)]]).T

def ult_row_accuracy(x1,x2, labels):
    for i in range(x1.shape[0]):
        if i == 0 :
            clean1 = row_average_features(x1[0])
        else:
            clean1 = np.hstack((clean1, row_average_features(x1[i])))
            
    for i in range(x2.shape[0]):
        if i == 0 :
            clean2 = row_average_features(x2[0])
        else:
            clean2 = np.hstack((clean2, row_average_features(x2[i]))) 
    
    ult_data = np.hstack((clean1, clean2))
    return(hw3.get_classification_accuracy(ult_data, labels))

def ult_col_accuracy(x1,x2, labels):
    for i in range(x1.shape[0]):
        if i == 0 :
            clean1 = col_average_features(x1[0])
        else:
            clean1 = np.hstack((clean1,col_average_features(x1[i])))
            
    for i in range(x2.shape[0]):
        if i == 0 :
            clean2 = col_average_features(x2[0])
        else:
            clean2 = np.hstack((clean2, col_average_features(x2[i]))) 
    
    ult_data = np.hstack((clean1, clean2))
    return(hw3.get_classification_accuracy(ult_data, labels))
    
def ult_top_bottom_accuracy(x1,x2, labels):
    for i in range(x1.shape[0]):
        if i == 0 :
            clean1 = top_bottom_features(x1[0])
        else:
            clean1 = np.hstack((clean1,top_bottom_features(x1[i])))
            
    for i in range(x2.shape[0]):
        if i == 0 :
            clean2 = top_bottom_features(x2[0])
        else:
            clean2 = np.hstack((clean2, top_bottom_features(x2[i]))) 
    
    ult_data = np.hstack((clean1, clean2))
    return(hw3.get_classification_accuracy(ult_data, labels))
    
print(ult_row_accuracy(np.array(d9),np.array(d0), labels90), ult_col_accuracy(np.array(d9),np.array(d0), labels90), ult_top_bottom_accuracy(np.array(d9),np.array(d0), labels90))