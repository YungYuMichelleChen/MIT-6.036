# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def positive(x, th, th0):
   return np.sign(th.T@x + th0)
def score(data, labels, th, th0):
    A = positive(data, th, th0)== np.array(labels)
    return np.sum(A)


#-----------------------------4.2d--------------------------------------
x1 = [-1, 1]
x2 = [1, -1]
x3 = [1,1]
x4 = [2, 2]
y = [1, 1, -1, -1]

data = np.array([x1, x2, x3, x4])

print(data)

Theta = [0, 0]
Theta0 = 0 
mistake_count = 0 
for i in range(len(data)):
    for j in range(data.shape[0]):
        if y[j]*(np.matmul(np.array(Theta), data[j])+Theta0) <= 0 :
            #Update Theta with x * y 
            #Update Theta0 with y
            Theta = Theta + np.multiply(data[j], y[j])
            Theta0 +=  y[j]
            mistake_count += 1

print(Theta0,Theta, mistake_count )
print(np.linalg.norm(data[0]))

#-----------------------------6.2--------------------------------------
gamma = [.00001, .0001, .001, .01, .1, .2, .5]
list1 = []

for i in range(len(gamma)):
    list1.append((1/gamma[i])**2)
   
print(list1)

#-----------------------------7--------------------------------------
def perceptron_test(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    th0 = 0
    th = np.zeros(data.shape[0])

    # Your implementation here
    
    for t in range(T):
        for i in range(data.shape[1]):
            if labels[0][i]*(np.matmul(np.transpose(th), data[:, i]) + th0)<= 0:
	            th = th + np.multiply(data[:, i], labels[0][i])
	            th0 += labels[0][i]


    return (np.transpose(np.array([th])), np.array([[th0]]))

def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0

#---------------------Generate data-----------------------------------
x1 = [-1, 1]
x2 = [1, -1]
x3 = [1,1]
x4 = [2, 2]   
x5 = [2, -3]
x6 = [4,4]
x7 = [3, 5]
data9 = np.array([x1, x2, x3, x4, x5, x6, x7])
datat = np.array([x1, x2, x3])

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    th0 = 0.0
    th = np.zeros(data.shape[0])
    th0s = 0.0
    ths = np.zeros(data.shape[0])
    (d, n) = data.shape

    # Your implementation here
    
    for t in range(T):
        for i in range(n):
	        if labels[0][i]*(np.matmul(np.transpose(th), data[:, i]) + th0) <= 0:
	            th = th + np.multiply(data[:, i], labels[0][i])
	            th0 += labels[0][i]
	        ths = ths+th
	        th0s = th0s+th0


    return (np.transpose(np.array([ths])/(n*T)), np.array([[th0s]])/(n*T))

def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k



def gen_lin_separable(num_points=20, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    ''' 
    Generate linearly separable dataset X, y given theta and theta0
    Return X, y where
    X is a numpy array where each column represents a dim-dimensional data point
    y is a column vector of 1s and -1s
    '''
    X = np.random.uniform(low=-5, high=5, size=(dim, num_points))
    y = np.sign(np.dot(np.transpose(th), X) + th_0)
    return X, y
def gen_flipped_lin_separable(num_points=20, pflip=0.25, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    '''
    Generate difficult (usually not linearly separable) data sets by
    "flipping" labels with some probability.
    Returns a method which takes num_points and flips labels with pflip
    '''
    def flip_generator(num_points=20):
        X, y = gen_lin_separable(num_points, th, th_0, dim)
        flip = np.random.uniform(low=0, high=1, size=(num_points,))
        for i in range(num_points):
            if flip[i] < pflip: y[0,i] = -y[0,i]
        return X, y
    return flip_generator

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    correct_num = score(data_test,labels_test, th, th0 )
    (d,n) = data_test.shape
    
    return correct_num/n

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    sum_test = 0 
    for i in range(it):
        (data1, labels1) = data_gen(n_train)
        (data2, labels2) = data_gen(n_test)
        sum_test += eval_classifier(learner, data1, labels1, data1, labels1)
        #sum_test += eval_classifier(learner, data1, labels1, data2, labels2)
        
    return sum_test/it

def xval_learning_alg_test(learner, data_gen, n_train, k, it):
    (data, labels) = data_gen(n_train)
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for j in range(it):
        for i in range(k):
            data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
            labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
            data_test = np.array(s_data[i])
            labels_test = np.array(s_labels[i])
            score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
        
    return score_sum/(k*it)



