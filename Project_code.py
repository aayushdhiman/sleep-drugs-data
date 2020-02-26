# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:28:48 2020

@author: 789136
"""

import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection as model_selection
from keras import models
from keras import layers


#------------------------------------------------------------------------------
def standardize(x):
    savedMean = np.mean(x, axis=0)
    savedStd = np.std(x, axis=0)
#    print('mean: ', savedMean, 'std', savedStd)
    x = (x - savedMean)/savedStd
    return x

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#                               Setting Up
#------------------------------------------------------------------------------

'''

1    TTOREM           00
2    PERWAKE          01
3    PERREM           02
4   PERLIGHT          03
5    PERDEEP          04 
6    NWAKES           05
10    DAYSUNDAY       06
11    DAYMONDAY       07
12    DAYTUESDAY      08
13    DAYWEDNESDAY    09
14    DAYTHURSDAY     10
15    DAYFRIDAY       11
16    DAYSATURDAY     12
17    CYMDOSE         13
18    MELDOSE         14
19    TEMDOSE         15

22    CYM30           Y0
23    CYM60           Y1
24    MEL3            Y2
25    MEL6            Y3
26    TEM15           Y4
27    TEM30           Y5
28    CYM30MEL3       Y6
29    CYM30MEL6       Y7
30    CYM60MEL3       Y8
31    CYM60MEL6       Y9
32    CYM60TEL15      Y10
33    CYM60TEL30      Y11
34    NODRUG          Y12

'''

global save_Xtest
global save_Ytest
global save_Xval
global save_Yval


fileName = 'Dataset.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')


data = np.loadtxt(raw_data, usecols = (1, 2, 3, 4, 5, 6, 10, 11,12,13,14,15,16,
                      17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34), 
                      skiprows = 1, delimiter=",", dtype="str")

totally_float_data = data[(data != 'NA').all(axis=1)]
float_data = totally_float_data.astype(float)

#separating
y = float_data[:, 16:29]
X = float_data[:, :16]

#standardizing
TTOREM = standardize(X[:, 0:1])
PERWAKE = standardize(X[:, 1:2])
PERREM = standardize(X[:, 2:3])
PERLIGHT = standardize(X[:, 3:4])
PERDEEP = standardize(X[:, 4:5])
NWAKES = standardize(X[:, 5:6])
CYMDOSE = standardize(X[:, 13:14])
MELDOSE = standardize(X[:, 14:15])
TELDOSE = standardize(X[:, 15:16])

#Concatination
x = np.hstack((TTOREM, PERWAKE, PERREM, PERLIGHT, PERDEEP, NWAKES, X[:, 6:13], 
              CYMDOSE, MELDOSE, TELDOSE))

#creating training and testing data
x, save_Xtest, y, save_Ytest = model_selection.train_test_split(x, y, 
                            train_size = 70, test_size=.30,random_state=101)

#Creating test and validation data
save_Xtest, save_Xval, save_Ytest, save_Yval = model_selection.train_test_split(
           save_Xtest, save_Ytest,train_size=.70,test_size=.30,random_state=101)

#Weights
w = np.zeros((16, 13))
a = .5
it = 10000


#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def sigmoid(z):
#    z = np.dot(x, w)
    e = np.exp(-z)
    
    return 1/(1+e)


#------------------------------------------------------------------------------
def cost(w, x, y):
    
    s = sigmoid(np.dot(x, w))
    t2 = np.log(s)
    print('1 ', len(t2))
    t2 = t2.reshape(1, len(t2))
    print('t2: ', t2)
    t1_2 = t2*y
    t4 = np.log(1-s)
    t3 = 1-y
    t3_4 = t3*t4
    top = t1_2+t3_4
    t = sum(top)
    print('t len: ', len(t))

    return (-t)/len(x)

#------------------------------------------------------------------------------
def gradient(w, x, y):
    
    s = sigmoid(np.dot(x, w))
    diff = s-y
    val = np.dot(x.transpose(), diff)
#    print(val)
    return val/len(y)

#------------------------------------------------------------------------------ 
def run(w, x, y):
    
#    cA = []
    wA = []
    i = 0
    
    L2 = np.linalg.norm(gradient(w, x, y))
    
    while i < it and L2 != .001:
        
        gd = gradient(w, x, y)
        g = gd
        L2 = np.linalg.norm(g)
        
        w = w - (a*g)
#        print('test ', a*g)
#        print('CHECK IF SLICING IS CORRECT OR NOT')
        wA.append(w)
#        cA.append(cost(w, x, y))
        if i%500==0:
            print('iterations: %5i , norm: %1.17f' % (i, L2))
        i += 1
        Ct = cost(w, x, y)
        print('train cost: ', Ct)
        Cv = cost(w, save_Xval, save_Yval)
        
#    return w, wA, cA
#    print('\niterations: ', i)
#    return w, cA
    return w, Ct, Cv
    
#------------------------------------------------------------------------------ 
def train(w, x, y):
    '''
    I = 0
    for I in range(500):
        for i in range(len(w[0])): 
            w[:, i] = run(w[:, i], x, y[:, i])
        I+=1
    '''
    
    Ctrain = []
    Cval = []
    
    for i in range(len(w[0])): 
            w[:, i], Ct, Cv = run(w[:, i], x, y[:, i])
            Ctrain.append(Ct)
            Cval.append(Cv)

    
    return w, Ctrain, Cval
    
#------------------------------------------------------------------------------
def processData(w, x, y):
    
    z = np.dot(x, w)
    prob = sigmoid(z)
    
    pred = np.argmax(prob, axis = 1).reshape(len(x), 1)
    Y = np.argmax(y, axis = 1).reshape(len(y), 1)
    
    diff = Y - pred
    
    co = np.count_nonzero(diff)
    
    e = co/len(diff)
    
    ac = 1-e

    
    return e, ac, prob, pred, Y

    
#------------------------------------------------------------------------------




#******************************************************************************

w, Ctrain, Cval = train(w, x, y)




#******************************************************************************


