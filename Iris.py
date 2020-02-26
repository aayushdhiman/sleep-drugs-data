# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:22:45 2019

@author: 789136
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as model_selection 

#******************************************************************************

s_Xtest = 0
s_Ytest = 0


#******************************************************************************

#------------------------------------------------------------------------------
def createData():
    
    global save_Xtest
    global save_Ytest
    
    fileName = 'irisdata.csv'
    print('File Name: ', fileName)
    raw_data = open(fileName, 'rt')
    
    data = np.loadtxt(raw_data, usecols = (0, 1, 2, 3, 4), skiprows = 1,
                      delimiter = ',')
    
    x = (data[:, 0:4])
    
    y = (data[:, 4:])
    ohe = OneHotEncoder(categories='auto')
    y = ohe.fit_transform(y).toarray()
    
    
    savedMean = np.mean(x, axis=0)
    savedStd = np.std(x, axis=0)
    print('mean: ', savedMean, 'std', savedStd)
    x = (x - savedMean)/savedStd
#    '''
    
    x = np.concatenate((np.ones((len(x),1)), x), axis = 1)
    
    #----------------------------Shuffling-------------------------------------
    x, save_Xtest, y, save_Ytest = model_selection.train_test_split(x, y, 
                                            train_size = .75, test_size = .25,
                                            random_state = 101) 
    #--------------------------------------------------------------------------
    
    return x, y

#------------------------------------------------------------------------------

#******************************************************************************

x, y = createData()
w = np.array([[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
a = 1
it = 10000

#******************************************************************************

#------------------------------------------------------------------------------
def sigmoid(z):
#    z = np.dot(x, w)
    e = np.exp(-z)
    
    return 1/(1+e)


#------------------------------------------------------------------------------
def cost(w, x, y):
    
    s = sigmoid(np.dot(x, w))
    t2 = np.log(s)
    t1_2 = y*t2
    t4 = np.log(1-s)
    t3 = 1-y
    t3_4 = t3*t4
    top = t1_2+t3_4
    t = sum(top)

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
        
#    return w, wA, cA
#    print('\niterations: ', i)
#    return w, cA
    return w
    
#------------------------------------------------------------------------------ 
def train(w, x, y):
    '''
    I = 0
    for I in range(500):
        for i in range(len(w[0])): 
            w[:, i] = run(w[:, i], x, y[:, i])
        I+=1
    '''
    
    for i in range(len(w[0])): 
            w[:, i] = run(w[:, i], x, y[:, i])
            
    return w
    
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
def plot(w, x):
    
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_title('Data Boundaries')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    '''
    Y = np.dot(x, w)
    x2 = np.arange(0, len(x), 1)
    
    ax.plot(x2, Y, 'bo')
    '''
    
    l1 = (-w[0][0]/w[2][0])-((w[1][0]/w[2][0])*x)
    l2 = (-w[0][1]/w[2][1])-((w[1][1]/w[2][1])*x)
    l3 = (-w[0][2]/w[2][2])-((w[1][2]/w[2][2])*x)
    
    ax.plot(x[:, 1], x[:, 2], 'bo')
    
    ax.plot(x[:, 1], l1, 'b')
    ax.plot(x[:, 1], l2, 'r')
    ax.plot(x[:, 1], l3, 'g')
    



#------------------------------------------------------------------------------

#******************************************************************************

print('\nInitial Weights: \n', w)
w = train(w, x, y)
print('\nFinal Weights: \n', w)


err, acc, p, pred, Y = processData(w, x, y)




    #----------------------------Testing Data----------------------------------
eT, aT, pT, prT, T = processData(w, save_Xtest, save_Ytest)
    #--------------------------------------------------------------------------

print('\n')
print("% 23s" % ('Train'))
print("% 20s: % 5.3f" % ('Error', err))
print("% 20s: % 5.3f" % ('Accuracy', acc))

print("% 23s" % ('Test'))
print("% 20s: % 5.3f" % ('Error', eT))
print("% 20s: % 5.3f" % ('Accuracy', aT))

#plot(w, x)

    

#******************************************************************************













