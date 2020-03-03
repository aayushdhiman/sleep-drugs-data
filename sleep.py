import keras
from keras import models
from keras import layers
keras.__version__
import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt

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

def standardize(x):
    savedMean = np.mean(x, axis=0)
    savedStd = np.std(x, axis=0)
#    print('mean: ', savedMean, 'std', savedStd)
    x = (x - savedMean)/savedStd
    return x


def create_data():
    fileName = 'Dataset.csv'
    print("fileName: ", fileName)
    raw_data = open("U:\\My Drive\\nn\\"+fileName, 'rt')

    data = np.loadtxt(raw_data, usecols = (1, 2, 3, 4, 5, 6, 10, 11,12,13,14,15,16,
                      17,18,19,22,23,24,25,26,27,28,29,30,31,32,33), 
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
    
    #Concatenation
    x = np.hstack((TTOREM, PERWAKE, PERREM, PERLIGHT, PERDEEP, NWAKES, X[:, 6:13], 
                  CYMDOSE, MELDOSE, TELDOSE))
    
    #creating training and testing data
    x_train, save_Xtest, y_train, save_Ytest = model_selection.train_test_split(x, y, 
                                train_size = .70, test_size=.30,random_state=101)
    
    #Creating test and validation data
    # save_Xtest, save_Xval, save_Ytest, save_Yval = model_selection.train_test_split(
    #            save_Xtest, save_Ytest,train_size=.50,test_size=.50,random_state=101)
    
    return x_train, save_Xtest, y_train, save_Ytest

x_train, x_test, y_train, y_test = create_data()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(len(x_train[0]),)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

yTest = np.argmax(y_test, axis=1)
yTrain = np.argmax(y_train, axis=1)

#'''
history = model.fit(x_train, yTrain, 
                    epochs=100, batch_size=1)
'''
mae_history = history.history['mean_absolute_error']
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

predicted_drugs = model.predict(x_test)

accuracy = y_test - predicted_drugs
res =  [abs(ele) for ele in accuracy] 
q = len(res)
res = sum(res)/q
error = (abs(res - test_mae_score)*100)

print("Verification of test_mae_score: \ntest_mae_score:")
print(test_mae_score)
print("Percent Error of test_mae_score:")
print(error)

fig = plt.figure()
ax1 = fig.add_axes([1, 1, 1, 1])
ax1.set_title('Mean Absolute Error')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Absolute Error')
ax1.plot(mae_history, range(len(mae_history)))
'''



