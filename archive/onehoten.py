import numpy as np
from archive.loader import toReturn

xtrain, ytrain, xtest, ytest = toReturn()

def oneHotEncoding(y):
    encoded = []

    for i in y:
        array = np.zeros(10, dtype = int)
        array[i] = 1
        encoded.append(array)

    return np.array(encoded)

def toReturnEncoded():
    ytrainEncoded = oneHotEncoding(ytrain)
    ytestEncoded = oneHotEncoding(ytest)

    #transformed data
    xtrainReshaped = xtrain.T.reshape(784,60000)
    ytrainEncoded = ytrainEncoded.reshape(60000, 10).T
    xtestReshaped = xtest.T.reshape(784, 10000)
    ytestEncoded = ytestEncoded.reshape(10000, 10).T

    return xtrainReshaped, ytrainEncoded, xtestReshaped, ytestEncoded, xtrain, ytrain
