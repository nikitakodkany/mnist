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
    
    return xtrain, ytrainEncoded, xtest, ytestEncoded
