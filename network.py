#libraries
import csv
import math
import time
import tkinter
import numpy as np
from time import sleep
from pymsgbox import *
from archive.activations import *
from progressbar import progressbar
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message= "the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses")
warnings.filterwarnings("ignore", message= "unclosed file <_io.TextIOWrapper name='dymplot.csv' mode='r' encoding='UTF-8'>")


from archive.onehoten import toReturnEncoded
xtrain, ytrain, xtest, ytest, X, Y = toReturnEncoded()


# #data visualization
# ximage = X.T
# image = ximage[:,:,1]
# fig = plt.figure
# plt.imshow(image, cmap = 'gray')
# plt.show()

#normalize
xtrain = (xtrain - np.mean(xtrain))/np.std(xtrain)
xtest = (xtest - np.mean(xtest))/ np.std(xtest)

#initialization
m = xtrain.shape[1]
lambd = 0.1

class network:
    def __init__(self,xtrain, ytrain):

        #layer size
        n_x = xtrain.shape[0]
        n_1 = 112
        n_2 = 86
        n_3 = 90
        n_y = ytrain.shape[0]

        np.random.seed(2)
        self.cache = {}

        self.cache['x'] = xtrain
        self.cache['y'] = ytrain

        #HE initialization
        self.cache['w1'] = np.random.randn(n_1, n_x) * (np.sqrt(2.0 / n_x))
        self.cache['w2'] = np.random.randn(n_2, n_1) * (np.sqrt(2.0 / n_1))
        self.cache['w3'] = np.random.randn(n_3, n_2) * (np.sqrt(2.0 / n_2))
        self.cache['w4'] = np.random.randn(n_y, n_3) * (np.sqrt(2.0 / n_3))
        #bias initialization
        self.cache['b1'] = np.zeros((n_1, 1))
        self.cache['b2'] = np.zeros((n_2, 1))
        self.cache['b3'] = np.zeros((n_3, 1))
        self.cache['b4'] = np.zeros((n_y, 1))

    def forward(self, x):
        #forward propagation with bias
        w1, w2, w3, w4, b1, b2, b3, b4 = self.get('w1', 'w2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4')

        z1 = np.dot(w1, x) + b1
        a1 = leakyrelu(z1)
        dz1 = leakyrelu_prime(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = leakyrelu(z2)
        dz2 = leakyrelu_prime(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = leakyrelu(z3)
        dz3 = leakyrelu_prime(z3)

        z4 = np.dot(w4, a3) + b4
        a4 = softmax(z4)

        self.put(dz1 = dz1, dz2 = dz2, dz3 = dz3, a1 = a1, a2 = a2, a3 = a3, a4 = a4)
        return a4

    def cost(self, ypred, y):
        #cost with l2 regularization
        w1, w2, w3, w4 = self.get('w1', 'w2', 'w3', 'w4')

        cost_entropy = cost = -np.mean(y * np.log(ypred + 1e-8))
        l2_regularization = lambd / (2*m) * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)) + np.sum(np.square(w4)))

        cost = cost_entropy + l2_regularization
        return cost


    def backward(self, x, y):
        #back propagation with bias and regularization
        a1, a2, a3, a4, w1, w2, w3, w4, dz1, dz2, dz3= self.get('a1', 'a2', 'a3', 'a4', 'w1', 'w2', 'w3', 'w4', 'dz1', 'dz2', 'dz3')

        t4 = a4 - y
        dw4 = 1./m * (np.dot(t4, a3.T) + ((lambd/m) * w4))
        db4 = 1./m * (np.sum(t4, axis=1, keepdims = True))

        t3 = np.multiply(dz3, np.dot(w4.T, t4))
        dw3 = 1./m * (np.dot(t3, a2.T) + ((lambd/m) * w3))
        db3 = 1./m * (np.sum(t3, axis=1, keepdims = True))

        t2 = np.multiply(dz2, np.dot(w3.T, t3))
        dw2 = 1./m * (np.dot(t2, a1.T) + ((lambd/m) * w2))
        db2 = 1./m * (np.sum(t2, axis=1, keepdims = True))

        t1 = np.multiply(dz1, np.dot(w2.T, t2))
        dw1 = 1./m * (np.dot(t1, x.T) + ((lambd/m) * w1))
        db1 = 1./m * (np.sum(t1, axis=1, keepdims = True))

        self.put(dw4 = dw4, dw3 = dw3, dw2 = dw2, dw1 = dw1, db4 = db4, db3 = db3, db2 = db2, db1 = db1)

    def update(self, rate = 0.05):
        w1, w2, w3, w4, dw1, dw2, dw3, dw4, b1, b2, b3, b4, db1, db2, db3, db4 = self.get('w1', 'w2', 'w3', 'w4','dw1', 'dw2', 'dw3', 'dw4', 'b1', 'b2', 'b3', 'b4', 'db1', 'db2', 'db3', 'db4')

        w1 -= rate * dw1
        b1 -= rate * db1

        w2 -= rate * dw2
        b2 -= rate * db2

        w3 -= rate * dw3
        b3 -= rate * db3

        w4 -= rate * dw4
        b4 -= rate * db4

        self.put(w1 = w1, w2 = w2, w3 = w3, w4 = w4, b1 = b1, b2 = b2, b3 = b3, b4 = b4)

    def put(self, **kwargs):
        for key, value in kwargs.items():
            self.cache[key] = value


    def get(self, *args):
        x = tuple(map(lambda x: self.cache[x], args))
        return x

    def func_minibatch(self, batchsize = 32):

        x, y = self.get('x', 'y')
        np.random.seed(2)

        mini_batches =[]
        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation]

        #number of minibatches of batchsize in partitioning
        num_complete_minibatches = math.floor(m/ batchsize)
        for k in range(0, num_complete_minibatches):
            minibatch_x = shuffled_x[:, k * batchsize : (k+1) * batchsize]
            minibatch_y = shuffled_y[:, k * batchsize : (k+1) * batchsize]
            mini_batch = (minibatch_x, minibatch_y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % batchsize != 0:
            minibatch_x = shuffled_x[:, num_complete_minibatches * batchsize : m]
            minibatch_y = shuffled_y[:, num_complete_minibatches * batchsize : m]
            mini_batch = (minibatch_x, minibatch_y)
            mini_batches.append(mini_batch)

        return mini_batches


def main():

    costs = []
    epochs = 10
    n = network(xtrain, ytrain)


    for epoch in progressbar(range(epochs)):
        minibatches = n.func_minibatch()
        for minibatch in minibatches:
            (minibatch_x, minibatch_y) = minibatch
            ypred = n.forward(minibatch_x)
            costs.append(n.cost(ypred, minibatch_y))
            n.backward(minibatch_x, minibatch_y)
            n.update()

            ypred = np.argmax(ypred.T, axis = 1).reshape(32,1)
            minibatch_y = np.argmax(minibatch_y.T, axis = 1).reshape(32,1)
            count = 0
            for a, b in zip(ypred, minibatch_y):
                if a == b:
                    count += 1
    print("Model accuracy of minibatch: {}%".format((count*100)/ypred.shape[0]))

    ypredicted = n.forward(xtrain)
    ypredicted = np.argmax(ypredicted.T, axis = 1)
    print(ypredicted.shape)

    # To plot Cost vs Epochs
    plt.title('Cost vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(costs)
    plt.show()


    alert(text='Training Complete!', title='ALERT', button='OK')

if __name__ == '__main__':
    main()
