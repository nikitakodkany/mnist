#libraries
import sys
import numpy as np
import time
from archive.activations import *
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from archive.onehoten import toReturnEncoded
from sklearn.preprocessing import StandardScaler

xtrain, ytrain, xtest, ytest = toReturnEncoded()

#feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
ytrain = sc.fit_transform(xtrain)

#initialization
m = xtrain.shape[1]
lambd = 0.1

class network:
    def __init__(self,xtrain, ytrain):
        #layer size
        n_x = xtrain.shape[0]
        n_1 = 28
        n_2 = 24
        n_3 = 16
        n_y = ytrain.shape[0]

        np.random.seed(2)
        self.cache = {}

        self.cache['x'] = xtrain
        self.cache['y'] = ytrain
        # #HE initialization
        # self.cache['w1'] = np.random.randn(n_1, n_x) * (np.sqrt(2.0 / n_x))
        # self.cache['w2'] = np.random.randn(n_2, n_1) * (np.sqrt(2.0 / n_1))
        # self.cache['w3'] = np.random.randn(n_3, n_2) * (np.sqrt(2.0 / n_2))
        # self.cache['w4'] = np.random.randn(n_y, n_3) * (np.sqrt(2.0 / n_3))

        # Xavier initialization
        self.cache['w1'] = np.random.randn(n_1, n_x) * (np.sqrt(2.0 / (n_x + n_1)))
        self.cache['w2'] = np.random.randn(n_2, n_1) * (np.sqrt(2.0 / (n_1 + n_2)))
        self.cache['w3'] = np.random.randn(n_3, n_2) * (np.sqrt(2.0 / (n_2 + n_3)))
        self.cache['w4'] = np.random.randn(n_y, n_3) * (np.sqrt(2.0 / (n_3 + n_y)))
        #bias initialization
        self.cache['b1'] = np.zeros((n_1, 1))
        self.cache['b2'] = np.zeros((n_2, 1))
        self.cache['b3'] = np.zeros((n_3, 1))
        self.cache['b4'] = np.zeros((n_y, 1))

    def forward(self, x = None):
        #forward propagation with bias

        if x is None:
            x = self.cache['x']

        w1, w2, w3, w4, b1, b2, b3, b4 = self.get('w1', 'w2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4')

        z1 = np.dot(w1, x) + b1 #e+71
        a1 = leakyrelu(z1)
        dz1 = leakyrelu_prime(z1)

        z2 = np.dot(w2, a1) + b2 #164
        a2 = leakyrelu(z2) #inf values
        print(z2)
        dz2 = leakyrelu_prime(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = leakyrelu(z3)
        dz3 = leakyrelu_prime(z3)

        z4 = np.dot(w4, a3) + b4
        a4 = softmax(z4)

        self.put(dz1 = dz1, dz2 = dz2, dz3 = dz3, a1 = a1, a2 = a2, a3 = a3, a4 = a4)
        return a4

    def cost(self, ypred, y= None):
        #cost with l2 regularization

        if y is None:
            y = self.cache['y']

        w1, w2, w3, w4 = self.get('w1', 'w2', 'w3', 'w4')

        cost_entropy = cost = -np.mean(y * np.log(ypred + 1e-8))
        # l2_regularization = lambd / (2*m) * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)) + np.sum(np.square(w4)))

        cost = cost_entropy #+ l2_regularization
        return cost


    def backward(self):
        #back propagation with bias and regularization
        a1, a2, a3, a4, y, w1, w2, w3, w4, dz1, dz2, dz3, x = self.get('a1', 'a2', 'a3', 'a4', 'y', 'w1', 'w2', 'w3', 'w4', 'dz1', 'dz2', 'dz3', 'x')

        t4 = a4 - y
        # dw4 = 1./m * (np.dot(t4, a3.T) + ((lambd/m) * w4))
        dw4 = 1./m * (np.dot(t4, a3.T))
        db4 = 1./m * (np.sum(t4, axis=1, keepdims = True))

        t3 = np.multiply(dz3, np.dot(w4.T, t4))
        # dw3 = 1./m * (np.dot(t3, a2.T) + ((lambd/m) * w3))
        dw3 = 1./m * (np.dot(t3, a2.T))
        db3 = 1./m * (np.sum(t3, axis=1, keepdims = True))

        t2 = np.multiply(dz2, np.dot(w3.T, t3))
        # dw2 = 1./m * (np.dot(t2, a1.T) + ((lambd/m) * w2))
        dw2 = 1./m * (np.dot(t2, a1.T))
        db2 = 1./m * (np.sum(t2, axis=1, keepdims = True))

        t1 = np.multiply(dz1, np.dot(w2.T, t2))
        # dw1 = 1./m * (np.dot(t1, x.T) + ((lambd/m) * w1))
        dw1 = 1./m * (np.dot(t1, x.T))
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

def main():
    bar = ProgressBar()
    costs = []
    epoch = 10
    n = network(xtrain, ytrain)

    print("Training Started...")
    # tic = time.process_time()

    # np.warnings.filterwarnings('ignore')

    for i in bar(range(epoch)):
        ypred = n.forward()
        costs.append(n.cost(ypred))
        n.backward()
        n.update()

    # toc = time.process_time()

    print("Training Done!")
    # print("Time Elapsed: ", toc-tic )

    # yhat = n.forward(xtest)
    #
    # yhat_rounded = np.argmax(yhat, axis = 0)
    # ytest_rounded = np.argmax(ytest, axis = 0)
    #
    # #classification report on test dataset
    # print(classification_report(ytest_rounded, yhat_rounded))
    #
    # print(costs)
    #
    # # cost vs epochs graph
    # plt.plot(costs)
    # plt.show()


if __name__ == '__main__':
    main()
