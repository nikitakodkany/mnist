#libraries
import math
import time
import tkinter
import numpy as np
from pymsgbox import *
from archive.functions import *
from progressbar import progressbar
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message= "the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses")
warnings.filterwarnings("ignore", message= "unclosed file <_io.TextIOWrapper name='dymplot.csv' mode='r' encoding='UTF-8'>")
warnings.filterwarnings("ignore", message= "module 'gi.repository.Gdk' has no attribute 'CursorType'")



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
        n_1 = 200
        n_2 = 186
        n_3 = 190
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

        cost_entropy = -np.mean(y * np.log(ypred + 1e-8))
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


    def initialize_adam(self):
        w1, w2, w3, w4, b1, b2, b3, b4 = self.get('w1', 'w2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4')

        vdw1 = np.zeros(w1.shape)
        vdb1 = np.zeros(b1.shape)
        sdw1 = np.zeros(w1.shape)
        sdb1 = np.zeros(b1.shape)

        vdw2 = np.zeros(w2.shape)
        vdb2 = np.zeros(b2.shape)
        sdw2 = np.zeros(w2.shape)
        sdb2 = np.zeros(b2.shape)

        vdw3 = np.zeros(w3.shape)
        vdb3 = np.zeros(b3.shape)
        sdw3 = np.zeros(w3.shape)
        sdb3 = np.zeros(b3.shape)

        vdw4 = np.zeros(w4.shape)
        vdb4 = np.zeros(b4.shape)
        sdw4 = np.zeros(w4.shape)
        sdb4 = np.zeros(b4.shape)

        self.put(vdw1=vdw1,vdw2=vdw2,vdw3=vdw3,vdw4=vdw4,vdb1=vdb1,vdb2=vdb2,vdb3=vdb3,vdb4=vdb4)
        self.put(sdw1=sdw1,sdw2=sdw2,sdw3=sdw3,sdw4=sdw4,sdb1=sdb1,sdb2=sdb2,sdb3=sdb3,sdb4=sdb4)

    def update_parameters_with_adam(self, t, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, rate = 0.05):
        """
        t  -- adam counter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates
        v -- Adam variable, moving average of the first gradient
        s -- Adam variable, moving average of the squared gradient
        """

        w1,w2,w3,w4,dw1,dw2,dw3,dw4,b1,b2,b3,b4,db1,db2,db3,db4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4','b1','b2','b3','b4','db1','db2','db3','db4')
        vdw1,vdw2,vdw3,vdw4,vdb1,vdb2,vdb3,vdb4, = self.get('vdw1','vdw2','vdw3','vdw4','vdb1','vdb2','vdb3','vdb4')
        sdw1,sdw2,sdw3,sdw4,sdb1,sdb2,sdb3,sdb4 = self.get('sdw1','sdw2','sdw3','sdw4','sdb1','sdb2','sdb3','sdb4')

        #moving average of the gradients
        vdw1 = (beta1*vdw1) + ((1-beta1)*dw1)
        vdb1 = (beta1*vdb1) + ((1-beta1)*db1)

        vdw2 = (beta1*vdw2) + ((1-beta1)*dw2)
        vdb2 = (beta1*vdb2) + ((1-beta1)*db2)

        vdw3 = (beta1*vdw3) + ((1-beta1)*dw3)
        vdb3 = (beta1*vdb3) + ((1-beta1)*db3)

        vdw4 = (beta1*vdw4) + ((1-beta1)*dw4)
        vdb4 = (beta1*vdb4) + ((1-beta1)*db4)

        #compute bias-corrected first memnt estimate
        vc_dw1 = vdw1 / (1-np.power(beta1,t))
        vc_db1 = vdb1 / (1-np.power(beta1,t))

        vc_dw2 = vdw2 / (1-np.power(beta1,t))
        vc_db2 = vdb2 / (1-np.power(beta1,t))

        vc_dw3 = vdw3 / (1-np.power(beta1,t))
        vc_db3 = vdb3 / (1-np.power(beta1,t))

        vc_dw4 = vdw4 / (1-np.power(beta1,t))
        vc_db4 = vdb4 / (1-np.power(beta1,t))

        #moving average of the squared gradients
        sdw1 = (beta2*sdw1) + ((1-beta2)*(dw1*dw1))
        sdb1 = (beta2*sdb1) + ((1-beta2)*(db1*db1))

        sdw2 = (beta2*sdw2) + ((1-beta2)*(dw2*dw2))
        sdb2 = (beta2*sdb2) + ((1-beta2)*(db2*db2))

        sdw3 = (beta2*sdw3) + ((1-beta2)*(dw3*dw3))
        sdb3 = (beta2*sdb3) + ((1-beta2)*(db3*db3))

        sdw4 = (beta2*sdw4) + ((1-beta2)*(dw4*dw4))
        sdb4 = (beta2*sdb4) + ((1-beta2)*(db4*db4))

        #compute bias-corrected first memnt estimate
        sc_dw1 = sdw1 / (1-np.power(beta2,t))
        sc_db1 = sdb1 / (1-np.power(beta2,t))

        sc_dw2 = sdw2 / (1-np.power(beta2,t))
        sc_db2 = sdb2 / (1-np.power(beta2,t))

        sc_dw3 = sdw3 / (1-np.power(beta2,t))
        sc_db3 = sdb3 / (1-np.power(beta2,t))

        sc_dw4 = sdw4 / (1-np.power(beta2,t))
        sc_db4 = sdb4 / (1-np.power(beta2,t))

        #update parameters
        w1 -= rate * vc_dw1 / np.sqrt(sc_dw1+epsilon)
        b1 -= rate * vc_db1 / np.sqrt(sc_db1+epsilon)

        w2 -= rate * vc_dw2 / np.sqrt(sc_dw2+epsilon)
        b2 -= rate * vc_db2 / np.sqrt(sc_db2+epsilon)

        w3 -= rate * vc_dw3 / np.sqrt(sc_dw3+epsilon)
        b3 -= rate * vc_db3 / np.sqrt(sc_db3+epsilon)

        w4 -= rate * vc_dw4 / np.sqrt(sc_dw4+epsilon)
        b4 -= rate * vc_db4 / np.sqrt(sc_db4+epsilon)

        self.put(w1 = w1, w2 = w2, w3 = w3, w4 = w4, b1 = b1, b2 = b2, b3 = b3, b4 = b4)
        self.put(vdw1=vdw1,vdw2=vdw2,vdw3=vdw3,vdw4=vdw4,vdb1=vdb1,vdb2=vdb2,vdb3=vdb3,vdb4=vdb4)
        self.put(sdw1=sdw1,sdw2=sdw2,sdw3=sdw3,sdw4=sdw4,sdb1=sdb1,sdb2=sdb2,sdb3=sdb3,sdb4=sdb4)


    # def update(self, rate = 0.05):
    #     w1, w2, w3, w4, dw1, dw2, dw3, dw4, b1, b2, b3, b4, db1, db2, db3, db4 = self.get('w1', 'w2', 'w3', 'w4','dw1', 'dw2', 'dw3', 'dw4', 'b1', 'b2', 'b3', 'b4', 'db1', 'db2', 'db3', 'db4')
    #
    #     w1 -= rate * dw1
    #     b1 -= rate * db1
    #
    #     w2 -= rate * dw2
    #     b2 -= rate * db2
    #
    #     w3 -= rate * dw3
    #     b3 -= rate * db3
    #
    #     w4 -= rate * dw4
    #     b4 -= rate * db4
    #
    #     self.put(w1 = w1, w2 = w2, w3 = w3, w4 = w4, b1 = b1, b2 = b2, b3 = b3, b4 = b4)

    def put(self, **kwargs):
        for key, value in kwargs.items():
            self.cache[key] = value


    def get(self, *args):
        x = tuple(map(lambda x: self.cache[x], args))
        return x

    def random_mini_batches(self, batchsize = 32, seed = 0):

        x, y = self.get('x', 'y')
        np.random.seed(seed)

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
    t = 0
    seed = 10
    costs = []
    epochs = 5
    n = network(xtrain, ytrain)
    n.initialize_adam()


    for epoch in progressbar(range(epochs)):

        seed += 1
        minibatches = n.random_mini_batches(seed)
        # cost_total = 0

        for minibatch in minibatches:
            (minibatch_x, minibatch_y) = minibatch
            ypred = n.forward(minibatch_x)
            costs.append(n.cost(ypred, minibatch_y))
            # cost_total += n.cost(ypred, minibatch_y)
            n.backward(minibatch_x, minibatch_y)
            # n.update()
            t += 1
            n.update_parameters_with_adam(t)

    # #append cost every 100 epoch
    # cost_avg = cost_total / m
    # if epoch%100 == 0:
    #     costs.append(cost_avg)


    print("")

    #accuracy for the training network
    ytrain_pred = n.forward(xtrain)
    ytrain_pred = np.argmax(ytrain_pred.T, axis = 1).reshape(60000, 1)
    y_train = np.argmax(ytrain.T, axis = 1).reshape(60000, 1)
    count = accuracy(ytrain_pred, y_train)
    print("Model accuracy - training network: {}%".format((count*100)/y_train.shape[0]))

    #accuracy for test network
    ytest_pred = n.forward(xtest)
    ytest_pred = np.argmax(ytest_pred.T, axis = 1).reshape(10000,1)
    y_test = np.argmax(ytest.T, axis = 1).reshape(10000,1)
    count = accuracy(ytest_pred, y_test)
    print("Model accuracy - test network: {}%".format((count*100)/y_test.shape[0]))


    alert(text='Training Complete!', title='ALERT', button='OK')


    # To plot Cost vs Epochs
    plt.title('Learning rate = 0.05')
    plt.xlabel('epochs (per 100)')
    plt.ylabel('cost')
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
