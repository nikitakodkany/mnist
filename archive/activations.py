import numpy as np

def sigmoid(z):
	return 1/ (1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1-sigmoid(z))

def relu(z):
	return np.maximum(z, 1)

def relu_prime(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

def softmax(z):
	maxs = np.max(z,axis= 0)

	shift = z - maxs
	exp = np.exp(shift)
	sums = np.sum(exp , axis = 0)
	return exp/sums

def softmax(z):
	exp_z = np.exp(z - np.max(z, axis=0))
	return exp_z / exp_z.sum(axis=0, keepdims=True)
