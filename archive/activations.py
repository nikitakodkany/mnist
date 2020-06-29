import numpy as np
import warnings
warnings.filterwarnings("error")

def sigmoid(z):
	return 1/ (1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1-sigmoid(z))

def leakyrelu(z):
	z = np.where(z > 0, z, z * 0.01)
	return z


def leakyrelu_prime(z):
	z = np.where(z > 0, 1.0, 0.0)
	return z

# def softmax(z):
# 	maxs = np.max(z,axis= 0)
#
# 	shift = z - maxs
# 	exp = np.exp(shift)
# 	sums = np.sum(exp , axis = 0)
# 	return exp/sums

def softmax(z):
	exp_z = np.exp(z - np.max(z, axis=0))
	return exp_z / exp_z.sum(axis=0, keepdims=True)
