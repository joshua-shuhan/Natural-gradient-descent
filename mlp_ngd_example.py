# -*- coding: utf-8 -*-
"""
This is an implementation of natural gradient descent optimization on multilayer perceptron.
Change the boolean variable `usefisher` to see the effect of the algorithm.

@author: Shuhan Zheng

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg

np.random.seed(seed=0)


class Layer(object):

    def __init__(self, *args):
        pass

    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class Relu(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return np.maximum(0, input_)

    def backward(self, input_, grad_input):
        temp = input_ > 0
        dEdX = grad_input * temp
        return dEdX


class Linear(Layer):
    def __init__(self, batch_size, input_dim, output_dim, learning_rate, adaptive, usefisher):
        self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2 / (input_dim + output_dim))
        self.b = np.zeros([1, output_dim])
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input_):
        """
        Return a matrix after multiplication with weight self.w

        :param input_: the shape of input matrix should be batch_size*input_dim
        :return: an output matrix with shape batch_size*output_dim
        """
        output = input_ @ self.w + self.b

        return output

    def compute_fisher_inverse(self, grad_input, input_):
        """
        :param grad_input: with the shape of batch_size*output_dim(from the loss module,bp4NGD or dEdX from the last
        layer).
        :param input_: with the shape of batch_size*input_dim
        :return: the inverse of empirical fisher information matrix
        """
        assert grad_input.shape[0] == self.batch_size
        vec_dim = self.input_dim * self.output_dim
        fisher = np.zeros([vec_dim, vec_dim])
        # Sum up vectorized gradient over a batch
        for i in range(self.batch_size):
            fisher += np.outer(grad_input[i, :], input_[i, :]).reshape((vec_dim, 1)) @ np.outer(grad_input[i, :],
                                                                                                input_[i, :]).reshape(
                (vec_dim, 1)).T
        # Take average value
        fisher = fisher / self.batch_size
        # Test symmetric property
        assert fisher.all() == fisher.T.all()
        fisher_inv = linalg.pinv(fisher)
        # print(fisher_inv)
        fisher4w = fisher_inv @ (input_.T @ grad_input).reshape((vec_dim, 1))
        # Prevent fisher from large norm, which may cause the learning process to be unstable
        fisher4w = fisher4w.reshape(self.input_dim, self.output_dim) / (fisher4w.max())

        fisherb = np.zeros([self.output_dim, self.output_dim])
        for i in range(self.batch_size):
            fisherb += np.outer(grad_input[i, :], grad_input[i, :])
        fisherb = fisherb / self.batch_size
        assert fisherb.all() == fisherb.T.all()
        fisherb_inv = linalg.pinv(fisherb)
        fisher4b = fisherb_inv @ (grad_input.sum(axis=0, keepdims=True).T)
        fisher4b = fisher4b.T
        assert fisher4b.shape == self.b.shape
        return fisher4w, fisher4b

    def compute_lambdaf(self, softmaxi, input_, i):
        """
        :param softmax: The softmax function, with the shape batchsize*output_dim
        :param input_: The input x, with the shape batch_size * 4
        :return: The gradient of the ith sample softmax function w.r.t w, with the shape [vec_dim,3]
        """
        vec_dim = self.input_dim * self.output_dim
        lambdaft = np.zeros([vec_dim, softmaxi.shape[0]])
        for j in range(softmaxi.shape[0]):
            tempm = np.zeros([self.input_dim, self.output_dim])
            for k in range(softmaxi.shape[0]):
                tempx = np.zeros([self.input_dim, self.output_dim])
                tempx[:, k] = input_[i, :]
                if k == j:
                    tempm += softmaxi[j] * (1 - softmaxi[j]) * tempx
                else:
                    tempm += -softmaxi[k] * softmaxi[j] * tempx

            lambdaft[:, j] = tempm.reshape([vec_dim])

        return lambdaft

    def compute_fisher_inverse_adp(self, grad_input, input_, softmax, epsilont=0.01):
        """
        # TODO: Optimize complexity, implement fisher4b

        :param grad_input: batch_size*output_dim
        :param input_: batch_size*input_dim
        :param softmax: batch_size*output_dim
        :param epsilont: controls the speed of iterative
        :return:
        """
        vec_dim = self.input_dim * self.output_dim
        fisher_init = np.outer(grad_input[0, :], input[0, :]).reshape((vec_dim, 1)) @ np.outer(grad_input[0, :],
                                                                                               input_[0, :]).reshape(
            (vec_dim, 1)).T
        fisher_inv = linalg.pinv(fisher_init)
        for i in range(self.batch_size)[1:]:
            softmaxi = softmax[i, :]
            lambdaf = self.compute_lambdaf(softmaxi, input_, i)
            fisher_inv = (1 + epsilont) * fisher_inv - epsilont * (fisher_inv @ lambdaf) @ (fisher_inv @ lambdaf).T
        fisher4w = fisher_inv @ (input_.T @ grad_input).reshape((vec_dim, 1))
        fisher4w = fisher4w.reshape(self.input_dim, self.output_dim) / (fisher4w.max())

        return fisher4w

    def backward(self, input_, grad_input, softmax=None):
        """

        :param input: The shape is (batch_size*input_dim)
        :param grad_input:  The shape is (batch_size*output_dim)
        :param fisher:  Flag for using NGD or GD
        :param learning_rate:
        :return: return the grad_output dEdX to the previous layer (batch_size*input_dim)
        """

        # compute gradients; dE/dX
        dEdX = grad_input @ self.w.T  # (batch_size*output_dim) * (output_dim*input_dim)
        # print(dEdX.shape)

        # compute gradients of W and b;  dE/dW and dE/db
        dEdW = input_.T @ grad_input / self.batch_size  # (input_channel * batch_size) * (batch_size * output_channel).
        dEdb = grad_input.sum(axis=0,
                              keepdims=True) / self.batch_size  # (batch_size * output_channel) sum over batch_size

        # test
        assert dEdW.shape == self.w.shape
        assert dEdb.shape == self.b.shape

        # Update self.w and self.b
        if (usefisher == True) and (adaptive == False):
            fisher4w, fisher4b = self.compute_fisher_inverse(grad_input, input_)
            self.w = self.w - learning_rate * fisher4w
            self.b = self.b - learning_rate * fisher4b

        if (usefisher == True) and (adaptive == True):
            fisher4w = self.compute_fisher_inverse_adp(grad_input, input_, softmax)
            fisher4b = dEdb
            self.w = self.w - learning_rate * fisher4w
            self.b = self.b - learning_rate * fisher4b  #

        if usefisher == False:
            self.w = self.w - learning_rate * dEdW
            self.b = self.b - learning_rate * dEdb
        # print(self.w)

        return dEdX


class LossModule(Layer):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def compute_loss_f(self, input_, target):
        """
        :param input_: The shape of input must be batch_size*3
        :param target: The shape should be batch_size*1
        :return: Return a scalar loss value and a `f` vector with shape(batch_size*1)
        """
        assert input_.shape[0] == self.batch_size
        assert input_.shape[1] == 3  # There are 3 classes
        # One-hot encoding
        one_hot_target = np.zeros_like(input_)
        one_hot_target[np.arange(len(input_)), target] = 1

        # np.sum(a,axis=1,keepdims=True) This return a vector with shape (batch_size,1) but not (batch_size,)
        # input = input - input.max(axis=1,keepdims=True)  ###
        logits = np.exp(input_) / np.sum(np.exp(input_), axis=1, keepdims=True)

        # print(input_)
        f = np.sum(np.multiply(one_hot_target, logits), axis=1, keepdims=True)
        assert f.shape == (self.batch_size, 1)
        # Here we define loss as negative of log likelihood
        loss = -np.sum(np.log(f))
        return loss

    def backward(self, input_, target):
        """
        :param input_:  The shape of input must be batch_size*3
        :param target: The shape should be batch_size*1. The values of target are 0,1,2
        :return: The gradient dL/d(wx) with shape batch_size*3
        """
        # backward pass, compute dE/dZ  Z:batch_size*3
        one_hot_target = np.zeros_like(input_)
        one_hot_target[np.arange(len(input_)), target] = 1
        softmax = np.exp(input_) / np.exp(input_).sum(axis=1, keepdims=True)  # broadcasting. e.g. b_z*3 / b_z
        bp4gd = (-one_hot_target + softmax)
        return softmax, bp4gd


# Preprocessing data
# Change catagorical name to numeric value
data = pd.read_csv('IRIS.csv')
data.loc[data['species'] == 'Iris-setosa', 'species'] = 0
data.loc[data['species'] == 'Iris-versicolor', 'species'] = 1
data.loc[data['species'] == 'Iris-virginica', 'species'] = 2

data = data.iloc[np.random.permutation(len(data))]  # Shuffle the data
x_train = data.iloc[0:100, 0:4].to_numpy().astype(np.double)
y_train = data.iloc[0:100, 4].to_numpy().astype(int)

x_test = data.iloc[100:, 0:4].to_numpy().astype(np.double)
y_test = data.iloc[100:, 4].to_numpy().astype(int)

# The boolean variables are used to control applied algorithms
adaptive = False
usefisher = True
# usefisher = False # uncomment this command to see the effect of traditional gradient descent.

# Network setup
learning_rate = 0.001
batch_size = 100
net = []
net.append(Linear(batch_size, 4, 5, learning_rate, adaptive, usefisher))
net.append(Linear(batch_size, 5, 3, learning_rate, adaptive, usefisher))


# Define the computation flow function
def forward(network, X):
    activations = []
    input = X

    for layer_i in network:
        activations.append(layer_i.forward(input))
        input = activations[-1]

    return activations


def predict(network, x):
    logits = forward(network, x)[-1]
    return logits.argmax(axis=-1)  # Returns the indices of the maximum values along an axis


def train(network, x, y):
    # forward phase; get activations of each layer
    layer_activations = forward(network, x)
    layer_inputs = [x, ] + layer_activations
    logits = layer_activations[-1]

    # compute loss

    loss_fn = LossModule(batch_size)
    loss = loss_fn.compute_loss_f(logits, y)

    if adaptive == False:
        _, loss_grad_gd = loss_fn.backward(logits, y)
        loss_grad = loss_grad_gd

        # backward phase
        for layer_index in range(len(net))[::-1]:
            layer = net[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
        return loss

    if adaptive == True:
        softmax, loss_grad_gd = loss_fn.backward(logits, y)
        loss_grad = loss_grad_gd
        for layer_index in range(len(net))[::-1]:
            layer = net[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad, softmax)
        return loss


# Train session

train_log = []
val_log = []

for epoch in range(1000):
    loss = train(net, x_train, y_train)

    train_log.append(loss)
    val_log.append(np.mean(predict(net, x_test) == y_test))

# Display
mpl.style.use('seaborn')
plt.title('Training curve-GD', fontsize=20)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('L(W,b)', fontsize=20)
plt.semilogy(np.arange(1000), train_log)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

plt.plot(val_log)
plt.show()
print(val_log[-1])
