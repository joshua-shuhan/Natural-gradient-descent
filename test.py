import numpy as np
import pandas as pd
from scipy import linalg

data = pd.read_csv('IRIS.csv')
data.loc[data['species'] == 'Iris-setosa', 'species'] = 0
data.loc[data['species'] == 'Iris-versicolor', 'species'] = 1
data.loc[data['species'] == 'Iris-virginica', 'species'] = 2

data = data.iloc[np.random.permutation(len(data))]  # Shuffle the data
x_train = data.iloc[0:100, 0:4].to_numpy().astype(np.double)
y_train = data.iloc[0:100, 4].to_numpy().astype(int)

x_test = data.iloc[100:, 0:4].to_numpy().astype(np.double)
y_test = data.iloc[100:, 4].to_numpy().astype(int)


def compute_lambdaF(softmaxi, input, i):
    '''

    :param softmax: The softmax function, with the shape batchsize*output_dim
    :param input: The input x, with the shape batch_size * 4
    :return: The gradient of the ith sample softmax function w.r.t w, with the shape [vec_dim,3]
    '''
    vec_dim = 12
    lambdaFt = np.zeros([vec_dim, softmaxi.shape[0]])
    for j in range(softmaxi.shape[0]):  # 每一个sample有三个prob
        tempM = np.zeros([4, 3])
        for k in range(softmaxi.shape[0]):  # 在求梯度时，要有三项
            tempx = np.zeros([4, 3])
            tempx[:, k] = input[i, :]
            if k == j:
                tempM += softmaxi[j] * (1 - softmaxi[j]) * tempx
            else:
                tempM += -softmaxi[k] * softmaxi[j] * tempx

        lambdaFt[:, j] = tempM.reshape([vec_dim])

    return lambdaFt


def compute_fisher_inverse_adp(grad_input, input, softmax, epsilont=0.01):
    vec_dim = 12
    fisher_init = np.outer(grad_input[0, :], input[0, :]).reshape((vec_dim, 1)) @ np.outer(grad_input[0, :],
                                                                                           input[0, :]).reshape(
        (vec_dim, 1)).T
    fisher_inv = linalg.pinv(fisher_init)
    for i in range(150)[1:]:
        softmaxi = softmax[i, :]
        lambdaF = compute_lambdaF(softmaxi, input, i)
        fisher_inv = (1 + epsilont) * fisher_inv - epsilont * (fisher_inv @ lambdaF) @ (fisher_inv @ lambdaF).T
    fisher4w = fisher_inv @ (input.T @ grad_input).reshape((vec_dim, 1))
    fisher4w = fisher4w.reshape(4, 3) / (fisher4w.max() / 2)

    fisher_init = np.outer(grad_input[0,:],grad_input[0,:])
    return fisher4w

def compute_fisher_inverse(grad_input, input):

    assert grad_input.shape[0] == 150
    vec_dim = 12
    fisher = np.zeros([vec_dim, vec_dim])
    # Sum up vectorized gradient over a batch
    for i in range(150):
        fisher += np.outer(grad_input[i, :], input[i, :]).reshape((vec_dim, 1)) @ np.outer(grad_input[i, :],
                                                                                               input[i, :]).reshape(
                (vec_dim, 1)).T
        # Take average value
    fisher = fisher / 150
    assert fisher.all() == fisher.T.all()
    fisher_inv = linalg.pinv(fisher)
    fisher4w = fisher_inv @ (input.T @ grad_input).reshape((vec_dim, 1))
    fisher4w = fisher4w.reshape(4, 3) / (fisher4w.max() / 2)
    fisherb = np.zeros([3, 3])

    for i in range(150):
        fisherb += np.outer(grad_input[i, :], grad_input[i, :])
    fisherb = fisherb / 150
    assert fisherb.all() == fisherb.T.all()
        # print(fisherb)
    fisherb_inv = linalg.pinv(fisherb)
    fisher4b = fisherb_inv @ (grad_input.sum(axis=0, keepdims=True).T)
    fisher4b = fisher4b.T

    return fisher4w, fisher4b

grad_input = np.random.rand(150,3)
input = np.random.rand(150,4)
softmax = np.random.rand(150,3)

a = compute_fisher_inverse_adp(grad_input,input,softmax)
b,c = compute_fisher_inverse(grad_input,input)

print(a)
print(b)