#encoding=utf-8

""" 原 Post 的 backpropagation 算法，其实是错误的
    d3 不是 a3 - y ，而应该是 Cost函数对 z3 的求导，见吴立德视频 """

import numpy as np
from sigmoid import sigmoid

# y, a are both x*1, so the result will be 1*x * x*1 --> 1*1
def cross_ent_cost(y, a):
    return (-1 * y.T * np.log(a)) - (1 - y).T * np.log(1 - a)

def costRNN(thetaVec, *args):
    X = args[0]
    Y = args[1]
    numIn, numHid, numOut = 1, 4, 1
    theta1 = thetaVec[0:24].reshape(numIn + numHid + 1, numHid)
    theta2 = thetaVec[24:].reshape(numHid + 1, numOut)
    theta1_grad = np.zeros((numIn + numHid + 1, numHid))
    theta2_grad = np.zeros((numHid + 1, numOut))

    # keep track of the output from hidder layer
    hid_last = np.zeros((numHid, 1))
    m = X.shape[0]
    J = 0 # cost output
    results = np.zeros((m, 1))  # store the output of the network

    # find gredients:
    for j in range(m):  # for each sample
        y = Y[j]
        context = hid_last
        # forward
        x_context = np.concatenate((X[j], context))
        a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]'))))
        z2 = theta1.T * a1
        a2 = np.concatenate((sigmoid(z2), np.matrix('[1]')))
        hid_last = a2[0:-1, 0]
        z3 = theta2.T * a2
        a3 = sigmoid(z3)
        results[j] = a3
        # backward
        d3 = (a3 - y)
        d2 = np.multiply((theta2 * d3), np.multiply(a2, (1 - a2)))
        theta1_grad = theta1_grad + (d2[0:numHid, :] * a1.T).T
        theta2_grad = theta2_grad + (d3 * a2.T).T
    # calculate network cost
    for n in range(m):
        a3n = results[n].T
        yn = Y[n].T
        J = J + cross_ent_cost(yn, a3n)
    J = (1/m) * J
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1)  # unroll gradients
    return J, grad
