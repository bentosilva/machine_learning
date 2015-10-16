# encoding: utf-8

"""
    http://outlace.com/Simple-Genetic-Algorithm-Python-Addendum/
    这里实现一个只有前馈没有回溯的 2 层神经网络
"""
import numpy as np


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])  # the third element in each tuple is the bias value (1)
y = np.array([[0, 1, 1, 0]]).T
init_theta = 10 * (np.random.random((13, 1)) - 0.5)
# theta below is results from authos's last post about 13 lines of GA
theta = np.array([-10.27649069, -14.03, 10.45888659, 9.12, 14.87, -21.50294038, 1.85, -13.28,
                  -0.15360052, -11.21345025, 35.77912716, 11.05, -2.49589577])


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def runForward(X, theta):
    """
        2 inputs, with a bias, make it 3-d
        3 hidden units, so input -> hidden weight is 3X3
        3 hidden units, with a bias, make it 4-d
        1 output, so hidden -> output weight is 4X1

        X is input with bias, and it may be a input array, like 4 inputs
        so, X is #samplesX3 dimension matrix
    """
    theta1 = np.array(theta[:9]).reshape(3, 3)
    theta2 = np.array(theta[9:]).reshape(4, 1)
    h1 = sigmoid(np.dot(X, theta1))   # #samples X 3 * 3X3
    h1_bias = np.insert(h1, 3, [1] * len(X), axis=1)  # make it #samples X 4
    output = sigmoid(np.dot(h1_bias, theta2))  # #samples X 1
    return output


def costFunction(X, y, theta):
    outputs = np.array(runForward(X, theta))
    return np.sum(np.abs(y - outputs))


def demoRun():
    print("Random theta: \n%s\n" % (np.round(runForward(X, init_theta), 2)))
    print("Cost: %s\n" % (costFunction(X, y, init_theta)))
    print("Optimal theta: \n%s\n" % (np.round(runForward(X, theta), 2)))
    print("Cost: %s\n" % (costFunction(X, y, theta)))


if __name__ == '__main__':
    demoRun()
