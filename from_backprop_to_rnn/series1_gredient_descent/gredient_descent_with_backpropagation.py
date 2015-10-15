# encoding=utf-8

""" http://outlace.com/Beginner-Tutorial-Backpropagation/ """
""" 吴立德讲座 """

""" part I. vanilla (normal) gredient descent """
def f_derivative(x):
    return 4 * x**3 - 9 * x**2

def min():
    x_old = 0
    x_new = 4
    gamma = 0.01  # learning rate
    precision = 0.00001

    i = 1
    while abs(x_new - x_old) > precision:
        if i % 50 == 0:
            print 'iteration: %d' % i
        i += 1
        x_old = x_new
        x_new = x_old - gamma * f_derivative(x_old)

    print("Local min occurs at %f after iteration: %d" % (x_new, i))


""" part II. simple 1-d regression (forward-only neural network) with MSE cost, 
    no bias item and only single weight parameter to predict """
import numpy as np

def sigmoid(x):
    return np.matrix(1.0 / (1.0 + np.exp(-x)))

def run(X, weight):
    return sigmoid(X * weight)   # 5*1 * 1*1 => 5*1 matrix result here

# This func is acturally no use in weight prediction
def cost(X, y, weight):
    nn_out = run(X, weight)
    m = X.sharp[0] # which is number of the samples
    return np.sum((1/m) * np.square(nn_out - y))

def d1_regression():
    X = np.matrix('0.1;0.25;0.5;0.75;1')
    y = np.matrix('0;0;0;0;0')
    weight = np.matrix('5.0')  # randomly initialized to 5
    alpha = 0.5 # learning rate
    epochs = 2000 # max iterations
    for i in range(epochs):
        cost_derivative = np.sum(np.multiply((run(X, weight) - y), np.multiply(run(X, weight), (1 - run(X, weight)))))
        # 我觉得原文有误，因为最后还要把 w*x 对 w 求导，故此，最后要再乘以一个 x
        # 不过这个例子中，显然 w 越趋向负无穷越好，故此，乘上 x 反而降低了速度
        # 也是因为趋向负无穷，故此，算法不会自动收敛，而且越跑越大，于是，原文采用了 epochs=2000，强制结束
        #cost_derivative = np.sum(np.multiply(X, np.multiply((run(X, weight) - y), np.multiply(run(X, weight), (1 - run(X, weight))))))
        weight = weight - alpha * cost_derivative
    print "Final weight: %s" % weight


""" part III. Backpropagation, bias provided to input and 1 hidden layer, to predict XOR
    2 inputs (plus a bias) + 3 hidden units (plus a bias) --> 1 ouput 
    cost func: logistic hypotesis cost func (also known as cross-entropy)"""

""" Original post has something wrong, or maybe another kind of implement
    I will implement it on my own as below """

# cross_ent_cost = -1 * y * log(a) - (1 - y) * log(1 - a)
# y could be regarded as a const in here, deriv is with respect to a
def cross_ent_deriv(a, y):
    return (a - y) / (a * (1 - a))

# regardless of x
def sigmoid_deriv(a):
    return np.multiply(a, (1 - a))

# see the article for the nn topology
def backprop():
    X = np.matrix([ [0,0], [0,1], [1,0], [1,1] ]) # 4*2， 4 samples which are 2-dimensions
    y = np.matrix([ [0, 1, 1, 0] ]).T  # 4 * 1,  4 results which represent XOR proc
    numIn, numHid, numOut = 2, 3, 1  # In & Hid will be append a bias item (equals 1 later)
    # init weights
    theta1 = (0.5 * np.sqrt(6 / (numIn + numHid)) * np.random.randn(numIn+1, numHid))  # add a weight for input bias
    theta2 = (0.5 * np.sqrt(6 / (numHid + numOut)) * np.random.randn(numHid+1, numOut)) # add weight for hiden bias
    alpha = 0.1 # learning rate
    epochs = 10000 # iterations
    m = X.shape[0] # number of samples

    ita = 1
    for j in range(epochs):
        if ita % 500 == 0:
            print "Iteration: %d" % ita
        ita += 1
        for x in range(m):
            # forward process
            # X[x,:] & np.ones((1,1)) are both matrix
            a1 = np.matrix( np.concatenate((X[x,:], np.ones((1,1))), axis=1) ) # append input bias, a1 like matrix([[0., 0., 1.]])
            z2 = np.matrix(a1.dot(theta1))  # like matrix([[ 0.11218412,  0.65383585,  0.08888726]])
            a2 = np.matrix( np.concatenate((sigmoid(z2), np.ones((1,1))), axis=1) )  # append hidden bias
            z3 = np.matrix(a2.dot(theta2))
            a3 = np.matrix(sigmoid(z3))  # final output

            # now back propagation
            delta2 = cross_ent_deriv(a3, y[x]).dot(sigmoid_deriv(a3))  # cross_ent_deriv(a3, y[x]) 本例中只有一个值 [[delta]]，本质是个对角矩阵
            theta2_grad = np.multiply(delta2, a2)
            theta2 -= np.multiply(alpha, theta2_grad.T)
            delta1 = np.multiply(delta2.dot(theta2.T), sigmoid_deriv(a2)).T[:-1,:]  # remove bias delta
            theta1_grad = delta1.dot(a1)
            theta1 -= np.multiply(alpha, theta1_grad.T)

    print "Results:\n"
    a1 = np.matrix( np.concatenate((X, np.ones((4,1))), axis=1) )
    z2 = np.matrix(a1.dot(theta1))
    a2 = np.matrix( np.concatenate((sigmoid(z2), np.ones((4,1))), axis=1) )
    z3 = np.matrix(a2.dot(theta2))
    a3 = np.matrix(sigmoid(z3))
    print(a3)
    
    """
    My result is like:
    [[  4.87814823e-04]
    [  9.97400642e-01]
    [  9.97619068e-01]
    [  4.11507398e-03]]

    much better than origin post' result
    [[ 0.01205117]
    [ 0.9825991 ]
    [ 0.98941642]
    [ 0.02203315]]
    """


if __name__ == '__main__':
#    min()
#    d1_regression()
    backprop()


