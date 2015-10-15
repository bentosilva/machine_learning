#encoding=utf-8
import numpy as np
import theano
import theano.tensor as T

""" multiple layer perceptron in theano 
http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
W: np.ndarray
b: np.ndarray
activation: function which will be a theano.tensor.=elemwise.Elemwise for layer output
x: theano.tensor.var.TensorVariable
y: theano.tensor.var.TensorVariable
"""
class Layer(object):
    def __init__(self, W_init, b_init, activation):
        """
        activation func will be a theano.tensor.elemwise.Elemwise for layer output
        """
        n_output, n_input = W_init.shape
        assert b_init.shape == (n_output, )
        # all parameters should be shared variables.
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
            name='W', borrow=True) # use W_init as internal buffer, so not deepcopy
        self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX),
            name='b', borrow=True, broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))


class MLP(object):
    def __init__(self, W_init, b_init, activation):
        # for multiple layers
        assert len(W_init) == len(b_init) == len(activation)
        self.layers = []
        for W, b, activation in zip(W_init, b_init, activation):
            self.layers.append(Layer(W, b, activation))
        self.params = []
        for layer in self.layers:
            self.params += layer.params   # concat as a flat list

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)


# When doing gradient descent on neural nets, it's very common to use momentum, 
# which is simply a leaky integrator on the parameter update
# In Theano, we store the previous parameter update as a shared variable so that its value is preserved across iterations.
# we not only update the parameters, but we also update the previous parameter update shared variable.
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    """
    cost: theano.tensor.var.TensorVariable
    params: list of theano.tensor.var.TensorVariable
    learning_rate: float
    momentum: float, should be at least 0 (standard gradient descent) and less then 1
    return: list of updates, one for each parameters
    """
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        # param_update shared variable keeps track of param's update across iterations, init it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate * param_update))
        # no need to do backpropagation to compute updates, just use T.grad !!
        updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
    return updates


def mlp_train(X, y):
    layer_size = [X.shape[0], X.shape[0] * 2, 1]  # input 2, hidden 4, output 1
    W_init = []
    b_init = []
    activations = []
    for n_input, n_output in zip(layer_size[:-1], layer_size[1:]):
        W_init.append(np.random.randn(n_output, n_input))
        b_init.append(np.ones(n_output))
        activations.append(T.nnet.sigmoid)
    mlp = MLP(W_init, b_init, activations)

    mlp_input = T.matrix('mlp_input')
    mlp_target = T.vector('mlp_target')
    learning_rate = 0.01
    momentum = 0.9
    cost = mlp.squared_error(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_target], cost, 
        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

    iteration = 0
    max_iteration = 20
    while iteration < max_iteration:
        current_cost = train(X, y)
        current_output = mlp_output(X)
        accuracy = np.mean((current_output > 0.5) == y)
        print "Iteration: %d ......" % iteration
        print "cost: %f" % current_cost
        print "Accuracy: %f" % accuracy
        print ""
        iteration += 1


def create_gaussian_clusters():
    np.random.seed(0)
    N = 1000
    # generate N' 0 or 1, as label
    y = np.random.random_integers(0, 1, N)
    # mean of each cluster
    means = np.array([[-1, 1], [-1, 1]])
    # 2 * 2, and ranged by [1,2]  ( 1 + [0,1] )
    covariances = np.random.random_sample((2, 2)) + 1
    # 2 * N,  N samples, each of which is a 2*1 vector (2-d point)
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    return X, y


if __name__ == '__main__':
    X, y = create_gaussian_clusters()
    mlp_train(X, y)

