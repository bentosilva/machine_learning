# encoding=utf-8
"""
    http://www.deeplearning.net/tutorial/lenet.html#lenet

    This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit
import cPickle

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logreg_softmax import LogReg, MnistLoader
from mlp import MLP, HiddenLayer


# 一层卷积 + 一层MaxPooling 的组合
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4，四维矩阵
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4，四个维度的数字组成的 list
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
                              # of filters 也就是卷积输出层的 feature maps 数
                              这个参数其实就是卷积层的 W 的维度

        :type image_shape: tuple or list of length 4，四个维度的数字组成的 list
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
                             这个参数实际就是卷积输入层的 X 的维度

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        效果就是pooling输入层的每 #rows X #cols 个点选出一个最大值，组成 pooling输出层
        比如，输入层为 (3,2,6,6)，而 factor 为 (2,2)，那么 max_pooling 得到 (3,2,3,3)

        该类实际上定义了一个卷积层，加一个 pooling 层 
        输入层 aka 卷积输入层 ---> 卷积输出层 aka Hidden or pooling 输入层 ---> 输出层 aka pooling 输出层
        """

        assert image_shape[1] == filter_shape[1] # # of input feature maps，卷积输入层 feature maps 个数
        self.input = input

        # there are "num input feature maps * filter height * filter width" inputs to each hidden unit
        # 看到是对每个 hidden unit 也就是卷积输出层的每个 feature map 计算的，
        # 故此第一维度也就是卷积输出层 feature map 数是不需要投入计算的
        # 用于初始化 卷积层 W 参数 (Notes: pooling 层并不需要 W 参数)
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # 输入层的每个点，都会卷积到卷积层的每一个 feature map 上，每个 map上会影响到 filter width * filter height 个点
        # 而卷积输出层也即 pooling 输入层，会通过 pooling 缩小 poolsize (factor) 倍的尺寸
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights，可以看 MLP 一章的数学定义，用于初始化 W，得到 4 维矩阵
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        # 每个卷积输出层的 feature map 上固定一个 bias 不变，不管是那个输入层的 feature map 上过来的；
        # 一维，其值和 filter_shape[0] 一致，即 filter_shape[0] == len(b)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        # 通过调用函数，隐去了如何做卷积的过程
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        # 通过调用函数，隐去了如何做pooling的过程，如何利用 poolsize 做最大值比较，并返回正确维度的结果
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # 做完卷积 + pooling 之后，再加上 bias，然后再调用激活函数 tanh
        # 卷积之后，貌似原来的第一维跑到了第二维，于是 b 进行了 dimshuffle 得到 1 * len(b) * 1 * 1 的 4 维矩阵
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


# 里面写死了一些用于 Mnist 数据集的参数
# 故此，这并不是一个通用的 LeNet，是为 Mnist 定制的
# 通用的需要进一步抽象，从学习的角度暂时不这么做了，先这样
class MnistLeNet(object):
    def __init__(self, rng, input, nkerns=[20, 50], batch_size=500):
        """ Demonstrates lenet on MNIST dataset
        :type nkerns: list of ints
        :param nkerns: number of kernels(feature maps) on each layer 
        """
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        # Construct the first convolutional pooling layer:                  # 第一层 Conv + Pooling
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)  # 5 * 5 的 filter
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)        # 2 * 2 的 max pooling
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12) # nkerns[0] 个 feature map
        layer0 = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
 
        # Construct the second convolutional pooling layer                  # 第二层
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)     # 5*5 的 filter
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)            # 2 * 2 的 pooling
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)   # nkerns[1] 个 feature map
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
 
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)  # 输出 2 维，第一维就是 batch_size，第二维是 feature_map数 * 输出height * 输出 width
 
        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,  # 500 个隐藏节点
            activation=T.tanh
        )
 
        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogReg(input=layer2.output, n_in=500, n_out=10)

        self.input = input
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.logreg = layer3
 
    # the cost we minimize during training is the NLL of the model
    def negative_log_likelihood(self, y):
        return self.logreg.negative_log_likelihood(y)
 
    def errors(self, y):
        return self.logreg.errors(y)

    
class LeNetTrainer(object):
    def __init__(self, loaded_loader, learning_rate=0.1, n_epochs=200,
                 nkerns=[20, 50], batch_size=500):
        """ Demonstrates lenet on MNIST dataset
 
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
 
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
 
        :type loaded_loader: MnistLoader
        :param loaded_loader: a loader that already loaded with data
 
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        self.dl = loaded_loader
        self.lr = learning_rate
        self.ep = n_epochs
        self.nkerns = nkerns
        self.bs = batch_size

    def train(self, patience=10000, patience_increase=2, improvement_threshold=0.995):
        rng = np.random.RandomState(23455)
 
        train_set_x, train_set_y = self.dl.train_set_x, self.dl.train_set_y
        valid_set_x, valid_set_y = self.dl.valid_set_x, self.dl.valid_set_y
        test_set_x, test_set_y = self.dl.test_set_x, self.dl.test_set_y
 
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.bs
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / self.bs
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / self.bs
 
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
 
        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
 
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
 
        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        # 整个LeNet 的最初输入层，feature map = 1
        layer0_input = x.reshape((self.bs, 1, 28, 28)) 
        classifier = MnistLeNet(rng=rng, input=layer0_input, nkerns=self.nkerns, batch_size=self.bs)
 
        # the cost we minimize during training is the NLL of the model
        cost = classifier.negative_log_likelihood(y)
 
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            classifier.errors(y),
            givens={
                x: test_set_x[index * self.bs: (index + 1) * self.bs],
                y: test_set_y[index * self.bs: (index + 1) * self.bs]
            }
        )
 
        validate_model = theano.function(
            [index],
            classifier.errors(y),
            givens={
                x: valid_set_x[index * self.bs: (index + 1) * self.bs],
                y: valid_set_y[index * self.bs: (index + 1) * self.bs]
            }
        )
 
        # create a list of all model parameters to be fit by gradient descent
        params = classifier.params
 
        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)
 
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - self.lr * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]
 
        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * self.bs: (index + 1) * self.bs],
                y: train_set_y[index * self.bs: (index + 1) * self.bs]
            }
        )
        # end-snippet-1
 
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
 
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
 
        epoch = 0
        done_looping = False
 
        while (epoch < self.ep) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
 
                iter = (epoch - 1) * n_train_batches + minibatch_index
 
                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)
 
                if (iter + 1) % validation_frequency == 0:
 
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
 
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
 
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                            print('    [!!] patience improve to %d' % patience)
 
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
 
                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in xrange(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                        with open('best_model.pkl', 'w') as f:
                            cPickle.dump(classifier, f)
 
                if patience <= iter:
                    done_looping = True
                    break
 
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))


class Predicter(object):
    def predict(self, loader):
        classifier = cPickle.load(open('best_model.pkl'))
        predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.logreg.y_pred
        )
        # 必须要使用 get_value()， convert a varaible to an array object
        # >>> loader.test_set_x
        # <TensorType(float64, matrix)>
        # >>> loader.test_set_y
        # Elemwise{Cast{int32}}.0
        predicted_values = predict_model(loader.test_set_x.get_value()[:500].reshape((500, 1, 28, 28)))  # MnistLeNet classifier 使用了 500 作为 batch_size，故此我这里也只能写死 500
        print ('Predicted values for the first 10 examples in test set:')
        print predicted_values[:10]
        print ('Original values:')
        # 这里要用 eval
        # only shared variables actually contain a value, and have a get_value() method
        # test_set_y is a symbolic expression transforming a shared variable stored as a float vector 
        # on the GPU into an int32 vector; use eval() to compute a value from that symbolic expression, 
        # starting from the current value of the shared variable
        print loader.test_set_y.eval()[:10]


if __name__ == '__main__':
    loader = MnistLoader()
    loader.load_data('mnist.pkl.gz')
#    trainer = LeNetTrainer(loader)
#    trainer.train()
    predicter = Predicter()
    predicter.predict(loader)


