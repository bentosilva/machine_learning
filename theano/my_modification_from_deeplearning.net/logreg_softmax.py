# encoding=utf-8

import cPickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T

class LogReg(object):
    # n_in 为输入 X 的维度， n_out 为输出 y 的维度，使用 softmax，而不是 binary
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        # 1*n_in  X  n_in*n_out  +  1*n_out  ==>  1*n_out 个标签上的值；n个样本则为 n * n_out
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # 在第二维也就是沿行向量来取最大值
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # 需要训练的参数
        self.params = [self.W, self.b]
        # keep track of model input
        self.input = input

    # 损失函数，设为 y_pred 为真实标签 y 的几率的负值；也就是说几率越大，损失越小，越好
    # param y: corresponds to a vector that gives for each example the correct label
    def negative_log_likelihood(self, y):
        # T.arange(y.shape[0]) 是样本数为界的 [0, 1, 2, .. n-1]，即这个 minibatch 中共 n 个样本
        # T.log(self.p_y_given_x) 是一个 Log-Probabilities 矩阵，记为 LP
        # 每行是一个输入样本的结果数组，每列是属于这个列对应的标签的几率再取 log，故此 n*n_out 维
        # 于是，LP[T.arange(y.shape[0]), y] 为 [LP[0,y[0]], LP[1,y[1]], .. LP[n-1,y[n-1]]]
        # y 为 n*1 的向量，每一行为某一个样本的标签值，故此 y[i] >= 0 && y[i] < n_out
        # 故此， LP[i, y[i]] 就对应着序号为 i 的样本，取其真正标签 y[i] 的几率
        # 最后，All in all, 为对 minibatch 中样本取其自己标签的几率做平均，minibatch 的平均 log-likelihood
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # 同样，y 为向量，每一行为minibatch中每个样本对应的y标签
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should has the same shape as self.y_pred', ('y',y.type,'y_pred',self.y_pred.type))
        # 看到，本实现标签值为 0~n_out，都是 int
        if y.dtype.startswith('int'):
            # neq == not eq, 返回一个向量，每个分量表示对应的 minibatch 中的样本是否分类错误，错误标 1
            # 对于一个 minibatch 的 y_pred & y，那么取均值
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MnistLoader(object):
    def load_data(self, dataset):
        # download if not present
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            new_path = os.path.join(
                os.path.split(__file__)[0],
                '..',
                'data',
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path
        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading from %s' % origin
            urllib.urlretrieve(origin, dataset)
        print '... loading data'

        f = gzip.open(dataset, 'rb')
        # train_set, valid_set, test_set format: tuple(input, target)
        # input is an numpy.ndarray of 2 dimensions (a matrix), 每行一个样本，每个样本 28*28 列
        # target 为列向量，每行为一个样本对应的标签值
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        self.test_set_x, self.test_set_y = self.shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = self.shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = self.shared_dataset(train_set)

    # The reason we store dataset in shared variables is to allow Theano to copy it into the GPU memory
    # Since copying data into the GPU is slow, copying a minibatch everytime is needed would lead to 
    # a large decrease in performance
    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


class MnistTrainer(object):
    def __init__(self, learning_rate=0.13, n_epoch=1000, dataset='mnist.pkl.gz', batch_size=600, n_in=28*28, n_out=10):
        self.dl = MnistLoader()
        self.dl.load_data(dataset)
        self.lr = learning_rate
        self.ep = n_epoch
        self.bs = batch_size
        self.n_in = n_in
        self.n_out = n_out

    # stochastic gradient descent 
    # 总更新参数次数，不超过 patience
    # improvement_threshold 看是否在验证集上的结果很好，如果足够好，那么增大耐心
    def sgd_optimize(self, patience=5000, patience_inc=2, improvement_threshold=0.995):
        n_train_batches = self.dl.train_set_x.get_value(borrow=True).shape[0] / self.bs 
        n_valid_batches = self.dl.valid_set_x.get_value(borrow=True).shape[0] / self.bs 
        n_test_batches = self.dl.test_set_x.get_value(borrow=True).shape[0] / self.bs 

        print '... building the model'
        index = T.lscalar()  # index of a minibatch
        x = T.matrix('x')   # #sample * n_in
        y = T.ivector('y')  # #samele * 1
        classifier = LogReg(input=x, n_in=self.n_in, n_out=self.n_out)
        cost = classifier.negative_log_likelihood(y)
        # 每个 minibatch 做一次参数更新
        test_model = theano.function(
                inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: self.dl.test_set_x[index * self.bs: (index + 1) * self.bs],
                    y: self.dl.test_set_y[index * self.bs: (index + 1) * self.bs]
                }
        )
        valid_model = theano.function(
                inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: self.dl.valid_set_x[index * self.bs: (index + 1) * self.bs],
                    y: self.dl.valid_set_y[index * self.bs: (index + 1) * self.bs]
                }
        )
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
        updates = [(classifier.W, classifier.W - self.lr * g_W),
                   (classifier.b, classifier.b - self.lr * g_b)]
        train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: self.dl.train_set_x[index * self.bs: (index + 1) * self.bs],
                    y: self.dl.train_set_y[index * self.bs: (index + 1) * self.bs]
                }
        )

        print '... training'
        # 取 minibatch 总数和 patience/2 的最小值
        # 意味着这里，我们每一轮 epoch 也就是每一轮所有 minibatch 训练一遍之后，都要做一次验证
        # 即使耐心很小很小，小到小于 batch 个数，那么至少也会做一次validation (pationce/2)
        validation_frequency = min(n_train_batches, patience / 2)
        best_validation_loss = np.inf
        test_score = 0
        start_time = timeit.default_timer()

        done_looping = False        # checking early stopping
        epoch = 0
        # 对整个数据集迭代 epch 次
        while (epoch < self.ep) and (not done_looping):
            epoch += 1
            # 对数据集可能的 minibatch 数做循环
            for minibatch_index in xrange(n_train_batches):
                # 实际迭代次数
                iter = (epoch - 1) * n_train_batches + minibatch_index
                # 如果超过耐心，那么结束整个迭代
                if patience < iter:
                    done_looping = True
                    break
                # 训练一次 minibatch，更新一次结果，并返回 loss
                minibatch_avg_cost = train_model(minibatch_index)
                # 看是否需要计算 validation set 的错误率
                if (iter + 1) % validation_frequency == 0:
                    # 扫一圈 validation set 的全部 batch
                    validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]           
                    this_validation_loss = np.mean(validation_losses)
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    # 看是否最佳
                    if this_validation_loss < best_validation_loss:
                        # 如果足够好，那么增大 patience
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_inc)  # 足够轮数之后，不会再增加耐心，以防停不下来
                        # 记录最佳
                        best_validation_loss = this_validation_loss

                        # 在 test 集上计算最佳验证集结果的参数错误
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)
                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )
                        # 保存最佳
                        with open('best_model.pkl', 'w') as f:
                            cPickle.dump(classifier, f)

        end_time = timeit.default_timer()
        print (
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))


class Predicter(object):
    def predict(self):
        classifier = cPickle.load(open('best_model.pkl'))
        predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.y_pred
        )
        dataset = 'mnist.pkl.gz'
        loader = MnistLoader()
        loader.load_data(dataset)
        # 必须要使用 get_value()， convert a varaible to an array object
        # >>> loader.test_set_x
        # <TensorType(float64, matrix)>
        # >>> loader.test_set_y
        # Elemwise{Cast{int32}}.0
        predicted_values = predict_model(loader.test_set_x.get_value()[:10])
        print ('Predicted values for the first 10 examples in test set:')
        print predicted_values
        print ('Original values:')
        # 这里要用 eval
        # only shared variables actually contain a value, and have a get_value() method
        # test_set_y is a symbolic expression transforming a shared variable stored as a float vector 
        # on the GPU into an int32 vector; use eval() to compute a value from that symbolic expression, 
        # starting from the current value of the shared variable
        print loader.test_set_y.eval()[:10]


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.sgd_optimize()
    predicter = Predicter()
    predicter.predict()


