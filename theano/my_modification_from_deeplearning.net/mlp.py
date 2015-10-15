# encoding=utf-8

""" http://www.deeplearning.net/tutorial/mlp.html#mlp """
from logreg_softmax import LogReg, MnistLoader 
import numpy as np
import theano
import theano.tensor as T

import os
import sys
import timeit
import cPickle

# 输入到隐藏
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """ rng: numpy.random.RandomStat (short for random number generator)
            input: theano.tensor.dmatrix (若干个输入变量，每个输入都是一个向量)
            n_in：输入变量的维度
            n_out: 输出隐藏层的节点个数，相当于隐藏层的维度
            如果 W 未指定(为 None)，那么会根据公式来随机初始化
        """
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)  
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]
                
# 输入到输出，集成了 HidderLayer & LogReg 两层
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # 输入到隐藏
        self.hidden = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logreg = LogReg(input=self.hidden.output, n_in=n_hidden, n_out=n_out)
        # 计算 L1 & L2 Regularization
        self.L1 = (abs(self.hidden.W).sum() + abs(self.logreg.W).sum())
        self.L2_sqr = ((self.hidden.W ** 2).sum() + (self.logreg.W ** 2).sum())
        # 下面这两行，会导致 cPickle 失败，报警：TypeError: can't pickle instancemethod objects
        # 故此，改为定义两个成员函数，如下，这样可以正常的训练和做预测了
#        self.negative_log_likelihood = (self.logreg.negative_log_likelihood)
#        self.errors = self.logreg.errors
        self.params = self.hidden.params + self.logreg.params
        self.input = input 

    def negative_log_likelihood(self, y):
        return self.logreg.negative_log_likelihood(y)

    def errors(self, y):
        return self.logreg.errors(y)


class MLPTrainer(object):
    def __init__(self, loaded_loader, learning_rate=0.01, n_epoch=1000, batch_size=20, 
            n_in=28*28, n_hidden=500, n_out=10, L1_reg=0.00, L2_reg=0.0001):
        self.dl = loaded_loader
        self.lr = learning_rate
        self.ep = n_epoch
        self.bs = batch_size
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

    # stochastic gradient descent 
    # 总更新参数次数，不超过 patience
    # improvement_threshold 看是否在验证集上的结果很好，如果足够好，那么增大耐心
    def sgd_optimize(self, patience=10000, patience_inc=2, improvement_threshold=0.995):
        n_train_batches = self.dl.train_set_x.get_value(borrow=True).shape[0] / self.bs 
        n_valid_batches = self.dl.valid_set_x.get_value(borrow=True).shape[0] / self.bs 
        n_test_batches = self.dl.test_set_x.get_value(borrow=True).shape[0] / self.bs 

        print '... building the model'
        index = T.lscalar()  # index of a minibatch
        x = T.matrix('x')   # #sample * n_in
        y = T.ivector('y')  # #samele * 1

        rng = np.random.RandomState(1234)

        classifier = MLP(rng=rng, input=x, n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out)
        cost = (classifier.negative_log_likelihood(y) + self.L1_reg * classifier.L1 + self.L2_reg * classifier.L2_sqr)

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
        g_params = [T.grad(cost, param) for param in classifier.params]
        updates = [(param, param - self.lr * gparam)
            for param, gparam in zip(classifier.params, g_params)]
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
        best_iter = 0
        test_score = 0.
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
                            patience = max(patience, iter * patience_inc)  
                        # 记录最佳
                        best_validation_loss = this_validation_loss
                        best_iter = iter

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
                'obtained at iteration %i, with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))


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
    loader = MnistLoader()
    loader.load_data('mnist.pkl.gz')
#    trainer = MLPTrainer(loader)
#    trainer.sgd_optimize()
    predicter = Predicter()
    predicter.predict(loader)


