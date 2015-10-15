#encoding=utf-8

""" http://iamtrask.github.io/2015/07/28/dropout/ """

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_output_to_derivative(output):
    return output * (1- output)

def tuning_alpha_with_hidden():
    alphas = [0.001,0.01,0.1,1,10,100,1000]
    # 第四个参数都是1，故此其实没有意义
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],])
    y = np.array([[0],[1],[1],[0],])  # 把X的最后一维去掉后，我们发现其实就是一个 XOR 
    """ dropout 参数 """
    dropout_percent, do_dropout = (0.2, True)
    for alpha in alphas:
        print "\nTraining with alpha: " + str(alpha)
        np.random.seed(1)
        synapse_0 = 2*np.random.random((3,4)) - 1 # 3*4, no bias field obviously
        synapse_1 = 2*np.random.random((4,1)) - 1 # 4*1, so  4*3 (X) * 3*4 * 4*1 => 4*1 (y), 4 hidden units
        for j in xrange(60001):
            # Forward
            layer_0 = X  # 4*3
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))  # 4*4，第一个4 是4个样本，第二个4是4个隐藏节点值
            """ 这里做 dropout，清空一部分 layer_1值，并加强另一部分 """
            if do_dropout:
                layer_1 *= np.random.binomial([np.ones((len(layer_0), 4))], 1-dropout_percent)[0] * (1.0 / (1-dropout_percent))
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))  # 4*1
            layer_2_error = layer_2 - y
            if (j % 20000) == 0:
                print "Error after " + str(j) + " iteratioins:" + str(np.mean(np.abs(layer_2_error)))
            # lossfunc 对 laye_2 求导，是 loss func 对 z2 的求导
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)  # 4*1
            """ 这里要注意，根据原文，这里开始计算 layer_1_error:
            是 layer_2_delta 沿网络传播, 同时为 loss func 对 a1 的求导 == loss 对 z2 的求导 乘以 z2 对 a1 的求导"""
            layer_1_error = layer_2_delta.dot(synapse_1.T)  # 4*4
            """ 进而计算 layer_1_delta，也就是 loss 对 z1 的求导，== loss对a1的求导 * a1 对 z1 的求导 """
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)  # 4*4
            # lossfunc 对 synapse_1 求导，见下面等式右边；是上式再乘以 layer_2 对 synapse_1 求导，which is layer_1
            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))  # 4*1，对应隐藏层4个节点
            """ 原文这里类似上式计算 synapse_0，从而完成一次完整的 weight 更新，注意看整个流程和计算顺序 
            其实这里就是 loss 对 synapse_0 的求导 == loss对z1的求导 乘以 z1 对 synapse_0 的求导"""
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
            """ 这里的代码流程，和吴立德教程的方式不同，但是根据注释，其实是异曲同工的，理论上是完全一致的"""    


if __name__ == "__main__":
    tuning_alpha_with_hidden()



