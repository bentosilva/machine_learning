#encoding=utf-8

""" http://iamtrask.github.io/2015/07/27/python-network-part2/
    对 array，dot 为点积以及矩阵乘法
              * 为按位乘法
    对 matrix，dot 为点积，* 为矩阵乘法
              np.multiply为按位乘法
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_output_to_derivative(output):
    return output * (1- output)

""" part 1. 无隐藏层，相当于 logistic regression，损失函数为最小二乘 
 loss func = (h(layer_1) - y) ** 2
[[ 0.00505119]
 [ 0.00505119]
 [ 0.99494905]
 [ 0.99494905]]
"""
def no_hidden():
    X = np.array([[0,1],[0,1],[1,0],[1,0]]) # 4*2
    y = np.array([[0, 0, 1, 1]]).T          # 4*1
    np.random.seed(1)
    synapse_0 = 2 * np.random.random((2,1)) - 1  # init weights with mean 0, 2*1
    for iter in xrange(10000):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))  # 4*2 * 2*1 => 4*1
        layer_1_error = layer_1 - y
        # 这里是 lossfunc 对 layer_1 求导
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1) # 4*1 按位乘 4*1
        # lossfunc 对 synapse 求导，等于对 layer_1 的求导，乘以 layer_1 对 synapse 的求导
        synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)  # 2*4 * 4*1
        # learning rate set to 1.0
        synapse_0 -= synapse_0_derivative
    print "Output After Training:"
    print layer_1


""" part 2. 加入隐藏层，并加入和调控 alpha 参数 """
def tuning_alpha_with_hidden():
    alphas = [0.001,0.01,0.1,1,10,100,1000]
    # 第四个参数都是1，故此其实没有意义
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],])
    y = np.array([[0],[1],[1],[0],])  # 把X的最后一维去掉后，我们发现其实就是一个 XOR 
    for alpha in alphas:
        print "\nTraining with alpha: " + str(alpha)
        np.random.seed(1)
        synapse_0 = 2*np.random.random((3,4)) - 1 # 3*4, no bias field obviously
        synapse_1 = 2*np.random.random((4,1)) - 1 # 4*1, so  4*3 (X) * 3*4 * 4*1 => 4*1 (y), 4 hidden units
        for j in xrange(60001):
            # Forward
            layer_0 = X  # 4*3
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))  # 4*4，第一个4 是4个样本，第二个4是4个隐藏节点值
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))  # 4*1
            layer_2_error = layer_2 - y
            if (j % 10000) == 0:
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
#    no_hidden()
    tuning_alpha_with_hidden()



