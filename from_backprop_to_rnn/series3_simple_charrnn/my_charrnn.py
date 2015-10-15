#encoding=utf-8

""" http://outlace.com/Simple-Recurrent-Neural-Network/ 
    build an elman recurrent nn to implement a simple char-rnn

    so: theta1 15*10 --> (4input + 10context + 1bias) * 10hidden
        theta2 11*4 --> (10hidden + 1bias) * 4result
    plz notes that #context == #hidden == 10, it is by design

    注意：
    1. 这里的Input和 simple_rnn 中的不同，是 1*xx 的；而后者是 xx*1 的
       参见 np.concatenate 函数调用中，是否带 axis=1 ，带的就是横向 concat
    2. 原 post 中的代码有两个问题
       - 运行中会报错 tnc: Linear search failed，是 scipy 无法找到优化值
       - 数学错误，根据吴立德视频， delta 是 loss 对 z 的导数，而不是 loss 本身
       故此，这里不采用 scipy.optimize，而且修正了原post中的数学错误
    3. 仍采用 cross-ent loss 函数，故此 backprop 的部分会有一些不同
    4. 最后很有可能训练不出来，是因为掉进了 local optimal，见原post，解释了为何难以训练
"""

import numpy as np
from scipy import optimize
from sigmoid import sigmoid

#Vocabulary: h,e,l,o
#Encoding: h=[0,0,1,0], e=[0,1,0,0], l=[0,0,0,1], o=[1,0,0,0]

def runForward(X, theta1, theta2, numHid, numOut):
    m = X.shape[0]
    hid_last = np.zeros((numHid, 1))  # context unit last time, initialized as 0
    results = np.zeros((m, numOut))  # save output, so given 4 samples (each of which is 1*4), output 4 * 4 too
    for j in range(m):  # one sample a time
        context = hid_last
        x_context = np.concatenate((X[j,:], context.T), axis=1)  # concat( (1*4, 10*1) ) ==> 1*14
        a1 = np.matrix(np.concatenate((x_context,np.matrix('[1]')), axis=1)).T # now add bias, make it 1 * 15; then .T -> 15*1
        z2 = theta1.T * a1  # (15*10).T * 15*1 ==> 10*1
        a2 = np.concatenate((sigmoid(z2), np.matrix('[1]')))  # now add hidden layer bias ,make it 5*1
        hid_last = a2[0:-1, 0]  # update hid_last
        z3 = theta2.T * a2  # (10*4).T * 10*1 ==> 4*1
        a3 = sigmoid(z3)
        results[j, :] = a3.reshape(numOut,)  # line of results is the result of the input on current step
    return results


def charTrain():
    X = np.matrix('0,0,1,0; 0,1,0,0; 0,0,0,1; 0,0,0,1; 1,0,0,0')  # encoding for hello
    numIn, numHid, numOut = 4, 10, 4
    numInTot = numIn + numHid + 1
    theta1 = np.matrix(1 * np.sqrt(6 / (numIn + numHid)) * np.random.randn(numInTot, numHid)) 
    theta2 = np.matrix(1 * np.sqrt(6 / (numOut + numHid)) * np.random.randn(numHid + 1, numOut)) 
    theta1_grad = np.zeros((numInTot, numHid))
    theta2_grad = np.zeros((numHid + 1, numOut))
    hid_last = np.zeros((numHid, 1))
    m = X.shape[0]
    alpha = 0.05
    for ita in range(5000):
        for j in range(m-1): #for every training element, except for the last one, which we don't know what is followed
            y = X[j+1, :]  # given the input char, the next char is expected
            # forward
            context = hid_last
            x_context = np.concatenate((X[j, :], context.T), axis=1)
            a1 = np.matrix(np.concatenate((x_context, np.matrix('[1]')), axis=1)).T
            z2 = theta1.T * a1;
            a2 = np.concatenate((sigmoid(z2), np.matrix('[1]')))
            hid_last = a2[0:-1, 0];
            z3 = theta2.T * a2
            a3 = sigmoid(z3)
            # backward propagation
            d3 = np.multiply(z3.T, (a3.T - y))   # 1*4, d(loss)/d(z) = z * (a3 - y)
            theta2 = theta2 - alpha * a2 * d3  # 11*1 * 1*4 => 11*4a  d(loss)/d(theta2) = d(loss)/d(z3) * d(z3)/d(theta2)
            d2 = np.multiply((theta2 * d3.T), np.multiply(a2, (1 - a2)))  # (11*4 * 4*1) multiply ( 11*1 multiply 11*1) => 11*1
            theta1 = theta1 - alpha * a1 * d2[0:numHid,:].T  # 15*1 * 1*10 => 15*10
    return theta1, theta2, numHid, numOut


if __name__ == '__main__': 
    theta1, theta2, numHid, numOut = charTrain()
    Xt = np.matrix('0,0,1,0; 0,1,0,0; 0,0,0,1; 0,0,0,1')  # expect 'ello' ==> 0,1,0,0; 0,0,0,1; 0,0,0,1; 1,0,0,0
    print np.round(runForward(Xt, theta1, theta2, numHid, numOut).T)


