#encoding=utf-8

""" http://outlace.com/Simple-Recurrent-Neural-Network/ 
    build an elman recurrent nn to solve xor problem
    with 4 hidden nodes

    so: theta1 6*4 --> (1input + 4context + 1bias) * 4hidden
        theta2 5*1 --> (4hidden + 1bias) * 1result
    plz notes that #context == #hidden == 4, it is by design

    Results: failed during scipy.optimize, as below:
    $ python2.7 simple_rnn.py
    tnc: Version 1.3, (c) 2002-2003, Jean-Sebastien Roy (js@jeannot.org)
    tnc: RCS ID: @(#) $Jeannot: tnc.c,v 1.205 2005/01/28 18:27:31 js Exp $
    NIT   NF   F                       GTG
    0    1  0.000000000000000E+00   3.59022086E-01
    0   53  0.000000000000000E+00   3.59022086E-01
    tnc: Linear search failed
"""

import numpy as np
from scipy import optimize
from sigmoid import sigmoid
import cost_xorRNN as cr

def runForward(X, theta1, theta2, numHid):
    m = X.shape[0]
    hid_last = np.zeros((numHid, 1))  # context unit last time, initialized as 0
    results = np.zeros((m, 1))  # save output
    for j in range(m):  # one sample a time
        context = hid_last
        x_context = np.concatenate((X[j,:], context))  # concat( (1*1, 4*1) ) ==> 5*1
        a1 = np.matrix(np.concatenate((x_context,np.matrix('[1]'))))  # now add bias, make it 6*1
        z2 = theta1.T * a1  # (6*4).T * 6*1 ==> 4*1
        a2 = np.concatenate((sigmoid(z2), np.matrix('[1]')))  # now add hidden layer bias ,make it 5*1
        hid_last = a2[0:-1, 0]  # update hid_last
        z3 = theta2.T * a2  # (5*1).T * 5*1 ==> 1*1
        a3 = sigmoid(z3)
        results[j] = a3
    return results


def xorTrain():
    X = np.matrix('[0;0;1;1;0]')  # xor of the previous 2 bits
    y = np.matrix('[0;0;1;0;1]')  # so, first bit should be ignored
    numIn, numHid, numOut = 1, 4, 1
    theta1 = np.matrix(0.5 * np.sqrt(6 / (numIn + numHid)) * np.random.randn(numIn + numHid + 1, numHid)) 
    theta2 = np.matrix(0.5 * np.sqrt(6 / (numOut + numHid)) * np.random.randn(numHid + 1, numOut)) 

    # unrolling (concatnate) theta1 & theta2 into 1-d long vector
    thetaVec = np.concatenate((theta1.flatten(), theta2.flatten()), axis=1) # concat((1*24, 1*25),axis=1) --> 1*29
    # give the optimizer our cost function and our unrolled weight vector
    opt = optimize.fmin_tnc(cr.costRNN, thetaVec, args=(X,y), maxfun=5000)
    # retrieve optmized weight
    optTheta = np.array(opt[0])
    # reshape back to update theta1 & theta2
    theta1_len = (numIn + numHid + 1) * numHid
    theta1 = optTheta[0:theta1_len].reshape(numIn + numHid + 1, numHid)
    theta2 = optTheta[theta1_len:].reshape(numHid + 1, numOut)
    return theta1, theta2, numHid


if __name__ == '__main__': 
    theta1, theta2, numHid = xorTrain()
    print theta1
    print theta2
    print numHid
    Xt = np.matrix('[1;0;0;1;1;0]')
    print np.round(runForward(Xt, theta1, theta2, numHid).T)


