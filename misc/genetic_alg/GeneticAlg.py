# encoding: utf-8

"""
    使用 GA 算法来更新 weight ，而不是使用 bp
"""

import random as rn
import numpy as np
import NeuralNetXOR as NN


# [Initial population size, mutation rate (=1%), num generations (30), solution length (30), # winners/per gen]
# solLen is 13, which is exactly the weight len for the Xor Neural Network
initPop, mutRate, numGen, solLen, numWin = 80, 0.01, 30, 13, 10
# 从 np.arange(-15, 15, step=0.01) 这个集合中，choose (initPop, solLen) 个样本， without replacement，即选中过的就不能再被选
curPop = np.random.choice(np.arange(-15, 15, step=0.01), size=(initPop, solLen), replace=False)  # 80 X 13
nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))  # 80 X 13
fitVec = np.zeros((initPop, 2))  # 1st col is indices, 2nd is cost, 80 X 2

# 使用 GA 来更新 weight, which is curPop[x]
for i in range(numGen):
    # 80 个 weight，每个都计算一次前馈错误，并求和(Xor 4 种情况的错误求和)，作为该 weight 的错误
    fitVec = np.array([np.array([x, np.sum(NN.costFunction(NN.X, NN.y, curPop[x].T))]) for x in range(initPop)])
    # 80 个错误再求和，作为本迭代的总错误
    print "Error of iteration #%d : %f" % (i, np.sum(fitVec[:, 1]))
    winners = np.zeros((numWin, solLen))
    # 找到本迭代 80 个 weight 中的 winner
    for n in range(len(winners)):
        # 每找一个 winner 都是先随机取出一小部分结果，在从这小部分中找到最好的
        # 而不是 80 个中的最好若干个，保证了随机性
        selected = np.random.choice(range(len(fitVec)), numWin / 2, replace=False)
        wnr = np.argmin(fitVec[selected, 1])
        # 每个winner 实际都是一个weight， 1 X 13 维
        winners[n] = curPop[int(fitVec[selected[wnr]][0])]

    # 开始变异，过程如下：
    # 下一代的前 len(winners) 个就是上一代的 winners
    nextPop[:len(winners)] = winners
    # 下一代的剩余部分，会同样使用 winners 来填充
    dupliWin = np.zeros((initPop - len(winners), winners.shape[1]))
    # 对 weight 的每一维 (共计 13 维)
    for x in range(winners.shape[1]):
        # 找到填充的倍数
        numDups = ((initPop - len(winners)) / len(winners))
        # 每一列重复填充上面的倍数，显然纵向填充，axis=0
        dupliWin[:, x] = np.repeat(winners[:, x], numDups, axis=0)
        dupliWin[:, x] = np.random.permutation(dupliWin[:, x])  # 然后洗一下，就是 cross-over 过程
    nextPop[len(winners):] = np.matrix(dupliWin)
    # mutation, 此处 nextPop.size 返回的是 shape[0] * shape[1]，即 80 * 13
    mutMatrix = [np.float(np.random.normal(0, 2, 1)) if rn.random() < mutRate else 1 for x in range(nextPop.size)]
    nextPop = np.multiply(nextPop, np.matrix(mutMatrix).reshape(nextPop.shape))
    curPop = nextPop

# 迭代完毕后，选取最后一代的最好的那个 weight
best_soln = curPop[np.argmin(fitVec[:, 1])]
result = np.round(NN.runForward(NN.X, best_soln.T))
print("Best Sol'n:\n%s\nCost:%s" % (best_soln, np.sum(NN.costFunction(NN.X, NN.y, best_soln.T))))
print("When X = \n%s \nhThetaX = \n%s" % (NN.X[:,:2], result,))
