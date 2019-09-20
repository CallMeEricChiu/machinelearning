# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/19
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

# 数据读取
def loaddata():
    data = pd.read_csv('C:/Users/asus/Desktop/python/machine learning/SVMdata(linear).csv')
    dataMatIn = data[['x1','x2']] # (100,2)
    classLabels = data['y'] # (1,100)
    return dataMatIn, classLabels

# 从[0,m)中随机选择一个和i不相等的整数
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 用于调整大于H或者小于L的alpha的值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # 初始化结构参数
        self.X = dataMatIn # 数组
        self.labelMat = classLabels # 数组
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 用于存放Ek值，第一列是有效标志

# 计算Ek
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 选择另一个alpha
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 将第i个样本的Ek缓存标志置1，表示i个数据的E已经计算好了
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0] # mat.A 将矩阵转化为数组
    if len(validEcacheList) > 1:
        for k in validEcacheList:  # 查询有效E缓存值，找到最大的delta E
            if k == i: continue  # k=i 无需计算直接跳出此次for循环
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 这种情况下(第一次) 我们没有任何有效的E缓存值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# 内循环
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): # 检验样本是否满足KKT条件
        j,Ej = selectJ(i, oS, Ei) # 选择第二个alpha_j
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if abs(oS.alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) #added this for the Ecache
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):    # 完整的SMO算法
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True    # alphas初值都为0 所以第一次先遍历所有数据样本，而后再去比遍历非边界数据样本
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        # 优先选择遍历非边界数据样本，因为非边界数据样本更有可能需要调整，边界数据样本常常不能得到进一步调整而留在边界上。
        # 由于大部分数据样本都很明显不可能是支持向量，因此对应的α乘子一旦取得零值就无需再调整。遍历非边界数据样本并选出他们
        # 当中违反KKT条件为止。当某一次遍历发现没有非边界数据样本得到调整时，遍历所有数据样本，以检验是否整个集合都满足KKT
        # 条件。如果整个集合的检验中又有数据样本被进一步进化，则有必要再遍历非边界数据样本。这样，不停地在遍历所有数据样本和
        # 遍历非边界数据样本之间切换，直到整个样本集合都满足KKT条件为止。以上用KKT条件对数据样本所做的检验都以达到一定精度ε
        # 就可以停止为条件。如果要求十分精确的输出算法，则往往不能很快收敛。

        if entireSet:   # 遍历全部数据样本
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:# 遍历非边界数据样本
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif alphaPairsChanged == 0: entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

# 获得超平面，并显示样本与超平面
def showSVM(alphas, b, dataMatIn, classLabels):
    # 数据散点图显示
    X = np.mat(dataMatIn)
    lableMat = np.mat(classLabels).transpose()
    m = np.shape(X)[0]
    for i in range(m):
        if lableMat[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], color = 'red')
        elif lableMat[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color = 'blue')
    plt.xlabel = 'x1'
    plt.ylabel = 'x2'

    # 计算w
    supportVectorsIndex = np.where(alphas>0)[0]
    w = np.zeros((2,1))
    for i in supportVectorsIndex:
        w += np.multiply(alphas[i]*lableMat[i],X[i].T) # w = (2,1)

    # 显示支持向量
    supportVectorsIndex = np.where(alphas>0)[0]
    for i in supportVectorsIndex:
        plt.plot(X[i, 0], X[i, 1], 'oy')

    # 显示超平面
    min_x = min(X[:, 0])[0, 0]
    max_x = max(X[:, 0])[0, 0]
    y_min_x = float(-b - w[0] * min_x) / w[1]
    y_max_x = float(-b - w[0] * max_x) / w[1]
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'y--')


starttime = time.time()
dataMatIn, classLabels = loaddata()
b,alphas = smoP(dataMatIn, classLabels, 0.6, 0.002, 40)
showSVM(alphas, b, dataMatIn, classLabels)
endtime = time.time()
print('运行时间：%f' %(endtime-starttime))
plt.show()































