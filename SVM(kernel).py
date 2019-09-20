# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/23
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split

# 数据读取
def loaddata():
    data = pd.read_csv('C:/Users/asus/Desktop/python/machine learning/SVMdata(nonlinear).csv')

    positivelable_data = data[data['y'].isin([1])]
    negativelable_data = data[data['y'].isin([-1])]

    positivelable_data_x = positivelable_data[['x1','x2']]
    positivelable_data_y = positivelable_data['y']
    negativelable_data_x = negativelable_data[['x1','x2']]
    negativelable_data_y = negativelable_data['y']

    x_train1, x_test1, y_train1, y_test1 = train_test_split(positivelable_data_x,positivelable_data_y,random_state=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(negativelable_data_x,negativelable_data_y,random_state=1)

    x_train = pd.concat([x_train1, x_train2])
    y_train = pd.concat([y_train1, y_train2])
    x_test = pd.concat([x_test1, x_test2])
    y_test = pd.concat([y_test1, y_test2])
    return x_train, y_train, x_test, y_test

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

# 核函数转换
def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': # 线性核函数
        K = X * A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 初始化结构参数
        self.X = dataMatIn # 数组
        self.labelMat = classLabels # 数组
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 用于存放Ek值，第一列是有效标志
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
# 计算Ek
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
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
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if abs(oS.alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) #added this for the Ecache
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# 获得超平面，并显示样本与超平面
def showSVM(alphas, dataMatIn, classLabels):
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

    # 显示支持向量
    supportVectorsIndex = np.where(alphas>0)[0]
    for i in supportVectorsIndex:
        plt.plot(X[i, 0], X[i, 1], 'oy')

    return w

# 完整的SMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup, x_test, y_test): # C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(), C, toler, kTup)
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
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:# 遍历非边界数据样本
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter , i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif alphaPairsChanged == 0: entireSet = True
        print("iteration number: %d" % iter)

    # 计算w
    supportVectorsIndex = np.where(oS.alphas>0)[0]
    w = np.zeros((2,1))
    for i in supportVectorsIndex:
        w += np.multiply(oS.alphas[i]* oS.labelMat[i], oS.X[i].T) # w = (2,1)

    accuarcy = testSVM(oS, x_test, y_test, kTup)
    return oS.b, oS.alphas,w ,accuarcy

def testSVM(oS, test_x, test_y, kTup):
	test_x = np.mat(test_x)
	test_y = np.mat(test_y).T
	numTestSamples = test_x.shape[0]
	supportVectorsIndex = np.nonzero(oS.alphas.A > 0)[0]
	supportVectors 		= oS.X[supportVectorsIndex]
	supportVectorLabels = oS.labelMat[supportVectorsIndex]
	supportVectorAlphas = oS.alphas[supportVectorsIndex]
	matchCount = 0
	for i in range(numTestSamples):
		kernelValue = kernelTrans(supportVectors, test_x[i, :], kTup)
		predict = kernelValue.T * np.multiply(supportVectorLabels, supportVectorAlphas) + oS.b
		if np.sign(predict) == np.sign(test_y[i]):
			matchCount += 1
	accuracy = float(matchCount) / numTestSamples
	return accuracy

if __name__ == '__main__':
    starttime = time.time()
    dataMatIn, classLabels, x_test, y_test= loaddata()
    b, alphas, w, accuarcy = smoP(dataMatIn, classLabels, 0.6, 0.002, 40,('rbf',1), x_test, y_test)
    # gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，
    # 支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
    showSVM(alphas, dataMatIn, classLabels)
    endtime = time.time()
    print('accuarcy = %f'%accuarcy)
    print('运行时间：%f' %(endtime-starttime))
    plt.show()

































