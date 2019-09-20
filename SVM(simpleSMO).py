# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/18
from numpy import *
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

# 简易SOM算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter): # 参数输入：数据集、类别标签、常数C、容错率和最大循环次数
    dataMatrix = mat(dataMatIn) # (100,2)
    labelMat = mat(classLabels).transpose() # (100,1)
    b = 0 #超平面截距初值设为0
    m,n = shape(dataMatrix) # m = 数据集个数，n = 每个样本特征值数量
    alphas = mat(zeros((m,1))) # alpha初值都设为0
    iter = 0 # 当前迭代次数初值设置为0
    while iter < maxIter:
        alphaPairsChanged = 0 # 用于记录两个alpha是否已经被优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b # (100,1).T * (100,1)
            Ei = fXi - float(labelMat[i])
            ### 检测违反KKT条件的alpha
            ## 满足 KKT 条件
            # 1) yi*f(i) >= 1 and alpha == 0 (在边界外)
            # 2) yi*f(i) == 1 and 0<alpha< C (在边界上)
            # 3) yi*f(i) <= 1 and alpha == C (在边界中间)
            ## 违反 KKT 条件
            # 因为 y[i]*Ei = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, 因此
            # 1) 若 y[i]*Ei < 0, so yi*f(i) < 1, 如果 alpha < C, 则违反!(alpha = C 才是正确的)
            # 2) 若 y[i]*Ei > 0, so yi*f(i) > 1, 如果 alpha > 0, 则违反!(alpha = 0 才是正确的)
            # 3) 若 y[i]*Ei = 0, so yi*f(i) = 1, 在边界上, 无需优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): # 检验样本是否满足KKT条件
                j = selectJrand(i,m) # 样本不满足KKT条件，那么随机选择另一个alpha，准备对这两个alpha进行优化
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy() # 保存原先的alphas[i]
                alphaJold = alphas[j].copy() # 保存原先的alphas[j]
                if (labelMat[i] != labelMat[j]): # alpha_j必须满足的上下界H和L，具体原理https://blog.csdn.net/zouxy09/article/details/17292011
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue # H、L相等不需要做改变
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); continue # abs()为取绝对值
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) # 更新alpha_i 与j的方向相反
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if alphaPairsChanged == 0 : iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

# 获得超平面，并显示样本与超平面
def showSVM(alphas, b, dataMatIn, classLabels):
    # 数据散点图显示
    X = mat(dataMatIn)
    lableMat = mat(classLabels).transpose()
    m = shape(X)[0]
    for i in range(m):
        if lableMat[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], color = 'red')
        elif lableMat[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color = 'blue')
    plt.xlabel = 'x1'
    plt.ylabel = 'x2'

    # 计算w
    supportVectorsIndex = where(alphas>0)[0]
    w = zeros((2,1))
    for i in supportVectorsIndex:
        w += multiply(alphas[i]*lableMat[i],X[i].T) # w = (2,1)

    # 显示支持向量
    supportVectorsIndex = where(alphas>0)[0]
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
b,alphas = smoSimple(dataMatIn, classLabels, 0.6, 0.002, 40)
showSVM(alphas, b, dataMatIn, classLabels)
endtime = time.time()
print('运行时间：%f' %(endtime-starttime))
plt.show()