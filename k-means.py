# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/24
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
# 数据载入
def loaddata():
    dataSet = pd.read_csv('K-means_testSet.csv')
    return mat(dataSet)

# 样本显示
def showdata(dataSet, centroids, clusterAssment):
    X = mat(dataSet)
    m = shape(X)[0]
    # clusterAssment = clusterAssment.astype(int16)
    colorlist = ['blue','green','yellow','black']
    for i in range(m):
        plt.scatter(X[i,0], X[i,1],color = colorlist[ int(clusterAssment[i,0]) ] )
    n = shape(centroids)[0]
    for j in range(n):
        plt.scatter(centroids[j,0], centroids[j,1], color = 'red')
    plt.show()
# 计算两个向量的欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为数据集设置一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1] # n=2 样本数据有两个特征
    centroids = mat(zeros((k,n))) # 创建质心矩阵
    for j in range(n): # 创建随机聚类中心
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ) # 所有样本每个特征的最小值与最大值的差值
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

# k均值算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] # 样本数
    clusterAssment = mat(zeros((m,2))) # 簇分配结果矩阵，第一列存放簇索引值，第二列存放误差
    centroids = createCent(dataSet, k) # 创建初始质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): # 为每个样本分配给距离质心最近的簇f
            minDist = inf # inf=正无穷
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k): # 重新计算质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] # 每个样本属于的簇
            centroids[cent,:] = mean(ptsInClust, axis=0) # 计算新的质心
    return centroids, clusterAssment

centroids, clusterAssment = kMeans(loaddata(), 4, distMeas=distEclud, createCent=randCent)
# print(centroids,'\n',clusterAssment)
showdata(loaddata(), centroids, clusterAssment)