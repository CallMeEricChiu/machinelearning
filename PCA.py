# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/28
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

# 数据载入
def loaddata():
    dataSet = pd.read_csv('PCA_testSet.csv')
    return mat(dataSet)

def pca(dataMat, topNfeat=1):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals # 中心化
    covMat = cov(meanRemoved, rowvar = False) # DATA.T*DATA 得到（特征数维度*特征数维度）的矩阵
    eigVals,eigVects = linalg.eig(mat(covMat)) # 获得协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals)            # argsort函数返回的是数组值从小到大的索引值，索引值越大数组值就越大
    eigValInd = eigValInd[:-(topNfeat-1):-1]  # 去掉不需要的特征，并且数组索引至从大到小排列
    redEigVects = eigVects[:,eigValInd]       # 根据从大到小的特征值读取其对应的特征向量
    lowDDataMat = meanRemoved * redEigVects # 将数据转换到新的空间中
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 数据重构回原来的维度
    return lowDDataMat, reconMat

data = loaddata()
lowDDataMat, reconMat = pca(data,1)
print(lowDDataMat,'\n',reconMat,'\n',data)




































