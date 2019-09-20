# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/8/21
from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

# 计算数据集信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 创建字典，分别记录标签为yes和no的个数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) # log以2为底
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 特征数量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):        # 对所有特征进行迭代
        featList = [example[i] for example in dataSet] # 创建所有样本的该特征的列表
        uniqueVals = set(featList)      # 将上述列表转换为内容不重复的数据集
        newEntropy = 0.0
        for value in uniqueVals: # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     # 计算该瓜分方式的信息增益
        if (infoGain > bestInfoGain):       # 记录最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                      # 返回最佳划分方式

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount, key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet,labels):
    thislabels = lables.copy()
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 类别相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1: # 使用完了所有的特征值任然不能讲数据集划分成仅包含位移类别的分组
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = thislabels[bestFeat]
    myTree = {bestFeatLabel:{}}

    del(thislabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) # 得到最佳划分特征的所有属性值

    for value in uniqueVals:
        subLabels = thislabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print('firstStr:'+firstStr)
    secondDict = inputTree[firstStr]
    print(secondDict)
    featIndex = featLabels.index(firstStr)
    print('featIndex:%d' % featIndex)
    key = testVec[featIndex]
    print('key:%d'%key)
    valueOfFeat = secondDict[key]
    print('valueOfFeat:',valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

if __name__=='__main__':
    dataSet,lables = createDataSet()
    myTree = createTree(dataSet,lables)
    print('mytree is:',myTree)
    classify(myTree,lables,[1,1])
