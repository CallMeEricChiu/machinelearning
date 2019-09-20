# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/1
import pandas as pd #数据读取
import matplotlib.pyplot as plt #坐标显示
import time
from numpy import *
from sklearn.model_selection import train_test_split  #引用交叉验证
import random
# 数据读取
def loadData():
    data = pd.read_csv('C:/Users/asus/Desktop/python/machine learning/logisticregression.csv')
    # 数据散点图显示
    # sns.pairplot(data,x_vars='x1',y_vars='x2',height=15,aspect=0.8,hue='y')
    # plt.show()

    x = data[['x0','x1', 'x2']]
    y = data['y']
    # 默认将数据集分为75%训练集和25%测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    return mat(x_train), mat(x_test), mat(y_train).transpose(), mat(y_test).transpose()

# 定义s函数
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

# 训练逻辑回归函数
def trainLogRegres(x_train, y_train, opts):
    #计算时间
    starttime = time.time()

    num_samples, num_features = shape(x_train)
    alpha = opts['alpha']; #学习率
    maxIter = opts['maxIter'] #最大迭代次数
    weights = ones((num_features, 1)) #num_features行1列的数组，值都为1，即设置模型参数初值为1

    # 梯度上升优化
    for k in range(maxIter):
        if opts['optimizeType'] == 'GradDescent':  #梯度上升(gradient descent)，这里使用梯度上升原理是相同的
            output = sigmoid(x_train * weights)
            error = y_train - output
            weights = weights + alpha * x_train.transpose() * error
        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            for i in range(num_samples):   #i = int(random.uniform(0, num_samples))
                output = sigmoid(x_train[i, :] * weights)
                error = y_train[i, 0] - output
                weights = weights + alpha * x_train[i, :].transpose() * error
        # else:
        #     raise NameError('Not support optimize method type!')
    print('Congratulations, training complete! Took %fs!' % (time.time() - starttime))
    return weights

# 逻辑回归模型曲线显示
def showLogRegres(weights, x_train, y_train):
    # train_x and train_y 是矩阵数据类型
    numSamples, numFeatures = shape(x_train)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # 显示所有数据点
    for i in range(numSamples):
        if int(y_train[i, 0]) == 0:
            plt.plot(x_train[i, 1], x_train[i, 2], 'or')
        elif int(y_train[i, 0]) == 1:
            plt.plot(x_train[i, 1], x_train[i, 2], 'ob')

    # 得到分界线
    min_x = min(x_train[:, 1])[0, 0] # min_x 的值为矩阵 min(train_x[:, 1]) 中0行0列的值
    max_x = max(x_train[:, 1])[0, 0]
    weights = array(weights)  # 转化为数组
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 测试训练出来的逻辑回归模型
def testLogRegress(weights, x_test, y_test):
	numSamples, numFeatures = shape(x_test)
	matchCount = 0
	for i in range(numSamples):
		predict = sigmoid(x_test[i, :] * weights)[0, 0] > 0.5
		if predict == bool(y_test[i, 0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy

print("step 1: load data...")
x_train, x_test, y_train, y_test = loadData()

print("step 2: training...")
opts = {'alpha':0.01, 'maxIter':200,'optimizeType':'stocGradDescent'}
optimalWeights = trainLogRegres(x_train, y_train, opts)

print("step 3: testing...")
accuracy = testLogRegress(optimalWeights, x_test, y_test)

print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showLogRegres(optimalWeights, x_train, y_train)




























