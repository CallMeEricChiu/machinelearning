# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/6/27
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = [13854,12213,11009,10655,9503] #程序员工资，顺序为北京，上海，杭州，深圳，广州
x = np.reshape(x,newshape=(5,1)) / 10000.0#特征缩放
y =  [21332, 20162, 19138, 18621, 18016] #算法工程师，顺序和上面一致
y = np.reshape(y,newshape=(5,1)) / 10000.0

def model(a, b, x):#预测模型
    return a*x + b

def cost_function(a, b, x, y):
    n = 5
    return 0.5/n * (np.square(y-a*x-b)).sum()

def optimize(a,b,x,y):#优化算法
    n = 5             #样本数量
    alpha = 1e-1      #学习率
    y_hat = model(a,b,x)
    da = (1.0/n) * ((y_hat-y)*x).sum()
    db = (1.0/n) * ((y_hat-y).sum())
    a = a - alpha*da
    b = b - alpha*db
    return a,b

def iterate(a,b,x,y,times): #设置迭代次数
    for i in range(times):
        a,b = optimize(a,b,x,y)
    y_hat = model(a,b,x)
    cost = cost_function(a, b, x, y)
    print(a,b,cost) #返回ab值 以及 代价函数值
    plt.scatter(x,y)#显示原始数据
    plt.plot(x,y_hat)#显示拟合曲线
    plt.show()
    return a,b

a = 0
b = 0
a,b=iterate(a,b,x,y,10000)











































