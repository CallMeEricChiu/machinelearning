# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/6/27
import pandas as pd #数据读取
import seaborn as sns #坐标显示
import matplotlib.pyplot as plt #坐标显示
from sklearn.model_selection import train_test_split  #这里引用了交叉验证
from sklearn.linear_model import LinearRegression #线性回归

data = pd.read_csv('C:/Users/asus/Desktop/python/machine learning/Advertising.csv')
#print(data.head(20))#输出excel文件中的前五行数据（第一行默认为列名，第一列默认为索引）

#sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', height=15, aspect=0.8,kind='reg')
#plt.show()

#create a python list of feature names
feature_cols = ['TV','radio']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# print the first 5 rows
# print(X.head())
# check the type and shape of X
# print(type(X))
# print(X.shape)

# select a Series from the DataFrame
y = data['sales']
# print the first 5 values
# print(y.head())

#设置训练集和测试集 train_test_split(X, y, test_size=0.33, random_state=0)
X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)#默认将数据集分为75%训练集和25%测试集

linreg = LinearRegression()
model=linreg.fit(X_train, y_train)

print(linreg.intercept_) #模型中的独立项
print(linreg.coef_) #模型中的各系数

y_pred = linreg.predict(X_test) #测试集得到的预测值
# print(y_test)
# print(y_pred)

plt.figure(num=None, figsize=None,dpi=100,facecolor='grey',edgecolor=None) #设置曲线窗口各参数
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()

































