# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/5
from sklearn.datasets import load_digits  # 数据集
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from sklearn.model_selection import train_test_split  # 数据集分割
from numpy import *
import matplotlib.pyplot as pl #数据可视化

def sigmoid(x):  # 激活函数
    sig=1/(1 + exp(-x))
    return sig
def dsigmoid(x):  # sigmoid的导数
    return x * (1 - x)
def data_split(X,y): # 将数据划分为训练集和测试集
    X_tarin, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    return X_tarin, X_test, y_train, y_test

class NeuralNetwork:
    def __init__(self, layers):  # 这里是三层网络，列表[64,100,10]表示输入，隐藏，输出层的单元个数
        # 初始化权值，范围1~-1
        self.V = random.random((layers[0] + 1, layers[1])) * 2 - 1  # 输入层至隐藏层权值(65,100)，之所以是65，因为有偏置W0
        self.W = random.random((layers[1], layers[2])) * 2 - 1  # 隐藏层至输出层权值(100,10)

    def train(self, X_train, lables_train, X_test, y_test, lr=0.1, epochs=20000):
        # lr为学习率，epochs为迭代的次数
        # lables_train为训练集y_train归一化后的矩阵

        # 为数据集添加偏置
        temp = ones([X_train.shape[0], X_train.shape[1] + 1])
        temp[:, 0:-1] = X_train # -1为导数为倒数第一个
        X_train = temp  # 这里最后一列为偏置

        # 进行权值训练更新
        for n in range(epochs + 1):
            i = random.randint(0,X_train.shape[0])  # 随机选取一行数据(一个样本)进行更新
            x = X_train[i]
            x = atleast_2d(x)  # 转为二维数据 x=1*65 X=1797*65

            L1 = x  # 输入层数据输入 L1=1*65
            L2 = sigmoid(L1.dot(self.V))  # 隐层输出(1,100)
            L3 = sigmoid(L2.dot(self.W))  # 输出层输出(1,10)

            # 误差delta
            L3_delta = lables_train[i] - L3 # (1,10)
            L2_delta = L3_delta.dot(self.W.T) * dsigmoid(L2)   # (1,100)
            # L1是输入层不存在误差

            # 更新
            self.W += lr * L2.T.dot(L3_delta)  # (100,10)
            self.V += lr * L1.T.dot(L2_delta)  #

            # 每训练1000次预测准确率
            if n % 1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):
                    out = self.predict(X_test[j])  # 用验证集去测试
                    predictions.append(argmax(out))  # 返回预测结果,argmax()返回最大值索引
                accuracy = mean(equal(predictions, y_test))  # 求平均值
                print('epoch:', n, 'accuracy:', accuracy)

    def predict(self, x):
        # 添加转置,这里是一维的
        temp = ones(x.shape[0] + 1)
        temp[0:-1] = x
        x = temp
        x = atleast_2d(x) # 转为二维数据

        L1 = x
        L2 = sigmoid(dot(L1, self.V))  # 隐层输出
        L3 = sigmoid(dot(L2, self.W))  # 输出层输出
        return L3

if __name__ == '__main__':
    digits = load_digits()  # 载入数据
    print(digits.data.shape)  # 打印数据集大小(1797L, 64L）
    # pl.gray()  # 灰度化图片
    # pl.matshow(digits.images[3])  # 显示第1张图片，上面的数字是0
    # pl.show()
    X = digits.data
    y = digits.target

    # 数据归一化,一般是x=(x-x.min)/x.max-x.min
    X1 = X - X.min()
    X2 = X.max() - X.min()
    X = X1/X2
    # 创建神经网络
    BPNN = NeuralNetwork([64,100,10])

    # 创建训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X , y)

    #标签二值化
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    BPNN.train(X_train, labels_train, X_test, y_test)