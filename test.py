# -*- coding：utf-8 -*-
# Author:Eric Chiu Time:2019/7/1
from sklearn.datasets import load_digits  # 数据集
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from sklearn.model_selection import train_test_split  # 数据集分割
import numpy as np


if __name__ == '__main__':
    a = np.array([[1,2],[3,4],[5,6]])
    print(a)

    b = np.sum(a[0][0])
    print(b)



















