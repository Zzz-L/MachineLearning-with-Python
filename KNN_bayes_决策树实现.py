# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:37:08 2017

@author: Zzz~L
"""

#---------------------KNN-------------------
"""参数n_neighbors：用于指定分类器中K的大小
   参数weights：设置选中的K个点对分类结果影响的权重
   参数algorithm：设置用于计算距离的方法
   一般情况下，K 会倾向选取较小的值，并使用交叉验证法选取最优 K 值
"""
from sklearn.neighbors import KNeighborsClassifier
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3) #创建KNN分类器
neigh.fit(X, y) #调用函数
print(neigh.predict([[1.1]])) #对未知类别预测
#---------------------------决策树----------------------------   
"""参数criterion ：用于选择特征的准则，可以传入“gini”代表基尼系数，或者“entropy”代表信息增益
   参数max_features ：表示在决策树结点进行分裂时，从多少个特征中选择最优特征
   在实际使用中，需要根据数据情况，调整DecisionTreeClassifier类中传入的
   参数，比如选择合适的criterion，设置随机变量等
"""
from sklearn.tree import DecisionTreeClassifier #导入决策树分类器
from sklearn.model_selection import cross_val_score #导入交叉验证函数
from sklearn.datasets import load_iris #导入数据集
clf = DecisionTreeClassifier() #创建决策树分类器,使用默认参数
iris = load_iris()
cross_val_score(clf,iris.data,iris.target,cv=10)#cross_val_score函数的返回值就是对于每次不同的的划分raw data时，在testdata上得到的分类的准确率   
clf.fit(X, y) #训练模型
clf.predict(2.2) #预测未知类别
#-------------------------朴素贝叶斯-----------------------------     
"""naive_bayes.GussianNB 高斯朴素贝叶斯
   naive_bayes.MultinomialNB 针对多项式模型的朴素贝叶斯分类器
   naive_bayes.BernoulliNB 针对多元伯努利模型的朴素贝叶斯分类器
   朴素贝叶斯一般在小规模数据上的表现很好，适合进行多分类任务
"""
#参数priors ：给定各个类别的先验概率
from sklearn.naive_bayes import GaussianNB #导入高斯贝叶斯分类器
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB(priors=None) #创建贝叶斯分类器
clf.fit(X, Y)  #模型训练
print(clf.predict([[-0.8, -1]])) #预测


   
