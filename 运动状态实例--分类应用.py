# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:39:53 2017

@author: Zzz~L
"""
"""  Imputer类处理缺失值,用法见31行
     train_test_split 拆分训练集与测试集
     classification_report 分类预测结果评估,用法见66行
     知识小结：1.np.ndarray(shape=(0,41)) 构造固定结构的数组
     2.np.concatenate((feature,df)) 按照行联结数组序列
     3.np.ravel将数组格式转化为一维
"""
#————————————————————————————————运动状态实例———————————————————————————————
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer #从sklearn库中导入预处理模块
from sklearn.model_selection import train_test_split #导入自动生成训练集和测试集的模块
from sklearn.metrics import classification_report #导入预测结果评估模块
from sklearn.neighbors import KNeighborsClassifier #KNN分类器
from sklearn.tree import DecisionTreeClassifier  #决策树分类器
from sklearn.naive_bayes import GaussianNB  #高斯贝叶斯
#==================数据导入====================
def load_dataset(feature_paths,label_paths):
    feature=np.ndarray(shape=(0,41))#构造固定结构的数组
    label=np.ndarray(shape=(0,1))
    for file in feature_paths: #指定分隔符为逗号、缺失值为问号且文件不包含表头行
        df=pd.read_table(file,delimiter=',',na_values='?',header=None) #na_values识别为缺失的值
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0) #通过设定strategy参数为‘mean’，使用平均值对缺失数据进行补全 ,并按照列求平均
        imp.fit(df)#训练预处理器
        df=imp.transform(df)#生成预处理结果
        feature=np.concatenate((feature,df)) ##按照行联结数组序列 最终包含所有特征文件
    for file in label_paths:
        df=pd.read_table(file,header=None)
        label=np.concatenate((label,df))
    label=np.ravel(label)#将标签整合为一维向量  np.ravel将数组格式转化为一维
    return feature,label
if _name_=='_main_':
    feature_paths=['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    label_paths=['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    #将前4个数据作为训练集
    x_train,y_train=load_dataset(feature_paths[:4],label_paths[:4])
    #将最后一个数据作为测试集
    x_test,y_test=load_dataset(feature_paths[4:],label_paths[4:])
    x_train,x_,y_train,y_=train_test_split(x_train,y_train,test_size=0) #利用train_test_split函数,将原始训练集顺序打乱,x_,y_都没有数据

#==================主函数KNN====================
print('start training KNN')
knn=KNeighborsClassifier().fit(x_train,y_train) #使用默认参数创建K近邻分类器
print('training done')
answer_knn=knn.predict(x_test)
print('predict done')
#KNN极度消耗时间
#==================主函数决策树====================        
dt=DecisionTreeClassifier().fit(x_train,y_train) 
answer_dt=dt.predict(x_test)     
#==================主函数贝叶斯====================           
gnb=GaussianNB().fit(x_train,y_train)
answer_gnb=gnb.predict(x_test)               
#==================主函数分类结果分析====================        
#使用classification_report函数对分类结果，从精确率precision、召回率recall、 f1值f1-score和支持度support四个维度进行衡量
classification_report(y_test,answer_knn)
classification_report(y_test,answer_dt)
classification_report(y_test,answer_gnb)
