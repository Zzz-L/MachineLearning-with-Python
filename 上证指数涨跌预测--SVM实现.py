# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:59:18 2017

@author: Zzz~L
"""

"""知识小结：1.pd.read_csv中parse_dates设置第0列解析为日期 index_col设置用作行索引的列编号
            2.sort_index按照某列排序
            3.svm实现见第40行
"""
#—————————————————————————————上证指数涨跌预测--SVM实现————————————————————————————————————
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
#===================数据获取====================
#tips:无论是数据表,还是数组,其获取数据都是总数据量-1(因为python从0开始计数)
data=pd.read_csv('000777.csv',encoding='gbk',parse_dates=[0],index_col=0)# parse_dates第0列解析为日期 index_col用作行索引的列编号
data.sort_index(0,ascending=True,inplace=True)#sort_index按照某列排序,ascending升序,inplace 替换原始的数据表
dayfeature=150 #选取150天数据
featurenum=5*dayfeature #选取5个特征*天数
#用前150天的数据预测,则实际能预测的样本量为data.shape[0]-dayfeature
#针对每个样本的特征数一共为featurenum+1,+1表示当天的开盘价作为特征数据
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))#构造特征样本
y=np.zeros((data.shape[0]-dayfeature))
for i in range(0,data.shape[0]-dayfeature):
    x[i,:featurenum]=np.array(data[i:i+dayfeature][['收盘价','最高价','最低价','开盘价','成交量']]).reshape((1,featurenum)) #转换为1行
    #将数据中的“收盘价”“最高价”“开盘价”“成交量” 存入x数组中
    x[i,featurenum]=data.ix[i+dayfeature,'开盘价']
    #最后一列记录当日的开盘价
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature,'收盘价']>data.ix[i+dayfeature,'开盘价']:
        y[i]=1
    else:
        y[i]=0
#===================模型训练====================
clf=svm.SVC(kernel='rbf') #调用svm函数 并设置核函数为rbf
result=[]
for i in range(5):
    x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
    clf.fit(x_train,y_train)
    test=clf.predict(x_test)
    result.append(np.mean(y_test==test)) ##np.mean等价于np.sum(y_test==test)/len(test)
print('svm classifier accuacy:')
print(result)
    
            
                
