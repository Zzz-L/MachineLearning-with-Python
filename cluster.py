# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:20:15 2017

@author: Zzz~L
"""
""" 1.pd.read_table 不能读取文件名为中文的文件
    2.if _name_=='_main_': 之后的程序只有文件作为脚本直接执行才会被执行
    3.np.reshape()更改原始数组的格式 -1表示,Numpy会根据剩下的维度计算出数组的另外一个shape属性值
    4.pd.set_option('display.max_columns', 500) 数据表显示全部 设置
    5.print('Noise raito:',format(raito, '.2%')) 以百分比格式,保留两位小数
    6.metrics.silhouette_score 轮廓系数
    7.np.flatten() flatten转化为1维的数组
    8.print('\t std of sample:',std)  \t 横向制表符(即空格)
"""
#----------------------------k-means聚类-----------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
city=pd.read_table('city.txt',header=None,encoding='gbk',sep=',')#不能读取中文文件名
data=city.iloc[:,1:9]
cityname=city.iloc[:,0]
#一个python的文件有两种使用的方法，第一是直接作为脚本执行，第二是import到其他的python脚本中
#被调用（模块重用）执行。因此if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
#在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，
#而import到其他脚本中是不会被执行的
if _name_=='_main_':
    km=KMeans(n_clusters=3)
    label=km.fit_predict(data)#预测 axis=1按行加和
    expenses = np.sum(km.cluster_centers_,axis=1)#km.cluster_centers_列出每个类中的数据点
    citycluster=[[],[],[]]
    for i in range(len(cityname)):#将城市分为三类***
        citycluster[label[i]].append(cityname[i])
    for i in range(len(citycluster)):#expenses和citycluster顺序为什么一致
        print("Expenses:%.2f" % expenses[i]) #.2表示保留两位小数
        print(citycluster[i])
        
#----------------------------DBSCAN聚类-----------------------    
from sklearn.cluster import DBSCAN
from sklearn import metrics #轮廓系数评价聚类效果,越大越好
import matplotlib.pyplot as plt
f=open('TestData.txt',encoding='utf-8')
mac2id=dict()
onlinetimes=[]
for line in f:
    mac=str(line.split(',')[2])
    onlinetime=int(line.split(',')[6])
    starttime=int(line.split(',')[4].split()[1].split(':')[0])
    mac2id[mac]=len(onlinetimes)#该id在文件的位置
    onlinetimes.append((starttime,onlinetime))#开始上网时间与上网时长
real_x=np.array(onlinetimes).reshape((-1,2))#转化为数组  -1表示,Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
#================开始上网时间的聚类分析===============
np.set_printoptions(threshold=np.Inf) #设置数组显示数目,Inf表示显示边界无穷
#pd.set_option('display.max_columns', 500) #数据表显示全部 设置
X=real_x[:,0:1] #开始上网时间
db=DBSCAN(eps=0.01,min_samples=20).fit(X)#模型训练 eps: 两个样本被看作邻居节点的最大距离  min_samples: 簇的样本数
labels = db.labels_
print('Labels:')
print(labels)
raito=len(labels[labels[:] == -1]) / len(labels) #表示-1的占比,即噪声数据占比
print('Noise raito:',format(raito, '.2%'))#以百分比格式,保留两位小数
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #计算类别个数,当label中有-1时,则减一 
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels)) #metrics.silhouette_score 轮廓系数
for i in range(n_clusters_):
    print('cluster',i,':')
    print(list(X[labels==i].flatten()))#flatten转化为1维的数组
plt.hist(X,24)  
#================上网时长的聚类分析===============    
X=np.log(1+real_x[:,1]).reshape(-1,1)#数据分布不均匀,因此进行log变换,为了防止生成无穷值,所以加1
plt.hist(X)     
db=DBSCAN(eps=0.14,min_samples=10).fit(X)   
labels=db.labels_
print('labels:')
print(labels)
ratio=len(labels[labels[:]==-1])/len(labels) 
print('Noise raito:',format(ratio,'.2%'))
n_clusters_=len(set(labels))-(1 if -1 in labels else 0)   
print('Estimated number of clusters:%d' % n_clusters_)
print('Silhouette Coefficient:%0.3f' % metrics.silhouette_score(X, labels))   
for i in range(n_clusters_):
    print('cluster',i,':')
    count=len(labels[labels==i])
    mean=np.mean(real_x[labels==i,1])
    std=np.std(real_x[labels==i,1])
    print('\t number of sample:',count)
    print('\t mean of sample:',mean)
    print('\t std of sample:',std) #\t 横向制表符
    
        
