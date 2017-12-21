# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:32:46 2017

@author: Zzz~L
"""

#—————————————————————————————————神经网络实现手写识别—————————————————————————————
"""神经网络参数：设置网络的隐藏层数、各隐藏层神经元个数、激活函数、学习率、优化方法、最大迭代次数。
hidden_layer_sizes：设置含100个神经元的隐藏层,其存放的是一个元组，表示第i层隐藏层里神经元的个数
activation：使用logistic激活函数
solver ：adam优化方法；learning_rate：初始学习率为0.0001
    参数设置技巧：1.一般设置较大的最大迭代次数来保证多层感知机能够收敛，达到较高的正确率
    2.较小的学习率一般要配备较大的迭代次数以保证其收敛
"""
import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier
def img2vector(filename):
    retmat=np.zeros(1024,int)
    f=open(filename)
    lines=f.readlines()
    for i in range(32):
        for j in range(32):
            retmat[i*32+j]=lines[i][j]
    return retmat
def readData(path):
    filelist=listdir(path)
    numfile=len(filelist)
    dataset=np.zeros((numfile,1024),int)
    hwlabel=np.zeros((numfile,10))
    for i in range(numfile):
        digit=int(filelist[i].split('_')[0])
        hwlabel[i][digit]=1
        dataset[i]=img2vector(path+'/'+filelist[i])
    return dataset,hwlabel
train_dataset,train_label=readData('D:/python code/machinelearninginaction/Ch02/digits/trainingDigits')            
clf=MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='adam',
                  alpha=0.0001,max_iter=2000)   
clf.fit(train_dataset,train_label)
dataset,label=readData('D:/python code/machinelearninginaction/Ch02/digits/testDigits')
res=clf.predict(dataset)
num=len(dataset)
error=0
for i in range(num):
    if np.sum(res[i]==label[i])<10:
        error += 1
print('total num:',num,'\nwrong num:',error,'\nwrong rate:',error/float(num))
  
#—————————————————————————————————KNN实现手写识别—————————————————————————————  
import numpy as np
from os import listdir
from sklearn import neighbors
def img2vector(filename):
    retmat=np.zeros(1024,int)
    f=open(filename)
    lines=f.readlines()
    for i in range(32):
        for j in range(32):
            retmat[i*32+j]=lines[i][j]
    return retmat
def readData(path):
    filelist=listdir(path)
    numfile=len(filelist)
    dataset=np.zeros((numfile,1024),int)
    hwlabel=np.zeros(numfile)
    for i in range(numfile):
        digit=int(filelist[i].split('_')[0])
        hwlabel[i]=digit
        dataset[i]=img2vector(path+'/'+filelist[i])
    return dataset,hwlabel
train_dataset,train_label=readData('D:/python code/machinelearninginaction/Ch02/digits/trainingDigits')            
knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)
knn.fit(train_dataset,train_label)
dataset,label=readData('D:/python code/machinelearninginaction/Ch02/digits/testDigits')
res=knn.predict(dataset)
error=np.sum(res != label)
num=len(dataset)
print('total num:',num,'\nwrong num:',error,'\nwrong rate:',error/float(num))
 
