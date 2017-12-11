# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:45:31 2017

@author: Zzz~L
"""
""" 第25行绘制不同类别的散点图
    第51行设置图像展现方式
    enumerate 在循环中 同时获得索引和值
"""
#——————————————————————————————————降维PCA————————————————————————————————————
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
data=load_iris()#字典的格式导入数据
x=data.data
y=data.target
#*********主函数*********
pca=PCA(n_components=2) #n_components指定主成分个数 #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x) #fit()与fit_transform()的区别，前者仅训练一个模型，没有返回nmf后的分支，而后者除了训练数据，并返回nmf后的分支
#*********主函数*********
red_x,red_y=[],[] #初始化三类数据
blue_x,blue_y=[],[]
green_x,green_y=[],[]
#类别不同的散点图绘制
for i in range(len(reduced_x)): #将三类数据分组
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')#在同一张图上绘制三类散点图
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
#——————————————————————————————降维NMF————————————————————————————————
#==================数据提取==================
from sklearn import decomposition #加载降维算法包
from sklearn.datasets import fetch_olivetti_faces #加载人脸数据集
from numpy.random import RandomState #加载randomstate 创建随机种子
n_row,n_col=2,3  #设置图像展示时的排列情况
n_components=n_row*n_col  #设置提取的特征的数目 该示例中只提取6个特征
image_shape=(64,64)  #设置人脸数据图片的大小
dataset=fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
faces=dataset.data  #400个数据,每个数据是64*64大小
#==================设置图像展现方式================
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) #创建图片,并指定图片大小
    plt.suptitle(title,size=16) #设置标题以及字号大小
    for i,comp in enumerate(images): #enumerate同时获得索引和值
        plt.subplot(n_row,n_col,i+1) #选择画制的子图
        vmax=max(comp.max(),-comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, #以灰度图形显示
                   interpolation='nearest', vmin=-vmax, vmax=vmax) #vmin,vmax标准化数据
        plt.xticks(())#去除子图的坐标轴标签
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.) #对子图位置以及间隔调整
plot_gallery("First centered Olivetti faces", faces[:n_components]) #绘制原图 
#=======================模型训练(主函数)=====================
#PCA作为对比  init：W矩阵和H矩阵的初始化方式，默认为‘nndsvdar’
estimators=[('Eigenfaces - PCA using randomized SVD',decomposition.PCA(n_components=6,whiten=True)),
            ('Non-negative components - NMF',decomposition.NMF(n_components=6,init='nndsvda',tol=5e-3))]
for name,estimator in estimators:
    estimator.fit(faces) ##调用PCA或NMF提取特征
    components=estimator.components_ #获取提取的特征 W矩阵
    plot_gallery(name, components[:n_components]) #按照固定格式进行排列
