# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:47:55 2017

@author: Zzz~L
"""

""" 1.调用sklearn.linear_model.LinearRegression()所需参数：
• fit_intercept : 布尔型参数，表示是否计算该模型截距。可选参数。
• normalize : 布尔型参数，若为True，则X在回归前进行归一化。可选参数。默认值为False。
• copy_X : 布尔型参数，若为True，则X将被复制；否则将被覆盖。 可选参数。默认值为True。
• n_jobs : 整型参数，表示用于计算的作业数量；若为-1，则用所有的CPU。可选参数。默认值为1
    2.fit,fit_transform,transform三者区别:
    fit方法是用于从一个训练集中学习模型参数，其中就包括了归一化时用到的均值，
标准偏差。transform方法就是用于将模型用于位置数据，fit_transform就很高效的将模型
训练和转化合并到一起，训练样本先做fit，得到mean，standard deviation，然后将这些
参数用于transform（归一化训练数据），使得到的训练数据是归一化的，而测试数据只需要
在原先得到的mean，std上来做归一化就行了，所以用transform就行了
"""
#———————————————————————————————————线性回归—————————————————————————————————————
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np
dataset=pd.read_csv('prices.txt',header=None)
length=len(dataset)
dataset_x=np.array(dataset.iloc[:,0]).reshape([length,1]) #自变量需转化为列向量
#将datasets_X转化为数组，并变为二维，以符合线性回归拟合函数输入参数要求。
dataset_y=np.array(dataset.iloc[:,1])
minx=min(dataset_x)
maxx=max(dataset_x)
X=np.arange(minx,maxx).reshape([-1,1])
linear=linear_model.LinearRegression() #*****主函数
linear.fit(dataset_x,dataset_y) #*****主函数
#查看回归方程系数
print('coefficients:',linear.coef_)  #*****主函数
#查看回归方程截距
print('intercept:',linear.intercept_)  #*****主函数
#可视化
plt.scatter(dataset_x,dataset_y,color='blue')
plt.plot(X,linear.predict(X),color='blue')
plt.xlabel('area')
plt.ylabel('price')

#———————————————————————————————————多项式回归—————————————————————————————————————
#多项式回归实际是先将变量X处理成多项式特征，然后使用线性模型进行回归，实质还是线性回归
from sklearn.preprocessing import PolynomialFeatures #导入多项式特征构造模块
poly_reg=PolynomialFeatures(degree=2) #首先设置二次多项式
x_poly=poly_reg.fit_transform(dataset_x) #fit_transform???
lin_reg_2=linear_model.LinearRegression()
lin_reg_2.fit(x_poly,dataset_y)
#可视化
plt.scatter(dataset_x,dataset_y)
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X))) #X仍需转化为二阶
plt.xlabel('area')
plt.ylabel('price')

#—————————————————————————————————岭回归———————————————————————————————————————
"""岭回归参数：  
• alpha：正则化因子，对应于损失函数中的alpha,惩罚函数
• fit_intercept：表示是否计算截距
• solver：设置计算参数的方法，可选参数‘auto’、‘svd’、‘sag’等
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation  #在model_selection中怎么导入
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_table('transport.txt',sep=',',index_col=0)
plt.plot(np.array(data.iloc[:,4]))
x=np.array(data.iloc[:,:4])
y=np.array(data.iloc[:,4])
poly=PolynomialFeatures(degree=6)
X=poly.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
clf=Ridge(alpha=1,fit_intercept=True)
clf.fit(x_train,y_train) #调用fit函数使用训练集训练回归器
clf.score(x_test,y_test) #利用测试集计算回归曲线的拟合优度
##绘制拟合曲线
start=200
end=300
y_pre=clf.predict(X)
time=np.arange(start,end) #生成序列
plt.plot(time,y_pre[start:end],'b',label='predict')
plt.plot(time,y[start:end],'r',label='real')
plt.legend()


