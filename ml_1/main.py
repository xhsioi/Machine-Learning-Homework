import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
print(data)
print(data.describe())

data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))
plt.show()

#计算代价函数
def computeCost(x,y,theta):
    inner=np.power(((x*theta.T)-y),2)
    return np.sum(inner)/(2*len(x))

data.insert(0,'Ones',1) #第一列增加
cols=data.shape[1] #获得data的列数
X=data.iloc[:,0:cols-1] #取前cols-1列，输入向量
Y=data.iloc[:,cols-1:cols] #取最后一列，目标向量

print(X)
print(Y) #判断是否正确

x=np.matrix(X.values)
y=np.matrix(Y.values)
theta=np.matrix([0,0])
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,epoch):
    temp=np.matrix(np.zeros(theta.shape)) #构建temp矩阵，和theta大小相同
    parameters = int(theta.flatten().shape[1])
    cost=np.zeros(epoch) #储存代价结果
    m=x.shape[0]
    for i in range(epoch):
        temp=theta-(alpha/m)*(x*theta.T-y).T*x
        theta=temp
        cost[i]=computeCost(x,y,theta)
    return theta,cost

alpha=0.01
epoch=1000
final_theta,cost=gradientDescent(x,y,theta,alpha,epoch)
computeCost(x,y,final_theta)

X=np.linspace(data.Population.min(),data.Population.max(),100)
f=final_theta[0,0]+(final_theta[0,1]*x)

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data['Population'],data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show();

fig,ax=plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Trsining Epoch')
plt.show()

#多变量
path='ex1data2.txt'
data2=pd.read_csv(path,names=['Size','Bedrooms','Price'])
cols2=data2.shape[1]
temp=data2.iloc[:,cols2-1:cols]
data2=(data2-data2.mean())/data2.std()
data2.iloc[:,cols2-1:cols]=temp
print(data2)

#重复单变量的操作
data2.insert(0,'Ones',1)
x2=data2.iloc[:,0:cols2-1]
y2=data2.iloc[:,cols2-1:cols2]
x2=np.matrix(x2.values)
y2=np.matrix(y2.values)
theta2=np.matrix([0,0])

final_theta2,cost2=gradientDescent(x2,y2,theta2,alpha,epoch)
computeCost(x,y,final_theta2)

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch),cost2,'r')
ax.set_xlabel('Interation')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x,y)

x=np.array(x2[:,1])
f=model.predict(x2).flatten()

fig,ax=plt.subplots(figsize=(8,5))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_title('Predicted Profit vs. Population Size')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
plt.show()