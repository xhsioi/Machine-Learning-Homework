import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##适用于单变量、多变量的线性回归，其中x始终为0：cols-1，y始终为cols-1：cols，可用于检验模型灵敏度、预测
path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
print(data.head())
print(data.describe())
data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))
plt.show()

def daijiahanshu(x,y,theta):
    num=np.power(((x*theta.T)-y),2)
    return np.sum(num)/(2*len(x))

data.insert(0,'Ones',1)
cols=data.shape[1]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
theta=np.matrix([0,0])
xx=np.matrix(x.values)
yy=np.matrix(y.values)
cost1=daijiahanshu(xx,yy,theta)
alpha=0.01
size=1000

def calculate_prediction(xx,yy,theta,alpha,size):
    theta1=np.matrix(np.zeros(theta.shape))
    cost=np.zeros(size)
    m=xx.shape[0]
    for i in range(size):
        theta1=theta-(alpha/m)*(xx*theta.T-yy).T*xx
        theta=theta1
        cost[i]=daijiahanshu(xx,yy,theta)
    return cost,theta

cost,final_theta=calculate_prediction(xx,yy,theta,alpha,size)
final_cost=daijiahanshu(xx,yy,final_theta)

X=np.linspace(data.Population.min(),data.Population.max(),100)
f=final_theta[0,0]+(final_theta[0,1]*X)

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(X,f,'r',label='Prediction')
ax.scatter(data['Population'],data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(np.arange(size),cost,'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Size')
plt.show()
