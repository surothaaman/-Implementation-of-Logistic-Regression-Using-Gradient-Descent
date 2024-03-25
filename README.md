# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# STEP 1 :

Use the standard libraries in python for finding linear regression.
# STEP 2 :

Set variables for assigning dataset values.
# STEP 3 :

Import linear regression from sklearn.
# STEP 4:

Predict the values of array.
# STEP 5:

Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
# STEP 6 :
Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: premji p
RegisterNumber:212221043004
*/


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:
# Array Value of x :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/031d22ed-b498-4389-b18d-695f7a820d45)

# Array Value of y :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/9beeb3db-0e03-45a1-bf1b-fc7535645b85)

# Exam 1 - score graph :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/c8497a8e-6941-47ef-abb6-60849fe68bf2)

# Sigmoid function graph :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/2e27848e-95c4-4f47-a45e-9422adda0416)

# X_train_grad value :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/42e28d10-d6f2-4aaa-a677-54bfb29c7150)

# Y_train_grad value :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/028f0410-7a7e-4fc1-bb26-4fa751733a06)

# Print res.x :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/8165a862-f168-4878-a469-69169a938340)

# Decision boundary - graph for exam score :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/dd18822d-91e8-4a11-bf83-cabc2152c0e8)

# Proability value :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/4d64293b-e240-44ab-b8d7-92d2acd6762a)

# Prediction value of mean :
![image](https://github.com/chandramohan3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/142579775/b143f71d-a5c5-42dc-a386-d64ad563808c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
