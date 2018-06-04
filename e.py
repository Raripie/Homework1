import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def minkowski_varying_num_of_k_KNN(x_test,y_test,x_train,y_train):
    neighbors=np.arange(1,902,10)
    train_error=np.empty(len(neighbors))
    test_error=np.empty(len(neighbors))
    for i,k in enumerate(neighbors):
        knn=KNeighborsClassifier(n_neighbors=k,p=1,metric='minkowski')
        knn.fit(x_train,y_train)
        train_error[i]=1-knn.score(x_train,y_train)
        test_error[i]=1-knn.score(x_test,y_test)
    plt.title('Varying Number of Neighbors')
    plt.plot(1/neighbors,test_error,label='Testing Error')
    plt.plot(1/neighbors,train_error,label='Training Error')
    plt.legend()
    plt.xlabel('1/(Number of Neighbors)')
    plt.ylabel('Error')
    plt.show()


def minkowski_varying_num_of_p_KNN(x_test,y_test,x_train,y_train):
    neighbors=np.arange(0.1,1.1,0.1)
    train_error=np.empty(len(neighbors))
    test_error=np.empty(len(neighbors))
    for i,k in enumerate(pow(10,neighbors)):
        knn=KNeighborsClassifier(n_neighbors=1,p=k,metric='minkowski')
        knn.fit(x_train,y_train)
        train_error[i]=1-knn.score(x_train,y_train)
        test_error[i]=1-knn.score(x_test,y_test)
    plt.title('Varying Number of Neighbors')
    plt.plot(neighbors,test_error,label='Testing Error')
    plt.plot(neighbors,train_error,label='Training Error')
    plt.legend()
    plt.xlabel('lg(p)')
    plt.ylabel('Error')
    plt.show()



        

f=open(r"C:\Users\samsung\Desktop\EE559\h1\banknote.txt")
first_ele=True
for data in f.readlines():
    data=data.strip('\n')
    nums=data.split(",")
    if first_ele:
        nums=[float(x) for x in nums]
        matrix=np.array(nums)
        first_ele=False
    else:
        nums=[float(x) for x in nums]
        matrix=np.c_[matrix,nums]
f.close()
matrix.shape=(5,1372)
m0=mat(zeros((1372,5)))
m1=mat(zeros((1372,5)))
matrix1=matrix.T
j=0
t=0
for i in range(1372):
    if matrix1[i,4]==0:
        m0[j,:]=matrix1[i,:]
        j=j+1
    else:
        m1[t,:]=matrix1[i,:]
        t=t+1

class0=mat(zeros((j,5)))
class1=mat(zeros((t,5)))
for i in range(j):
    class0[i,:]=m0[i,:]
for i in range(t):
    class1[i,:]=m1[i,:]

test0=mat(zeros((200,5)))
test1=mat(zeros((200,5)))
train0=mat(zeros((j-200,5)))
train1=mat(zeros((t-200,5)))
for i in range(200):
    test0[i,:]=class0[i,:]
    test1[i,:]=class1[i,:]
m=0
n=0
for i in range(200,j):
    train0[m,:]=class0[i,:]
    m=m+1
for i in range(200,t):
    train1[n,:]=class1[i,:]
    n=n+1

test=mat(zeros((400,5)))
train=mat(zeros((j+t-400,5)))
test=np.row_stack((test0,test1))
train=np.row_stack((train0,train1))
x_test=mat(zeros((400,4)))
y_test=mat(zeros((400,1)))
x_train=mat(zeros((j+t-400,4)))
y_train=mat(zeros((j+t-400,1)))

for i in range(4):
    x_test[:,i]=test[:,i]
    x_train[:,i]=train[:,i]
y_test[:,0]=test[:,4]
y_train[:,0]=train[:,4]
y_test=y_test.ravel()
y_train=y_train.ravel()
y_test=y_test.T
y_train=y_train.T



neighbors=np.arange(1,902,10)
train_error=np.empty(len(neighbors))
test_error=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',weights='distance',p=1)    ##change value of p to change metrics
    knn.fit(x_train,y_train)
    train_error[i]=1-knn.score(x_train,y_train)
    test_error[i]=1-knn.score(x_test,y_test)
plt.title('Varying Number of Neighbors')
plt.plot(1/neighbors,test_error,label='Testing Error')
plt.plot(1/neighbors,train_error,label='Training Error')
plt.legend()
plt.xlabel('Manhattan')
plt.ylabel('Error')
plt.show()

