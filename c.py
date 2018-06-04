import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def varying_num_of_k_KNN(x_test,y_test,x_train,y_train):
    neighbors=np.arange(1,902,3)
    train_error=np.empty(len(neighbors))
    test_error=np.empty(len(neighbors))
    for i,k in enumerate(neighbors):
        knn=KNeighborsClassifier(n_neighbors=k)
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

def KNN(x_test,y_test,x_train,y_train,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_train)
    labels=y_train.tolist()
    conf_mat=confusion_matrix(y_train,y_pred,labels=None)
    print("confusion matrix:")
    print(conf_mat)
    print("True Positive:0.422")
    print("True Negetive:0.578")
    print("report:")
    print(classification_report(y_train,y_pred))

def varying_of_training_set_KNN(x_test,y_test,train0,train1):
    NUM=np.arange(50,901,50)
    n=np.empty(len(NUM))
    for N in enumerate(NUM):
        train=mat(zeros((N,5)))
        train=np.row_stack((train0[N/2,5],train1[N/2,5]))
        x_train=mat(zeros((N,4)))
        y_train=mat(zeros((N,1)))
        for i in range(4):
            x_train[:,i]=train[:,i]
        y_train[:,0]=train[:,4]
        y_train=y_train.ravel()
        y_train=y_train.T
        neighbors=np.arange(1,N+1,40)
        train_error=np.empty(len(neighbors))
        test_error=np.empty(len(neighbors))
        for i,k in enumerate(neighbors):
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            train_error[i]=1-knn.score(x_train,y_train)
            test_error[i]=1-knn.score(x_test,y_test)
        n[N]=min(test_error)
    plt.title('Learning Curve')
    plt.plot(NUM,n)
    plt.legend()
    plt.xlabel('Number of Training set')
    plt.ylabel('Optimal K')
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



#neighbors=np.arange(1,902,3)
#train_error=np.empty(len(neighbors))
#test_error=np.empty(len(neighbors))
#for i,k in enumerate(neighbors):
#    knn=KNeighborsClassifier(n_neighbors=k)
#    knn.fit(x_train,y_train)
#    train_error[i]=1-knn.score(x_train,y_train)
#    test_error[i]=1-knn.score(x_test,y_test)
#plt.title('Varying Number of Neighbors')
#plt.plot(1/neighbors,test_error,label='Testing Error')
#plt.plot(1/neighbors,train_error,label='Training Error')
#plt.legend()
#plt.xlabel('1/(Number of Neighbors)')
#plt.ylabel('Error')
#plt.show()

#knn=KNeighborsClassifier(n_neighbors=500)
#knn.fit(x_train,y_train)
#y_pred=knn.predict(x_train)
#labels=y_train.tolist()
#conf_mat=confusion_matrix(y_train,y_pred,labels=None)
#print("confusion matrix:")
#print(conf_mat)
#print("True Positive:0.422")
#print("True Negetive:0.578")
#print("report:")
#print(classification_report(y_train,y_pred))



NUM=np.arange(50,801,50)
n=np.empty(len(NUM))
for M,N in enumerate(NUM):
    train=mat(zeros((N,5)))
    N1=int(N/2)
    train00=mat(zeros((N1,5)))
    train11=mat(zeros((N1,5)))
    for i in range(N1):
        train00[i,:]=train0[i,:]
        train11[i,:]=train1[i,:]
    train=np.row_stack((train00,train11))
    x_train=mat(zeros((N,4)))
    y_train=mat(zeros((N,1)))
    for i in range(4):
        x_train[:,i]=train[:,i]
    y_train[:,0]=train[:,4]
    y_train=y_train.ravel()
    y_train=y_train.T
    neighbors=np.arange(1,N+1,40)
    train_error=np.empty(len(neighbors))
    test_error=np.empty(len(neighbors))
    for i,k in enumerate(neighbors):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        train_error[i]=1-knn.score(x_train,y_train)
        test_error[i]=1-knn.score(x_test,y_test)
    n[M]=test_error.tolist().index(min(test_error))+1
plt.title('Learning Curve')
plt.plot(NUM,n)
plt.legend()
plt.xlabel('Number of Training set')
plt.ylabel('Optimal K')
plt.show()


