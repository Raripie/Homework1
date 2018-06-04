import numpy as np
import matplotlib.pyplot as plt
from numpy import *



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

fig=plt.figure()
plt.title('Scatter Plots')
ax1=fig.add_subplot(441)
ax1.scatter(class0[:,0].tolist(),class0[:,0].tolist(),c='b',marker='.')
ax1.scatter(class1[:,0].tolist(),class1[:,0].tolist(),c='g',marker='.')
plt.ylabel('Variance')

ax2=fig.add_subplot(442)
ax2.scatter(class0[:,0].tolist(),class0[:,1].tolist(),c='b',marker='.')
ax2.scatter(class1[:,0].tolist(),class1[:,1].tolist(),c='g',marker='.')

ax3=fig.add_subplot(443)
ax3.scatter(class0[:,0].tolist(),class0[:,2].tolist(),c='b',marker='.')
ax3.scatter(class1[:,0].tolist(),class1[:,2].tolist(),c='g',marker='.')

ax4=fig.add_subplot(444)
ax4.scatter(class0[:,0].tolist(),class0[:,3].tolist(),c='b',marker='.')
ax4.scatter(class1[:,0].tolist(),class1[:,3].tolist(),c='g',marker='.')

ax5=fig.add_subplot(445)
ax5.scatter(class0[:,1].tolist(),class0[:,0].tolist(),c='b',marker='.')
ax5.scatter(class1[:,1].tolist(),class1[:,0].tolist(),c='g',marker='.')
plt.ylabel('Skewness')

ax6=fig.add_subplot(446)
ax6.scatter(class0[:,1].tolist(),class0[:,1].tolist(),c='b',marker='.')
ax6.scatter(class1[:,1].tolist(),class1[:,1].tolist(),c='g',marker='.')

ax7=fig.add_subplot(447)
ax7.scatter(class0[:,1].tolist(),class0[:,2].tolist(),c='b',marker='.')
ax7.scatter(class1[:,1].tolist(),class1[:,2].tolist(),c='g',marker='.')

ax8=fig.add_subplot(448)
ax8.scatter(class0[:,1].tolist(),class0[:,3].tolist(),c='b',marker='.')
ax8.scatter(class1[:,1].tolist(),class1[:,3].tolist(),c='g',marker='.')

ax9=fig.add_subplot(449)
ax9.scatter(class0[:,2].tolist(),class0[:,0].tolist(),c='b',marker='.')
ax9.scatter(class1[:,2].tolist(),class1[:,0].tolist(),c='g',marker='.')
plt.ylabel('Curtosis')

ax10=fig.add_subplot(4,4,10)
ax10.scatter(class0[:,2].tolist(),class0[:,1].tolist(),c='b',marker='.')
ax10.scatter(class1[:,2].tolist(),class1[:,1].tolist(),c='g',marker='.')

ax11=fig.add_subplot(4,4,11)
ax11.scatter(class0[:,2].tolist(),class0[:,2].tolist(),c='b',marker='.')
ax11.scatter(class1[:,2].tolist(),class1[:,2].tolist(),c='g',marker='.')

ax12=fig.add_subplot(4,4,12)
ax12.scatter(class0[:,2].tolist(),class0[:,3].tolist(),c='b',marker='.')
ax12.scatter(class1[:,2].tolist(),class1[:,3].tolist(),c='g',marker='.')

ax13=fig.add_subplot(4,4,13)
ax13.scatter(class0[:,3].tolist(),class0[:,0].tolist(),c='b',marker='.')
ax13.scatter(class1[:,3].tolist(),class1[:,0].tolist(),c='g',marker='.')
plt.xlabel('Variance')
plt.ylabel('Entropy')

ax14=fig.add_subplot(4,4,14)
ax14.scatter(class0[:,3].tolist(),class0[:,1].tolist(),c='b',marker='.')
ax14.scatter(class1[:,3].tolist(),class1[:,1].tolist(),c='g',marker='.')
plt.xlabel('Skewness')

ax15=fig.add_subplot(4,4,15)
ax15.scatter(class0[:,3].tolist(),class0[:,2].tolist(),c='b',marker='.')
ax15.scatter(class1[:,3].tolist(),class1[:,2].tolist(),c='g',marker='.')
plt.xlabel('Curtosis')

ax16=fig.add_subplot(4,4,16)
ax16.scatter(class0[:,3].tolist(),class0[:,3].tolist(),c='b',marker='.')
ax16.scatter(class1[:,3].tolist(),class1[:,3].tolist(),c='g',marker='.')
plt.xlabel('Entropy')
plt.show()
