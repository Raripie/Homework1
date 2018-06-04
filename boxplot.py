import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
import seaborn as sns

def list_generator(mean,dis,number):
    return np.random.normal(mean,dis*dis,number)

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


#x1=class0[:,0].tolist()
#x2=class1[:,0].tolist()
#for i in range(j-1):
#    y1=x1[i]+x1[i+1]
#for i in range(t-1):
#    y2=x2[i]+x2[i+1]
#A1=pd.Series(x1)
#B1=pd.Series(x2)


#x1=class0[:,1].tolist()
#x2=class1[:,1].tolist()
#for i in range(j-1):
#    y1=x1[i]+x1[i+1]
#for i in range(t-1):
#    y2=x2[i]+x2[i+1]
#A2=pd.Series(y1)
#B2=pd.Series(y2)

#x1=class0[:,2].tolist()
#x2=class1[:,2].tolist()
#for i in range(j-1):
#    y1=x1[i]+x1[i+1]
#for i in range(t-1):
#    y2=x2[i]+x2[i+1]
#A3=pd.Series(y1)
#B3=pd.Series(y2)


#x1=class0[:,3].tolist()
#x2=class1[:,3].tolist()
#for i in range(j-1):
#    y1=x1[i]+x1[i+1]
#for i in range(t-1):
#    y2=x2[i]+x2[i+1]
#A4=pd.Series(y1)
#B4=pd.Series(y2)

#list1 = list_generator(0.8531, 0.0956, 70)
#list2 = list_generator(0.8631, 0.0656, 80)
#s1 = pd.Series(np.array(list1))
#s2 = pd.Series(np.array(list2))



#ax1=pd.DataFrame({"Class 0": A1,"Class 1": B1})
#ax1.boxplot()
#plt.xlabel('Variance of Wavelet Transformed Image')
#plt.title('Box Plot of Variance')
#plt.show()

#ax2=pd.DataFrame({"Class 0": A2,"Class 1": B2})
#ax2.boxplot()
#plt.xlabel('Skewness of Wavelet Transformed Image')
#plt.title('Box Plot of Skewness')
#plt.show()

#ax3=pd.DataFrame({"Class 0": A3,"Class 1": B3})
#ax3.boxplot()
#plt.xlabel('Curtosis of Wavelet Transformed Image')
#plt.title('Box Plot of Curtosis')
#plt.show()

#ax4=pd.DataFrame({"Class 0": A4,"Class 1": B4})
#ax4.boxplot()
#plt.xlabel('Entropy of Image')
#plt.title('Box Plot of Entropy')
#plt.show()

sns.boxplot(data=[class0[:,3],class1[:,3]])  ## change column of class0 and class1 to get different boxplot
plt.xlabel('Entropy of Image')
plt.title('Box Plot of Entropy')
#sns.boxplot(data=class1[:,0])
plt.show()
