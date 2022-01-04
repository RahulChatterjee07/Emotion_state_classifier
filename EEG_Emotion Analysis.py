
# coding: utf-8

# In[45]:


import pandas as pd
import scipy as sc


# # LOAD DATASET
# 

# In[46]:


df = pd.io.parsers.read_csv(filepath_or_buffer = 'High beta.csv', header= 0,
    sep=',')
df.dropna(how="all", inplace=True)
df.head()


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X = df[['Sub1']].values
x =  df[['Sub2']].values 
m = df[['Sub3']].values

p= df[['Sub4']].values
n = df[['Sub5']].values







# # divide data set into a sequence of  discrete segments of 6 seconds and calculating mean value for each of the time-segments.

# In[48]:


list1=[]
for t in range(2,31):
    list1.append(6*t)
    
list1


# In[49]:


c=[]

t=6


for i in list1:
    
    c.append(np.average(p[(t-3):(i-3)]))
    
    t=t+6    
                
c.insert(0,(np.average(p[0:6])))

c


# In[50]:


d=[]
a1=[]
a2=[]
t=6

for i in list1:
    
    d.append(np.average(n[(t-3):(i-3)]))
    
    t=t+6   
                
d.insert(0,(np.average(n[0:6])))

d


# In[51]:


a=[]
a1=[]
a2=[]
t=6

for i in list1:
    
    
    a.append(np.average(x[(t-3):(i-3)]))
    
    t=t+6   
                
a.insert(0,(np.average(x[0:6])))

len(a)




# In[52]:


b=[]
t=6

for i in list1:
    
    
    b.append(np.average(m[t-3:i-3]))
    
    t = t+6

b.insert(0,(np.average(m[0:6])))


# In[53]:


e=[]
t=6

for i in list1:
    
    
    e.append(np.average(X[t-3:i-3]))
    
    t = t+6

e.insert(0,(np.average(X[0:6])))


# In[54]:


y3 = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]
len(y3)


# In[55]:


X5=list(zip(a,b,c,d,e))
X5




# In[56]:


from math import floor
from math import ceil


# # Split data set into train and test data (70:30 ratio)

# In[57]:


X_train, X_test, y_train, y_test = train_test_split(
    X5, y3, test_size=0.30, random_state=42)



# # KNN Implementation

# In[58]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


    



# In[59]:


z=[]
for k in range (1,7):
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train,y_train)
    scores = cross_val_score(knn,X_train,y_train, cv=5)
    k=k+1
    z.append(np.average(scores))
z    


# In[60]:


import matplotlib.pyplot as plt

# empty list that will hold cv scores
cv_scores = []

# perform 5-fold cross validation
for k in range (1,9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())
MSE = [(1 - x) for x in cv_scores]


K=[1,2,3,4,5,6,7,8]

# plot misclassification error vs k
plt.plot(K, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

#print(MSE)


# # SVM Implementation

# In[61]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

svm = SVC(kernel='linear',C=1)

scores = cross_val_score(svm,X_train ,y_train , cv=5)

np.average(scores)


# # LDA Implementation

# In[62]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.io.parsers.read_csv(filepath_or_buffer = 'highbeta_ldafinal.csv', header= 0,
    sep=',')

X,y= df.iloc[:, 0:5].values,df.iloc[:, -1].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

y_train1 = [int(i) for i in y_train]
X


# In[63]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

sklearn_lda = LinearDiscriminantAnalysis(n_components=3)
transf_lda = sklearn_lda.fit_transform(X_train, y_train)

for label,marker,color in zip(range(1,4),('x', 'o', '^'),('red', 'green', 'blue')):
    plt.scatter(x=transf_lda[:,0][y_train == label],
                y=transf_lda[:,1][y_train == label],
                marker=marker, color=color,
                alpha=0.7, label='class {}'.format(label))



plt.legend(loc='lower right')


plt.show()

