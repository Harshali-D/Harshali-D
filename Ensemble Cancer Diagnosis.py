#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[6]:


df = datasets.load_breast_cancer()
df


# In[9]:


X = pd.DataFrame(columns = df.feature_names, data = df.data)
X.head()


# In[12]:


y = df.target
target = {'target' : df.target}
y = pd.DataFrame(data = target)


# In[14]:


y.value_counts()
y = y['target']
y


# In[18]:


dtc =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn =  KNeighborsClassifier()
xgb = xgboost.XGBClassifier()
clf = [dtc,rfc,knn,xgb]
for algo in clf:
    score = cross_val_score( algo,X,y,cv = 5,scoring = 'accuracy')
    print("The accuracy score of {} is:".format(algo),score.mean())


# In[19]:


dtc =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn =  KNeighborsClassifier()
xgb = xgboost.XGBClassifier()
clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] #list of (str, estimator)
clf


# In[21]:


lr = LogisticRegression()
stack_model = StackingClassifier( estimators = clf,final_estimator = lr)
score = cross_val_score(stack_model,X,y,cv = 5,scoring = 'accuracy')
print("The accuracy score of is:",score) 


# In[ ]:




