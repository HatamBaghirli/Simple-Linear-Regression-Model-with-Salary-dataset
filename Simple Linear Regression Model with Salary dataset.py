#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
lr=LinearRegression(normalize=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm


# In[2]:


data=pd.read_csv('Salary_Data.csv')


# In[3]:


data.head(10)


# In[4]:


data.isnull()


# In[5]:


data.dtypes


# In[6]:


data.corr()


# In[7]:


data.skew()


# In[8]:


data.describe()


# In[9]:


y=data['Salary']
x=data['YearsExperience']


# In[10]:


d=sm.add_constant(x)
results=sm.OLS(y,x).fit()
results.summary()


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[12]:


print('X train set')
print(x_train)
print('X test set')
print(x_test)
print('Y train set')
print(y_train)
print('Y test set')
print(y_test)


# In[13]:


lr.fit(x_test.values.reshape(-1,1),y_test)


# In[14]:


pred_test=lr.predict(x_test.values.reshape(-1,1))


# In[15]:


pred_test


# In[16]:


lr.fit(x_train.values.reshape(-1,1),y_train)
pred_train=lr.predict(x_train.values.reshape(-1,1))


# In[17]:


pred_train


# In[18]:


lr.fit(x.values.reshape(-1,1),y)
pred=lr.predict(x.values.reshape(-1,1))
pred


# In[19]:


plt.scatter(x,y,color='red')
plt.plot(x,pred,color='blue',linewidth=3)
plt.xticks(())
plt.yticks(())
plt.title('Simple linear regression')
plt.show()


# In[ ]:




