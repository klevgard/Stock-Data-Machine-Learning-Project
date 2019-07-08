#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries

import csv
import pandas as pd
import sys
import os
import re
import pymapd
#import DatetimeIndex
from pymapd import connect
import pyarrow as pa
import sklearn
# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils

# Load numpy
import numpy as np


# In[2]:


#connect to OmniSci database
conn = connect(user="max", password="max", host="localhost", dbname="max")
conn


# In[3]:


c = conn.cursor()


# In[4]:


query_train = "select * from concat_stocks where date_ > '2012-12-31' and date_ < '2015-01-01' order by date_ asc;  "


# In[5]:


query_train = "select * from concat_stocks where date_ > '2012-12-31' and date_ < '2015-01-01' and Symbol = 'aapl' order by date_ asc;  "


# In[6]:


c.execute(query_train)
results = c.fetchall()
train_dta = pd.DataFrame(results)


# In[7]:


query_test = "select * from concat_stocks where date_ > '2015-01-01' and Symbol = 'aapl';  "


# In[8]:


c.execute(query_test)
results = c.fetchall()
test_dta = pd.DataFrame(results)


# In[9]:


test_dta.columns = ['open_','high','low',
        'close_',
        'volume',
        'openint',
        'rsi',
        'wr_14',
        'wr' ,
        'rsv_9' ,
        'kdjk_9' ,
        'kdjk'  ,
        'kdjd_9',
        'kdjd',
        'kdjj_9',
        'kdjj',
        'close_12_ema',
        'close_26_ema',
        'macd',
        'macds',
        'macdh',
        'open_2r',
        'proc',
        'date_',
        'Symbol']


# In[10]:


test_dta = test_dta.copy()
test_dta['Year_'] = pd.DatetimeIndex(test_dta['date_']).year
test_dta['Month_'] = pd.DatetimeIndex(test_dta['date_']).month
test_dta['Day_'] = pd.DatetimeIndex(test_dta['date_']).dayofweek


# In[11]:


test_dta['Data_lagged'] = (test_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(3))


# In[12]:


test_dta = test_dta.dropna(subset=['Data_lagged'])


# In[13]:


test_dta['Data_lagged5'] = (test_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(5))


# In[14]:


test_dta = test_dta.dropna(subset=['Data_lagged5'])


# In[15]:


test_dta['Data_lagged7'] = (test_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(7))


# In[16]:


test_dta = test_dta.dropna(subset=['Data_lagged7'])


# In[17]:


test_dta.shape


# In[18]:


test_dta.head


# In[19]:


clf = RandomForestClassifier(n_jobs=12, random_state=0)


# In[20]:


train_dta.columns = ['open_','high','low',
        'close_',
        'volume',
        'openint',
        'rsi',
        'wr_14',
        'wr' ,
        'rsv_9' ,
        'kdjk_9' ,
        'kdjk'  ,
        'kdjd_9',
        'kdjd',
        'kdjj_9',
        'kdjj',
        'close_12_ema',
        'close_26_ema',
        'macd',
        'macds',
        'macdh',
        'open_2r',
        'proc',
        'date_',
        'Symbol']


# In[21]:


train_dta = train_dta.copy()
train_dta['Year_'] = pd.DatetimeIndex(train_dta['date_']).year
train_dta['Month_'] = pd.DatetimeIndex(train_dta['date_']).month
train_dta['Day_'] = pd.DatetimeIndex(train_dta['date_']).dayofweek


# In[22]:


train_dta['Data_lagged'] = (train_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(3))


# In[23]:


train_dta = train_dta.dropna(subset=['Data_lagged'])


# In[24]:


train_dta['Data_lagged5'] = (train_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(5))


# In[25]:


train_dta = train_dta.dropna(subset=['Data_lagged5'])


# In[26]:


train_dta['Data_lagged7'] = (train_dta.sort_values(by=['date_'], ascending=True)
                       .groupby(['Symbol'])['close_'].shift(7))


# In[27]:


train_dta = train_dta.dropna(subset=['Data_lagged7'])


# In[28]:


x = train_dta[train_dta['Symbol'] == 'aapl'].head(5)
x.head


# In[29]:


prediction_vars = ['open_','high','close_','low','rsi','wr','close_12_ema','macd','Month_','Day_','Year_','Symbol']


# In[30]:


y = ['Data_lagged']


# In[31]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in train_dta.columns:
    train_dta[i] = le.fit_transform(train_dta[i])


# In[32]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in test_dta.columns:
    test_dta[i] = le.fit_transform(test_dta[i])


# In[33]:


for i in y:
    train_dta[i] = le.fit_transform(train_dta[i])


# In[34]:


train_dta.dtypes


# In[35]:


test_dta.dtypes


# In[38]:


clf.fit(train_dta,train_dta[prediction_vars])


# In[39]:


rf_predictions = clf.predict(test_dta)


# In[46]:


rf_predictions


# In[44]:


import matplotlib.pyplot as plt
plt.plot(rf_predictions)


# In[ ]:



rf_probs = clf.predict_proba(test_dta)


# In[42]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(train_dta, test_dta))  
print('Mean Squared Error:', metrics.mean_squared_error(train_dta, test_dta))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_dta, test_dta)))  


# In[ ]:




