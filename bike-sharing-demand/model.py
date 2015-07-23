
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from pandas import DataFrame, read_csv
import pandas as pd
import arrow
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[9]:

train = read_csv('data/train.csv')
train['hour'] = train.datetime.apply(lambda dt: arrow.get(dt).hour)
train['day'] = train.datetime.apply(lambda dt: arrow.get(dt).day)
print(train.head(10))
# print(train.dtypes)


# In[4]:

sns.regplot(x="hour", y="count", data=train, x_jitter=0.25, order=3)


# In[5]:

sns.regplot(x="registered", y="count", data=train)


# In[10]:

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

xtrain0 = train[['season', 'hour', 'holiday', 'workingday', 'humidity', 'windspeed', 'weather', 'temp']]
ytrain = train['count']
mapper = DataFrameMapper([('season', LabelBinarizer()),
                          ('hour', LabelBinarizer()),
                          ('holiday', LabelBinarizer()),
                          ('workingday', LabelBinarizer()),
                          ('humidity', StandardScaler()),
                          ('windspeed', StandardScaler()),
                          ('weather', LabelBinarizer()),
                          ('temp', StandardScaler())])
xtrain1 = mapper.fit_transform(xtrain0)
model = LinearRegression()
model.fit(xtrain1, ytrain)
print(model)
print(model.coef_)


# In[11]:

model.score(xtrain1, ytrain)


# In[12]:

test = read_csv('data/test.csv')
test['hour'] = test.datetime.apply(lambda dt: arrow.get(dt).hour)
test['day'] = test.datetime.apply(lambda dt: arrow.get(dt).day)
print(test.head(10))


# In[26]:

xtest0 = test[['season', 'hour', 'holiday', 'workingday', 'humidity', 'windspeed', 'weather', 'temp']]
mapper = DataFrameMapper([('season', LabelBinarizer()),
                          ('hour', LabelBinarizer()),
                          ('holiday', LabelBinarizer()),
                          ('workingday', LabelBinarizer()),
                          ('humidity', StandardScaler()),
                          ('windspeed', StandardScaler()),
                          ('weather', LabelBinarizer()),
                          ('temp', StandardScaler())])
xtest1 = mapper.fit_transform(xtest0)
ytest = model.predict(xtest1)
submit = DataFrame({'dt':test['datetime'], 'count':ytest})
submit.to_csv('predicted.csv')

