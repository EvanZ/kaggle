__author__ = 'evanzamir'

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from pandas import DataFrame, read_csv
import pandas as pd
import arrow
import seaborn as sns


train = read_csv('data/train.csv')
train['hour'] = train.datetime.apply(lambda dt: arrow.get(dt).hour)
train['day'] = train.datetime.apply(lambda dt: arrow.get(dt).day)
print(train.head(30))
print(train.dtypes)
# g = sns.factorplot('season','count','hour',train,kind="box")
# g.savefig('plot')

print(train.cov())