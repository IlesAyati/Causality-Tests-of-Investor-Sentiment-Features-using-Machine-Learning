# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:41:38 2019

@author: yeeya
"""

from pandas.tseries.offsets import *
import statsmodels.api as sm
import numpy as np
import pandas as pd
#from sklearn.tree.export import export_graphviz
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


# Loading data
data        = pd.read_csv('letsdodis.csv')
columns     = data.columns.get_values()
columns     = columns.astype(dtype=str)

data.index  = pd.to_datetime(data.date)
data        = data.groupby(pd.Grouper(freq='M')).mean()

Sdata       = pd.read_csv('SP500returns.csv')
# Convert to np array
npdata      = np.array(data)
Sdata       = np.array(Sdata)

#Remove NaNs
#npdata      = npdata[np.nonzero(~pd.isnull(npdata[:,-1]))]
#Sdata       = Sdata[np.nonzero(~pd.isnull(npdata[:,-1]))]

#extract y = returns and x_i = discounts
ret         = np.array(Sdata[:,-2],dtype=np.float32)
disc        = np.array(npdata[:,-1],dtype=np.float32)
disc        = disc[1:]/disc[:-1]-1
#### SUMMARY ##################
meanret     = np.nanmean(ret)
meandisc    = np.nanmean(disc)

stdret      = np.nanstd(ret)
stddisc     = np.nanstd(disc)
##############################

# Add random x_i variable (preliminary)
rndshk      = np.random.normal(0,stddisc,disc.shape[0])
# create X matrix
X           = np.array([disc,rndshk])
# Add constant
X           = sm.add_constant(X.T)



#### OLS REGRESSION #########
reg         = sm.OLS(ret[2:],X[:-1], missing='drop')
results     = reg.fit()

print(results.summary()) # summary of regression


# Plot
sns.regplot(X[:,1], ret[1:])
############################

#### GLS MODEL #############
ols_resid   = results.resid
res_fit     = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
rho         = res_fit.params

# rho is a consistent estimator of the correlation of the residuals from
#an OLS fit of the longley data.  It is assumed that this is the true rho
#of the AR process data.
from scipy.linalg import toeplitz
nbyn        = np.arange(250,dtype=np.int32)
order       = np.array(toeplitz(nbyn),dtype=np.int32) 
# lag order matrisen i en annen fil og importer heller
sigma       = np.array(rho**order,dtype=np.float32)

#`sigma` is an n x n matrix of the autocorrelation structure of the data.

gls_model   = sm.GLS(ret[2:], X[:-1], missing='drop', sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary())
############################

#### LOGISTIC REGRESSION ###
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#import mglearn
#import graphviz

# Positive returns --> 
retpos  = ret>0
retpos.astype(np.int)
#Remove NaNs
XX      = X[np.nonzero(~pd.isnull(X[:,-2]))]
retpos  = retpos[np.nonzero(~pd.isnull(X[:,-2]))]

split   = np.floor(retpos.shape[0]/2)
X_train = XX[0:split,:]
y_train = retpos[0:split]
X_test  = XX[51:-1,:]
y_test  = retpos[51:-1]

log_reg = LogisticRegression()
logresults = log_reg.fit(X_train, y_train)
logresults.score(X_test,y_test)
###########################

