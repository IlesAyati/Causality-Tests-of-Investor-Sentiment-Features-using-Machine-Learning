# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:26:00 2019

@author: yeeya
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#### loading data ############################################
data2        = pd.read_csv('FUT86_18all.txt',sep='\t', header=0)
data2.index  = pd.to_datetime(data2.values[:,2])
data2        = data2.groupby(pd.Grouper(freq='M')).last()
Sdata        = pd.read_csv('SP500returns.csv')
ret          = Sdata.values[:,-2]
level        = np.cumprod(1+ret)
##############################################################

#### quick plot ################
oi           = data2.values[:,7]
plt.plot(data2.index,oi)

#### Slicing COT dataset by variables
nclong       = data2.values[:,8]
ncshort      = data2.values[:,9]
ncspread     = data2.values[:,10]
clong        = data2.values[:,11]
cshort       = data2.values[:,12]
totreplong   = data2.values[:,13]
totrepshort  = data2.values[:,14]
nonreplong   = data2.values[:,15]
nonrepshort  = data2.values[:,16]

#### Creating the Pressure Net Long Variables
NCPNL        = (nclong-ncshort)/(nclong+ncshort)
CPNL         = (clong-cshort)/(clong+cshort)

#### create X matrix
X           = np.array([CPNL,NCPNL]).T
X           = (X[1:,:]/X[:-1,:])-1
# Add constant
X           = sm.add_constant(X)
X           = X.astype(float)


#### OLS REGRESSION #########################################
reg         = sm.OLS(ret[2:],X[:-1])
results     = reg.fit()

print(results.summary()) # summary of regression


# Plot
sns.regplot(X[:-1,2], ret[2:])
############################################################


