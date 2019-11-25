# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:18:08 2019

@author: yeeya
"""
from statsmodels.tsa.vector_ar.var_model1 import VAR
# %% Vector Autoregression ###################################################
#
###### Cointegration test ####################################################
Pdata2Price        = pd.DataFrame(np.cumprod(np.exp(Pdata2.values),axis=0), columns=Pdata2.columns)
FF3Price           = np.cumprod(np.exp(famafrench),axis=0) 
COT2Price          = np.cumprod(np.exp(dfCOT), axis=0)
vix2Price          = np.cumprod(np.exp(vixret)) # same as the actual VIX Index.
data2Price         = pd.concat([cefd,FF3Price,COT2Price,vix2Price], 
                               axis=1)
data2Price.columns = dfall.columns
regdataLevels      = pd.concat([Pdata2Price,data2Price],axis=1)
regdataLevels.index= vixdata.index[1:]
adfullerLVL        = pd.DataFrame()
for i in regdataLevels.columns:
    adfullerLVL[i] = sm.tsa.stattools.adfuller(regdataLevels[i], maxlag=None, regression="ct", autolag='AIC')
# Cointegration test in the levels:
ERCO               = []
for i in range(len(Pdata2Price.columns)):
    for j in range(len(data2Price.columns)):
        ERCO.append(sm.tsa.stattools.coint(Pdata2Price.loc[:,Pdata2Price.columns[i]],
                                           data2Price.loc[:,data2Price.columns[j]],
                                           maxlag=None))
# Variables are NOT cointegrated in the levels --> No long-run dynamics.
# Variables are NOT stationary in the levels --> Differenced VAR. 
##############################################################################
#
#### VAR 1
# Using lagged values of responses and features to test for causality (individual)
var1        = []
varesults1  = []
varesults11 = []
var1_resid  = []
var1_acorr  = []
var1aic     = []
var1yesno   = []
var11yesno  = []
var1pvals   = []
var11pvals  = []
for resp in list_of_responses:
    for exog in notff3:
        var1.append(VAR(regdata[[resp,'ret',exog]], dates=regdata.index).fit(method='ols', maxlags=None, ic='aic', trend='c'))
        varesults1.append(var1[-1].test_causality(caused=resp, causing=exog, kind='f', signif=0.05))
        varesults11.append(var1[-1].test_causality(caused=exog,causing=resp, kind='f', signif=0.05))
        var1aic.append(var1[-1].aic)
        var1pvals.append(varesults1[-1].pvalue)
        var11pvals.append(varesults11[-1].pvalue)
        var1yesno.append(varesults1[-1].conclusion)
        var11yesno.append(varesults11[-1].conclusion)
        var1_resid.append(var1[-1].resid) 
        #var1_acorr.append(var1[len(var1)-1].plot_acorr()) # Autocorrelation plots of residuals
        print(var1[-1].summary())
#
### Mean performance of each feature 
var1pvals   = pd.DataFrame(np.split(np.array(var1pvals), 6), columns=notff3, index=list_of_responses).T.round(4)
print('Mean of p-values per lagged feature = ', var1pvals.mean(axis=1))
var11pvals   = pd.DataFrame(np.split(np.array(var11pvals), 6), columns=list_of_responses, index=notff3).T.round(4)
print('Mean of p-values per lagged response = ', var11pvals.mean(axis=1))

# Adding significance indicator inside dataframes
# Copy of pvalue dataframes: 
aux1pvals = var1pvals
aux11pvals = var11pvals
# creating masks to add indication for significance of the coefficients
# Feat --> Resp
r1 = var1pvals.applymap(lambda x: '{}*'.format(x))
r2 = var1pvals.applymap(lambda x: '{}**'.format(x))
r3 = var1pvals.applymap(lambda x: '{}***'.format(x))
# Apply where appropriate
var1pvals = var1pvals.mask(np.abs(aux1pvals.values)<=0.1,r1)
var1pvals = var1pvals.mask(np.abs(aux1pvals.values)<=0.05,r2)
var1pvals = var1pvals.mask(np.abs(aux1pvals.values)<=0.01,r3)
# Repeat for the other direction: Resp --> Feat
r1 = var11pvals.applymap(lambda x: '{}*'.format(x))
r2 = var11pvals.applymap(lambda x: '{}**'.format(x))
r3 = var11pvals.applymap(lambda x: '{}***'.format(x))
# Apply where appropriate
var11pvals = var11pvals.mask(np.abs(aux11pvals.values)<=0.1,r1)
var11pvals = var11pvals.mask(np.abs(aux11pvals.values)<=0.05,r2)
var11pvals = var11pvals.mask(np.abs(aux11pvals.values)<=0.01,r3)
