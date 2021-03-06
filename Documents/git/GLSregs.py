# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:15:09 2019

@author: yeeya
"""
from scipy.linalg import toeplitz
# %% GLS REGRESSIONS ##########################################################

#### Regression 1 
# Univariate regression of each feature on each stock portfolio
reg1   = []
params1= []
tvals1 = []
resid1 = []
white1ols  = []
acorr1ols  = []
nbyn   = np.arange(299,dtype=np.int32)
order  = np.array(toeplitz(nbyn),dtype=np.int32) 
for resp in list_of_responses:
    for exog in ['cefd','NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret']:
        # Define regression
        formula     = resp + " ~ " + exog
        # Regress exog on response
        reg1.append(sm.OLS.from_formula(formula, data=regdata).fit())
        # Obtain residuals
        resid1.append(reg1[-1].resid)
        # White's test for heterosced
        white1ols.append(sm.stats.diagnostic.het_white(resid1[-1],
                                                 sm.add_constant(regdata[[exog]])))
        # Breusch Godfrey test for autocorr
        acorr1ols.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg1[-1]))
        # Regress residual time series on its lag
        res_fit     = sm.OLS(resid1[-1].values[1:], resid1[-1].values[:-1]).fit()
        # Obtain parameter
        rho         = res_fit.params
        sigma       = np.array(rho**order,dtype=np.float32)
        reg1.pop()
        reg1.append(sm.GLS.from_formula(formula, data=regdata, sigma=sigma).fit())
        resid1.pop()
        resid1.append(reg1[-1].resid)
        params1.append(reg1[-1].params)
        tvals1.append(reg1[-1].tvalues)
        print(reg1[-1].summary())
        
tvals1copy = pd.DataFrame(tvals1.copy())
print('Mean of absolute t-values =', np.abs(tvals1copy).mean(axis=0))
params1copy = pd.DataFrame(params1.copy())

tvals1na    = []
params1na   = [] 
for i in notff3:
    tvals1na.append(tvals1copy[i].dropna().values)
    params1na.append(params1copy[i].dropna().values)
tvals1na  = pd.DataFrame(tvals1na, index=notff3).T.round(4)
params1na = pd.DataFrame(params1na, index=notff3).T.round(4)
# creating masks to add indication for significance of the coefficients
r1 = params1na.applymap(lambda x: '{}*'.format(x))
r2 = params1na.applymap(lambda x: '{}**'.format(x))
r3 = params1na.applymap(lambda x: '{}***'.format(x))
# Apply where appropriate
params1na = params1na.mask(np.abs(tvals1na.values)>1.645,r1)
params1na = params1na.mask(np.abs(tvals1na.values)>1.96,r2)
params1na = params1na.mask(np.abs(tvals1na.values)>2.326,r3)
#%% Regression routine 2 #####################################################
######  Define regdata - The regression dataset with PCs #####################
Pdata2            = pd.DataFrame(Pdata[1::])
Pdata2.index      = Pdata.index[1:]
list_of_responses = ["SMALLLoBM", "ME1BM2", "SMALLHiBM", "BIGLoBM", "ME2BM2", "BIGHiBM"]
Pdata2.columns    = list_of_responses
regdataPC           = pd.concat([Pdata2,dfallPC],axis=1,ignore_index=False)
regdataPC.columns   = np.append(list_of_responses,dfallPC.columns.values)
##############################################################################
# PC regression
# One regression per stock portfolio
reg2         = []
tvals2       = []
rsquared2    = []
params2      = []
resid2       = []
white2ols    = []
acorr2ols    = []
nbyn         = np.arange(len(regdataPC),dtype=np.int32)
order        = np.array(toeplitz(nbyn),dtype=np.int32)
for resp in list_of_responses:
    formula = resp + " ~ PC1 + PC2 + PC3"
    reg2.append(sm.OLS.from_formula(formula, data=regdataPC).fit())
    resid2.append(reg2[-1].resid)
    white2ols.append(sm.stats.diagnostic.het_breuschpagan(resid2[-1], 
                                                          sm.add_constant(regdataPC[['PC1', 'PC2', 'PC3']])))
    acorr2ols.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg2[-1]))
    res_fit     = sm.OLS(resid2[-1].values[1:], resid2[-1].values[:-1]).fit()
    rho         = res_fit.params
    sigma       = np.array(rho**order,dtype=np.float32)
    reg2.pop()
    reg2.append(sm.GLS.from_formula(formula, data=regdataPC, sigma=sigma).fit())
    resid2.pop()
    resid2.append(reg2[-1].resid)
    params2.append(reg2[-1].params)
    tvals2.append(reg2[-1].tvalues)
    rsquared2.append(reg2[-1].rsquared_adj)
    print(reg2[-1].summary())
    
tvals2copy = pd.DataFrame(tvals2.copy())
print('Mean of absolute t-values =', np.abs(tvals2copy).mean(axis=0))
print('Mean adjusted R-squared =', np.array(rsquared2).mean(axis=0))
params2copy = pd.DataFrame(params2.copy(), index=list_of_responses).round(4)
# creating masks to add indication for significance of the coefficients
r1 = params2copy.applymap(lambda x: '{}*'.format(x))
r2 = params2copy.applymap(lambda x: '{}**'.format(x))
r3 = params2copy.applymap(lambda x: '{}***'.format(x))
# Apply where appropriate
params2copy = params2copy.mask(np.abs(tvals2copy.values)>1.645,r1)
params2copy = params2copy.mask(np.abs(tvals2copy.values)>1.96,r2)
params2copy = params2copy.mask(np.abs(tvals2copy.values)>2.326,r3)
##############################################################################
#### Regression 3 ############################################################
# Features regression
# One regression per stock portfolio
reg3      = []
tvals3    = []
params3   = []
rsquared3 = []
resid3    = []
bp3       = []
bG3       = []
bp33      = []
bG33      = []
nbyn      = np.arange(len(regdata),dtype=np.int32)
order     = np.array(toeplitz(nbyn),dtype=np.int32)
for resp in list_of_responses:
    formula = resp + " ~ cefd + NONPNL + CPNL + NCPNL + OI + vixret "
    reg3.append(sm.OLS.from_formula(formula, data=regdata).fit())
    resid3.append(reg3[-1].resid)
    bp3.append(sm.stats.diagnostic.het_white(resid3[-1], 
                                                    sm.add_constant(regdata[notff3])))
    bG3.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg3[-1]))
    res_fit     = sm.OLS(resid3[-1].values[1:], resid3[-1].values[:-1]).fit()
    rho         = res_fit.params
    sigma       = np.array(rho**order,dtype=np.float32)
    reg3.pop()
    reg3.append(sm.GLS.from_formula(formula, data=regdata, sigma=sigma).fit())
    resid3.pop()
    resid3.append(reg3[-1].resid)
    params3.append(reg3[-1].params)
    tvals3.append(reg3[-1].tvalues)
    rsquared3.append(reg3[-1].rsquared_adj)
    print(reg3[-1].summary())
    
tvals3copy = pd.DataFrame(tvals3.copy())
print('Mean of absolute t-values =', np.abs(tvals3copy).mean(axis=0))
print('Mean adjusted R-squared =', np.array(rsquared3).mean(axis=0))
params3copy = pd.DataFrame(params3.copy(), index=list_of_responses).round(4)
# creating masks to add indication for significance of the coefficients
r1 = params3copy.applymap(lambda x: '{}*'.format(x))
r2 = params3copy.applymap(lambda x: '{}**'.format(x))
r3 = params3copy.applymap(lambda x: '{}***'.format(x))
# Apply where appropriate
params3copy = params3copy.mask(np.abs(tvals3copy.values)>1.645,r1)
params3copy = params3copy.mask(np.abs(tvals3copy.values)>1.96,r2)
params3copy = params3copy.mask(np.abs(tvals3copy.values)>2.326,r3)
##############################################################################