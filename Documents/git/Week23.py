# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:36:32 2019

@author: yeeya
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
#from sklearn.tree.export import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import toeplitz
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# %% Loading data ############################################################
data         = pd.read_csv('./Data/letsdodis3ll.csv')
data.index   = pd.to_datetime(data.date)
data2        = pd.read_csv('./Data/FUT86_18.txt',sep='\t', header=0)
data2.index  = pd.to_datetime(data2['Dates'], dayfirst=True)
data2        = data2.groupby(pd.Grouper(freq='M')).last()
Sdata        = pd.read_csv('./Data/FamaFrench5Factors2.csv')
Sdata.index  = data.index
Pdata        = pd.read_csv('./Data/6_Portfolios_2x3_Wout_Div.csv', index_col=0)/100
Pdata.index  = data.index
vixdata      = pd.read_csv('./Data/VIX94_18.csv', header=1)
vixdata.index= pd.to_datetime(vixdata.values[:,-1])
vixdata      = vixdata.groupby(pd.Grouper(freq='M')).last()

# Each month of the CEFD data represents the mean of discounts across funds
# Each month of the COT data represents the mean aggregate figure of the month
# COT reports are published weekly. 
# Each month of the SP500 data is the monthly return (value-weighted index)
##############################################################################
###############################################################################
# %% Data Management #########################################################

#   Fama French 3 factors
ret          = np.array(Sdata.values[:,1],dtype=float) 
ret = ret[1:]
smb          = np.array(Sdata.values[:,2],dtype=float)
smb = smb[1:]
hml          = np.array(Sdata.values[:,3],dtype=float)
hml = hml[1:]
rf           = np.array(Sdata.values[:,4],dtype=float)

famafrench   = pd.DataFrame([ret, smb, hml]).T


####  VIX data
vixret       = vixdata[['VIX Close']]
vixret       = pd.DataFrame(np.log(vixret.values[1:]/vixret.values[:-1]))

#### Slicing COT dataset by variables
oi           = data2['Open Interest (All)'] # Open Interest
nclong       = data2['Noncommercial Positions-Long (All)'] # Non-commercial long
ncshort      = data2['Noncommercial Positions-Short (All)'] # -----||------- short
ncspread     = data2['Noncommercial Positions-Spreading (All)'] # -----||-------spread
clong        = data2['Commercial Positions-Long (All)'] # Commercial long
cshort       = data2['Commercial Positions-Short (All)'] # Commercial short
totreplong   = data2[' Total Reportable Positions-Long (All)'] # Total reportables long
totrepshort  = data2['Total Reportable Positions-Short (All)'] # Total reportables short
nonreplong   = data2['Nonreportable Positions-Long (All)'] # Total Non-reportables long
nonrepshort  = data2['Nonreportable Positions-Short (All)'] # Total Non-reportables short

#   Creating the Pressure Net Long Variables

NCPNL        = nclong/ncshort # Non-Commercial Pressure Net
NCPNL        = np.log(NCPNL.values[1:]/NCPNL.values[:-1])  # Change in NCPNL
CPNL         = clong/cshort # Commercial Pressure Net Long
CPNL         = np.log(CPNL.values[1:]/CPNL.values[:-1])  # Change in CPNL
TOTPNL       = totreplong/totrepshort #TotRep PNL
TOTPNL       = np.log(TOTPNL.values[1:]/TOTPNL.values[:-1]) # Change in TotRep PNL
NONPNL       = nonreplong/nonrepshort # Non-Reportable Pressure Net Long
NONPNL       = np.log(NONPNL.values[1:]/NONPNL.values[:-1]) # Change in NCPNL
OI           = np.log(oi.values[1:]/oi.values[:-1])

####    DataFraming COT measures 
dfCOT        = pd.concat([pd.DataFrame(NONPNL),pd.DataFrame(CPNL),pd.DataFrame(NCPNL),pd.DataFrame(OI)],axis=1)
dfCOT.columns= range(4)


####    CEFD data
a            = data.navm.interpolate('linear').diff()
b            = data.prc.diff()

cefd         = a-b
cefd         = pd.DataFrame(cefd.values[1:])

####    Dickey Fuller test for unit root in variables ########################
adfuller        = pd.DataFrame()
dfall           = pd.concat([famafrench, cefd, dfCOT, vixret],axis=1)
dfall.columns   = np.arange(9)
for i in dfall.columns:
    adfuller[i] = sm.tsa.stattools.adfuller(dfall[i], regression="c", autolag='AIC')
    dfall[i]    = np.where(abs(dfall[i])>=5*np.std(dfall[i]), np.nan, dfall[i])
    dfall       = dfall.ffill()

adfuller.columns= ['ret','smb','hml','cefd','NONPNL','CPNL','NCPNL','OI','vixret']
dfall.columns   = adfuller.columns
dfall.index     = Sdata.index[1:]
# Stationarity assumption holds.
ff3             = ['ret', 'smb', 'hml']
notff3          = ['cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret']
sixcolors       = ['darkcyan', 'teal', 'seagreen' , 'mediumseagreen' , 'lightseagreen' , 'mediumaquamarine' ]

dfall[notff3].corr()
# There seems to be multicollinearity in the features. Let's extract the PCs
smPC        = sm.PCA(dfall, standardize=1, method='svd')
smPCcorr    = smPC.scores.corr()
dfallPC     = smPC.scores

#!! MULTIVARIATE REGRESSIONS MUST USE PCS
# %% ## Print section of data
"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(len(notff3),sharex=True)
fig.set_figheight(6)
fig.set_figwidth(8)
fig.set_label("")
#fig.suptitle('Timeseries - All features', fontsize=12)   
for exog, i, color in zip(notff3,range(len(notff3)), sixcolors):
    dfall[exog].plot(ax=axs[i], color=[color], legend=exog)
    axs[i].legend(loc='lower left')
    axs[i].set(xlabel="")
fig.savefig('./Figures/FeaturesSeries.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

## Export tables to latex
with open('dfCOT.tex','w') as tf:
    tf.write(dfCOT.head().to_latex(float_format='%.3f'))

## Export tables to latex
with open('dfall[notff3].tex','w') as tf:
    tf.write(dfall[notff3].head().to_latex(float_format='%.3f'))
    
## Export tables to latex
with open('Pdata2.tex','w') as tf:
    tf.write(Pdata2.head().to_latex(float_format='%.3f'))
    
## Export tables to latex
with open('params1na.tex','w') as tf:
    tf.write(params1na.to_latex(float_format='%.4f'))
""" 
##############################################################################
##############################################################################    
# %% OLS REGRESSIONS ##########################################################

######  Define regdata - The regression dataset ##############################
Pdata2            = pd.DataFrame(Pdata[1::])
Pdata2.index      = Pdata.index[1:]
list_of_responses = ["SMALLLoBM", "ME1BM2", "SMALLHiBM", "BIGLoBM", "ME2BM2", "BIGHiBM"]
Pdata2.columns    = list_of_responses
regdata           = pd.concat([Pdata2,dfall],axis=1,ignore_index=False)
regdata.columns   = np.append(list_of_responses,dfall.columns.values)
#############################################################################

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
        formula     = resp + " ~ " + exog
        reg1.append(sm.OLS.from_formula(formula, data=regdata).fit())
        #print(reg1[len(reg1)-1].summary())
        resid1.append(reg1[len(reg1)-1].resid)
        white1ols.append(sm.stats.diagnostic.het_white(resid1[len(reg1)-1],
                                                 sm.add_constant(regdata[[exog]])))
        acorr1ols.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg1[len(reg1)-1]))
        res_fit     = sm.OLS(resid1[len(reg1)-1].values[1:], resid1[len(reg1)-1].values[:-1]).fit()
        rho         = res_fit.params
        sigma       = np.array(rho**order,dtype=np.float32)
        reg1.pop()
        reg1.append(sm.GLS.from_formula(formula, data=regdata, sigma=sigma).fit())
        resid1.pop()
        resid1.append(reg1[len(reg1)-1].resid)
        params1.append(reg1[len(reg1)-1].params)
        tvals1.append(reg1[len(reg1)-1].tvalues)
        print(reg1[len(reg1)-1].summary())
        
tvals1copy = pd.DataFrame(tvals1.copy())
print('Mean of absolute t-values =', np.abs(tvals1copy).mean(axis=0))
params1copy = pd.DataFrame(params1.copy())

tvals1na    = []
params1na   = [] 
for i in notff3:
    tvals1na.append(tvals1copy[i].dropna().values)
    params1na.append(params1copy[i].dropna().values)
tvals1na  = pd.DataFrame(tvals1na, index=notff3).T
params1na = pd.DataFrame(params1na, index=notff3).T
"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals1copy[notff3].plot.barh(width=0.8, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stat Overview for Univariate regressions}', fontsize=11)
plt.xlabel(r'\textbf{T-statistic}', fontsize=10)
#plt.ylabel(r'\textbf{Regression}', fontsize=10)
plt.yticks(ticks=np.arange(0, 35, step=6), labels=list_of_responses)
reg1plot = plt.gcf()
reg1plot.set_figwidth(8)
reg1plot.set_figheight(5)
reg1plot.savefig('./Figures/reg1plot.pdf')
plt.show()
"""

######  Define regdata - The regression dataset with PCs #####################
Pdata2            = pd.DataFrame(Pdata[1::])
Pdata2.index      = Pdata.index[1:]
list_of_responses = ["SMALLLoBM", "ME1BM2", "SMALLHiBM", "BIGLoBM", "ME2BM2", "BIGHiBM"]
Pdata2.columns    = list_of_responses
regdataPC           = pd.concat([Pdata2,dfallPC],axis=1,ignore_index=False)
regdataPC.columns   = np.append(list_of_responses,dfall.columns.values)
#############################################################################

#### Regression 2
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
    formula = resp + " ~ NONPNL + CPNL + NCPNL + OI"
    reg2.append(sm.OLS.from_formula(formula, data=regdataPC).fit())
    resid2.append(reg2[len(reg2)-1].resid)
    white2ols.append(sm.stats.diagnostic.het_breuschpagan(resid2[len(reg2)-1], 
                                                          sm.add_constant(regdataPC[['NONPNL','CPNL','NCPNL', 'OI']])))
    acorr2ols.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg2[len(reg2)-1]))
    res_fit     = sm.OLS(resid2[len(reg2)-1].values[1:], resid2[len(reg2)-1].values[:-1]).fit()
    rho         = res_fit.params
    sigma       = np.array(rho**order,dtype=np.float32)
    reg2.pop()
    reg2.append(sm.GLS.from_formula(formula, data=regdataPC, sigma=sigma).fit())
    resid2.pop()
    resid2.append(reg2[len(reg2)-1].resid)
    params2.append(reg2[len(reg2)-1].params)
    tvals2.append(reg2[len(reg2)-1].tvalues)
    rsquared2.append(reg2[len(reg2)-1].rsquared)
    print(reg2[len(reg2)-1].summary())
    
tvals2copy = pd.DataFrame(tvals2.copy())
print('Mean of absolute t-values =', np.abs(tvals2copy).mean(axis=0))
params2copy = pd.DataFrame(params2.copy())

"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals2copy[['NONPNL','CPNL','NCPNL', 'OI']].plot.barh(width=0.5, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stats - Multivariate regressions: COT}', fontsize=11)
plt.xlabel(r'\textbf{T-stat}', fontsize=10)
#plt.ylabel(r'\textbf{Regression}', fontsize=10)
plt.yticks(ticks=range(6), labels=list_of_responses)
reg2plot = plt.gcf()
reg2plot.savefig('./Figures/reg2plot.pdf')
plt.show()
"""

#### Regression 3
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
nbyn      = np.arange(len(regdataPC),dtype=np.int32)
order     = np.array(toeplitz(nbyn),dtype=np.int32)
for resp in list_of_responses:
    formula = resp + " ~ cefd + NONPNL + CPNL + NCPNL + OI + vixret "
    reg3.append(sm.OLS.from_formula(formula, data=regdataPC).fit())
    resid3.append(reg3[len(reg3)-1].resid)
    bp3.append(sm.stats.diagnostic.het_white(resid3[len(reg3)-1], 
                                                    sm.add_constant(regdataPC[notff3])))
    bG3.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg3[len(reg3)-1]))
    res_fit     = sm.OLS(resid3[len(reg3)-1].values[1:], resid3[len(reg3)-1].values[:-1]).fit()
    rho         = res_fit.params
    sigma       = np.array(rho**order,dtype=np.float32)
    reg3.pop()
    reg3.append(sm.GLS.from_formula(formula, data=regdataPC, sigma=sigma).fit())
    resid3.pop()
    resid3.append(reg3[len(reg3)-1].resid)
    params3.append(reg3[len(reg3)-1].params)
    tvals3.append(reg3[len(reg3)-1].tvalues)
    rsquared3.append(reg3[len(reg3)-1].rsquared)
    print(reg3[len(reg3)-1].summary())
    
tvals3copy = pd.DataFrame(tvals3.copy())
print('Mean of absolute t-values =', np.abs(tvals3copy).mean(axis=0))
params3copy = pd.DataFrame(params3.copy())

"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals3copy[notff3].plot.barh(width=0.5, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stats - Multivariate regressions: All features}', fontsize=11)
plt.xlabel(r'\textbf{T-stat}', fontsize=10)
plt.yticks(ticks=range(6), labels=list_of_responses)
reg3plot = plt.gcf()
reg3plot.savefig('./Figures/reg3plot.pdf')
plt.show()
"""

#       Plots
#sns.regplot(X[:,3],ret[1:])
##############################################################################
##############################################################################

# %% Vector Autoregression ###################################################

from statsmodels.tsa.vector_ar.var_model1 import VAR

#### VAR 1
# Using lagged values of responses and regressors to test for individual causality

var1        = []
varesults1  = []
varesults11 = []
var1_acorr  = []
var1tvals   = []
var1yesno   = []
var11yesno  = []
for resp in list_of_responses:
    for exog in notff3:
        var1.append(VAR(regdata[[resp,'ret',exog]],dates=regdata.index).fit(maxlags=None, ic='aic', trend='c'))
        varesults1.append(var1[len(var1)-1].test_causality(caused=resp, causing=[resp,exog], kind='f', signif=0.1))
        varesults11.append(var1[len(var1)-1].test_causality(caused=resp,causing=resp, kind='f', signif=0.1))
        var1tvals.append(np.array(var1[len(var1)-1].tvalues))
        var1yesno.append(varesults1[len(var1)-1].conclusion)
        var11yesno.append(varesults11[len(var1)-1].conclusion)
        var1_acorr.append(var1[len(var1)-1].acorr)
        print(var1[len(var1)-1].summary())

### Plot results
plt.figure()
plt.hist(var1yesno, 3, alpha=0.5, label='With')
plt.hist(var11yesno, 3 , alpha=0.5, label= 'Without')
plt.legend()


#### VAR 2
# Using lagged values of responses and regressors to test for COT causality

var2        = []
varesults2  = []
var2_acorr   = []
varesults2summary = [] 
var2yesno   = []
for resp in list_of_responses:
    var2.append(VAR(regdata[[resp,'NONPNL','CPNL','NCPNL', 'OI']],dates=regdata.index, freq='M').fit(maxlags=12, method='ols', ic='aic'))
    varesults2.append(var2[len(var2)-1].test_causality(resp, causing=['NONPNL','CPNL','NCPNL','OI'], kind='f', signif=0.10))
    varesults2summary.append(varesults2[len(var2)-1].summary())
    var2_acorr.append(var2[len(var2)-1].acorr())
    var2yesno.append(varesults2[len(var2)-1].conclusion)
    print(varesults2[len(var2)-1].summary())
plt.hist(var2yesno,3)
plt.hist(var22yesno,3)
plt.bar(np.arange(0,len(var2)),height=var2tvals)

#### VAR 3
# Using lagged values of responses and regressors to test for joint causality

var3        = []
varesults3  = []
varesults33 = []
var_acorr3  = []
var3tvals   = []
var3yesno   = []
var33yesno  = []
for resp in list_of_responses:
    var3.append(VAR(regdata[[resp, 'cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret']],dates=data.index[1:], freq='M').fit(maxlags=12, ic='aic'))
    varesults3.append(var3[len(var3)-1].test_causality(resp, causing=['cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret'], kind='f', signif=0.10))
    varesults33.append(var3[len(var3)-1].test_causality(['cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret'], causing=resp, kind='f', signif=0.10))
    var3tvals.append(np.array(var3[len(var3)-1].tvalues)[-1][0])
    var3yesno.append(varesults3[len(var3)-1].conclusion)
    var33yesno.append(varesults33[len(var3)-1].conclusion)
    #var_acorr.append(var1[len(var1)-1].plot_sample_acorr())
    print(var3[len(var3)-1].summary())
### PLOT THIS
plt.hist(var3yesno,3)
plt.hist(var33yesno,3)
plt.bar(np.arange(0,len(var3)),height=var3tvals)
    
# %% LOGIT ####################################################################

# Transform abnormal returns into logical
Pdata.columns     = list_of_responses
Pdata3            = pd.DataFrame(Pdata[1::])
Pdata3.index      = regdata.index
retpos            = Pdata3.abs() > pd.core.window.Expanding(Pdata3,axis=0).std().bfill()
retpos            = retpos.astype(int)
retpos            = pd.DataFrame(retpos)
retpos.columns    = list_of_responses
regdata2          = pd.concat([retpos,dfall],axis=1,ignore_index=False)

# X defined as [ones, cefd, ret]
# One regression per stock portfolio
regL1  = []
tvalsL1 = []
for resp in list_of_responses:
    for exog in ['cefd','NONPNL', 'CPNL', 'NCPNL', 'vixret']:
        formula = resp + " ~ " + exog
        regL1.append(sm.Logit.from_formula(formula, data=regdata2).fit())
        print(regL1[len(regL1)-1].summary2())
        print(regL1[len(regL1)-1].get_margeff(at='overall').summary())
        tvalsL1.append(regL1[len(regL1)-1].get_margeff(at='overall').tvalues)
tvalsL1copy = np.array(tvalsL1).reshape(-6,6)
print('Mean of pvalues mfx =', np.abs(tvalsL1copy).mean(axis=0))
sns.regplot(regdata2[['cefd']],regdata2[[resp]], logistic=1)

# X defined as [ones, dfCOT]
# One regression per stock portfolio
regL2  = []
tvalsL2 = []
for resp in list_of_responses:
    formula = resp + " ~ NCPNL + CPNL"
    regL2.append(sm.Logit.from_formula(formula, data=regdata2).fit())
    print(regL2[len(regL2)-1].summary2())
    print(regL2[len(regL2)-1].get_margeff(at='overall').summary())
    tvalsL2.append(regL2[len(regL2)-1].get_margeff(at='overall').tvalues)
tvalsL2copy = np.array(tvalsL2).reshape(-6,2)
print('Mean of pvalues mfx =', np.abs(tvalsL2copy).mean(axis=0))

# X defined as [ones, cefd, dfCOT, vixret, ret]
# One regression per stock portfolio
regL3  = []
tvalsL3 = []
for resp in list_of_responses:
    formula = resp + " ~ cefd + CPNL + NCPNL + vixret "
    regL3.append(sm.Logit.from_formula(formula, data=regdata2).fit())
    print(regL3[len(regL3)-1].summary2())
    print(regL3[len(regL3)-1].get_margeff(at='overall').summary())
    tvalsL3.append(regL3[len(regL3)-1].get_margeff(at='overall').tvalues)
tvalsL3copy = np.array(tvalsL3).reshape(-6,4)
print('Mean of pvalues mfx =', np.abs(tvalsL3copy).mean(axis=0))
"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(logitres2.summary2(yname='retpos', 
                         xname=['const','cefd','CPNL','NCPNL','OI','TOTPNL','vixret'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logitres2.png')
"""
"""
plt.rc('figure', figsize=(7, 3))
plt.text(0.01, 0.05, str(logit2mfx.summary()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logit2mfx.png')
"""


##############################################################################
##############################################################################

# %% MACHINE LEARNING  ######################################################
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV, LeavePOut
from scipy.stats import ttest_ind, f_oneway                       
#from sklearn.feature_selection import f_regression
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.tree.export import export_graphviz
#from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.model_selection import TimeSeriesSplit
#import mglearn
#import graphviz
from timeit import default_timer as timer

######  Regdata with lagged values #########################################
dflag             = pd.DataFrame(np.roll(dfall,1,axis=0), 
                                            index=dfall.index, 
                                            columns=dfall.columns)
dada              = pd.concat([Pdata2, 
                               pd.DataFrame(np.roll(Pdata2,1,axis=0), 
                                            index=dfall.index, 
                                            columns=notff3)], axis=1)
regdata           = pd.concat([dada,dflag],axis=1,ignore_index=False).drop(index=dfall.index[0])

######             ######################################################## 
retpos            = Pdata3.abs() > pd.core.window.Expanding(Pdata3,axis=0).std().bfill()
retpos            = pd.DataFrame(retpos.astype(int))
retpos            = pd.concat([retpos, pd.DataFrame(np.roll(retpos,1,axis=0),
                                                    index=dfall.index,
                                                    columns=notff3)], axis=1)
regdata2          = pd.concat([retpos,dflag],axis=1,ignore_index=False).drop(index=dfall.index[0])
###########################################################################

# Define splits. The first 250 observations are used, expanded by one for each iteration
trainmin   = 200
trainmax   = len(regdata.values)

tsplit     = TimeSeriesSplit(n_splits=10, max_train_size=200)
scaler     = StandardScaler()
lin_reg    = []
linresults = []
causality  = []
log_reg    = []
logresults = []
y_predLin  = []
y_predLog  = []
axLin      = []
axLog      = []
logitparams= {'C': [2, 1, 0.5, 0.2, 0.1],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[:i], regdata[notff3].iloc[i:i+1]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2 = regdata[list_of_responses].iloc[:i], regdata[list_of_responses].iloc[i:i+1]
    # Scaling
    scalefit    = scaler.fit(X_train) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_test.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    for resp in list_of_responses:
        model      = {'OLS': LinearRegression(fit_intercept=True, normalize=False).fit(X_trainS,y_train2[resp]),
                      'LOGIT': GridSearchCV(LogisticRegression(fit_intercept=True, max_iter=500), 
                                            param_grid=logitparams, 
                                            cv=tsplit.split(X_trainS.index), iid=True).fit(X_trainS,y_train1[resp]).best_estimator_}
        ## OLS
        lin_reg.append(model['OLS'])
        y_predLin.append(lin_reg[len(lin_reg)-1].predict(X_testS))
        axLin.append(np.array([y_test2[resp], y_predLin[len(lin_reg)-1]]).T)
        ## LOGIT
        log_reg.append(model['LOGIT'])
        y_predLog.append(log_reg[len(log_reg)-1].predict(X_testS))
        #logresults.append(accuracy_score(y_test1[resp], y_predLog[len(log_reg)-1]))
        axLog.append(np.array([y_test1[resp], y_predLog[len(log_reg)-1]]).T)
t1 = timer()
print(t1-t0)

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predited y_test
y_predLin  = np.array(axLin).squeeze().reshape(trainmax-trainmin,12)
y_predLog  = np.array(axLog).squeeze().reshape(trainmax-trainmin,12)
linresults = []
axLin      = []
linresid   = []
logresults = []
axLog      = []
# Get the results
for i in range(0,y_predLin.shape[1]-1,2):
    linresults.append(r2_score(y_predLin[1:,i],y_predLin[:-1,i+1]))
    axLin.append(np.array([y_predLin[1:,i],y_predLin[:-1,i+1]]).T)
    #linresid.append(axLin[len(axLin)-1].T[0]-axLin[len(axLin)-1].T[1]) # Heteroscedasticity? 
    logresults.append(accuracy_score(y_predLog[1:,i],y_predLog[:-1,i+1]))
    axLog.append(np.array([y_predLog[1:,i],y_predLog[:-1,i+1]]).T)


## Plot results, LinearRegression
plt.bar(np.arange(0,6,step=1), height=linresults, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

## Plot results, LogisticRegression
plt.bar(np.arange(0,6,step=1), height=logresults, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

##############################################################################
##############################################################################

# %% PRINCIPAL COMPONENTS ANALYSIS ###########################################

# First, split the data, then scale, then apply SVD.
# Second, test significance of extracted pcs (LinearRegression)
# Third, perform GLS using extracted pcs
lin_reg2     = []
linresults2  = []
log_reg2     = []
logresults2  = []
y_predLin2   = []
y_predLog2   = []
axLin2       = []
axLog2       = []
X_trainS     = []
X_testS      = []
regPC        = []
tvalsPC      = []
residPC      = []
bpPC         = []
bGPC         = []
bpPC2        = []
bGPC2        = []
cv2          = []
regdataPC    = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[:i], regdata[notff3].iloc[i:i+1]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2 = regdata[list_of_responses].iloc[:i], regdata[list_of_responses].iloc[i:i+1]
    # Scaling
    scalefit    = scaler.fit(X_train[notff3]) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_train.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    # Apply PCA transformation
    pca         = PCA(n_components=0.99, whiten=1)
    pca.fit(X_trainS) # Fit
    X_pca1      = pca.transform(X_trainS) # transform
    X_pca2      = pca.transform(X_testS) # transform
    X_pca1      = pd.DataFrame(X_pca1, index=X_train.index, columns=['PC' + str(i+1) for i in range(0,X_pca1.shape[1])])
    X_pca2      = pd.DataFrame(X_pca2, index=X_test.index, columns=X_pca1.columns)
    regdataPC.append(X_pca2.values)
    for resp in list_of_responses:
        model      = {'OLS': LinearRegression(fit_intercept=True, normalize=False).fit(X_pca1, y_train2[resp]),
                      'LOGIT': GridSearchCV(LogisticRegression(fit_intercept=True, max_iter=500), 
                                            param_grid=logitparams, 
                                            cv=tsplit.split(X_trainS.index), iid=True).fit(X_pca1, y_train1[resp]).best_estimator_}
        ## OLS pcs
        lin_reg2.append(model['OLS'])
        y_predLin2.append(lin_reg2[len(lin_reg2)-1].predict(X_pca2))
        axLin2.append(np.array([y_test2[resp], y_predLin2[len(lin_reg2)-1]]).T)
        ## LOGIT pcs
        log_reg2.append(model['LOGIT'])
        y_predLog2.append(log_reg2[len(log_reg2)-1].predict(X_pca2))
        axLog2.append(np.array([y_test1[resp], y_predLog2[len(log_reg2)-1]]).T)
t1 = timer()
print(t1-t0)
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predited y_test
y_predLin2  = np.array(axLin2).squeeze().reshape(trainmax-trainmin,12)
y_predLog2  = np.array(axLog2).squeeze().reshape(trainmax-trainmin,12)
linresults2 = []
axLin2      = []
logresults2 = []
axLog2      = []
# Get the results
for i in range(0,y_predLin2.shape[1]-1,2):
    linresults2.append(r2_score(y_predLin2[1:,i],y_predLin2[:-1,i+1]))
    axLin2.append(np.array([y_predLin2[1:,i],y_predLin2[:-1,i+1]]).T)
    logresults2.append(accuracy_score(y_predLog2[1:,i],y_predLog2[:-1,i+1]))
    axLog2.append(np.array([y_predLog2[1:,i],y_predLog2[:-1,i+1]]).T)

plt.bar(np.arange(0,6,step=1), height=linresults2, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

plt.bar(np.arange(0,6,step=1), height=logresults2, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

####### GLS USING THE PCS  ##############################################

regdataPC   = pd.DataFrame(np.array(regdataPC).squeeze(), columns=X_pca2.columns)
regdataPC   = 
nbyn        = np.arange(len(test_index),dtype=np.int32)
order       = np.array(toeplitz(nbyn),dtype=np.int32) 
formula     = resp + " ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6"
regPC.append(sm.OLS.from_formula(formula, data=regdataPC).fit())
residPC.append(regPC[len(regPC)-1].resid)
bpPC.append(sm.stats.diagnostic.het_white(residPC[len(regPC)-1], 
                                          sm.add_constant(regdataPC)))
bGPC.append(sm.stats.diagnostic.acorr_breusch_godfrey(regPC[len(regPC)-1]))
res_fit     = sm.OLS(residPC[len(regPC)-1].values[1:], residPC[len(regPC)-1].values[:-1]).fit()
rho         = res_fit.params
sigma       = np.array(rho**order,dtype=np.float32)
regPC.pop()
regPC.append(sm.GLS.from_formula(formula, data=regdataPC, sigma=sigma).fit())
residPC.pop()
residPC.append(regPC[len(regPC)-1].resid)
bpPC2.append(sm.stats.diagnostic.het_white(residPC[len(regPC)-1], 
                                           sm.add_constant(regdataPC)))
bGPC2.append(sm.stats.diagnostic.acorr_breusch_godfrey(regPC[len(regPC)-1]))
tvalsPC.append(regPC[len(regPC)-1].tvalues)


############################################################################
############################################################################

#%%     Plotting Routine   ####################################################
plt.bar(range(logitres3.params.shape[0]-1),pca.explained_variance_ratio_)
plt.title('PCA: Explained variance ratio')
plt.xlabel('Principal Component')
plt.ylabel('EVR')

sns.regplot(X[:,1],retpos, logistic=1)
sns.regplot(X_pca3.values[:,2],retpos, logistic=1)

""" PC VS VARIABLES SCATTER PLOTS
plt.scatter(X_pca3.values[:,1],X[:,3], s=5, c='b')
plt.scatter(X_pca3.values[:,2],X[:,3], s=5, c='r')
plt.xlim(-2,2)
plt.ylim(-5,5)
plt.title('PC1 and PC2 relation to NCPNL')
plt.xlabel('Principal Components')
plt.ylabel('NCPNL')
plt.legend(['PC1','PC2'])
sns.regplot(X_pca3.values[:,2],X[:,1])
sns.regplot(X[:-1,1], retpos[1:], logistic=1)
"""
""" LOGIT TABLE
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(logitres3.summary2()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logitrespca.png')
"""
""" LOGIT MFX TABLE
plt.rc('figure', figsize=(7, 3))
plt.text(0.01, 0.05, str(logit3mfx.summary()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logit3mfx.png')
"""
""" OLS TABLE
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results4.summary(yname='Pret',
                                          xname=['const','pc1','pc2','pc3',
                                          'pc4','pc5'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('results4.png')
"""

#%%    FEATURE SELECTION    #############################

# FEATURE SELECTION STARTS BELOW
# First two models are Ridge and Lasso.

#%%    RIDGE AND LASSO  REGRESSIONS  #############################
lasso_reg       = []
y_predLinlasso  = []
axLinlasso      = []
ridge_reg       = []
y_predLinridge  = []
axLinridge      = []
lasso_params    = {'alpha':[0.001, 0.002, 0.003, 0.005, 0.01]}
ridge_params    = {'alpha':[0,0.1,0.5,1,2,5],
                   'solver': ['svd','lsqr','saga']}

t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[:i], regdata[notff3].iloc[i:i+1]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2 = regdata[list_of_responses].iloc[:i], regdata[list_of_responses].iloc[i:i+1]
    # Scaling
    scalefit    = scaler.fit(X_train) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_test.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    for resp in list_of_responses:
        model        = {'Lasso': GridSearchCV(Lasso(fit_intercept=True, normalize=False), 
                                      param_grid=lasso_params, 
                                      scoring='r2', 
                                      return_train_score=True, 
                                      cv=tsplit.split(X_trainS.index), iid=True).fit(X_trainS, y_train2[resp]).best_estimator_, 
                        'Ridge': GridSearchCV(Ridge(fit_intercept=True, normalize=False), 
                                      param_grid=ridge_params, 
                                      scoring='r2', 
                                      return_train_score=True, 
                                      cv=tsplit.split(X_trainS.index), iid=True).fit(X_trainS, y_train2[resp]).best_estimator_}
        ## RIDGE
        ridge_reg.append(model['Ridge'])
        y_predLinridge.append(ridge_reg[len(ridge_reg)-1].predict(X_testS))
        axLinridge.append(np.array([y_test2[resp], y_predLinridge[len(ridge_reg)-1]]).T)
        ## LASSO
        lasso_reg.append(model['Lasso'])
        y_predLinlasso.append(lasso_reg[len(lasso_reg)-1].predict(X_testS))
        axLinlasso.append(np.array([y_test2[resp], y_predLinlasso[len(lasso_reg)-1]]).T)
t1 = timer()
print(t1-t0)

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinridge  = np.array(axLinridge).squeeze().reshape(trainmax-trainmin,12)
y_predLinlasso  = np.array(axLinlasso).squeeze().reshape(trainmax-trainmin,12)
ridgeresults    = []
axLinridge      = []
lassoresults    = []
axLinlasso      = []
# Get the results
for i in range(0,y_predLinridge.shape[1]-1,2):
    ridgeresults.append(r2_score(y_predLinridge[1:,i],y_predLinridge[:-1,i+1]))
    axLinridge.append(np.array([y_predLinridge[1:,i],y_predLinridge[:-1,i+1]]).T)
    lassoresults.append(r2_score(y_predLinlasso[1:,i],y_predLinlasso[:-1,i+1]))
    axLinlasso.append(np.array([y_predLinlasso[1:,i],y_predLinlasso[:-1,i+1]]).T)

## Plot results, Ridge regression
plt.bar(np.arange(0,6,step=1), height=ridgeresults, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

## Plot results, Lasso regression
plt.bar(np.arange(0,6,step=1), height=lassoresults, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')
###############################################################################


#%% RIDGE AND LASSO REGRESSIONS WITH PCS #####################################

lasso_reg2       = []
y_predLinlasso2  = []
axLinlasso2      = []
ridge_reg2       = []
y_predLinridge2  = []
axLinridge2      = []
lasso_params2    = {'alpha':[0.001, 0.002, 0.003, 0.005, 0.01]}
ridge_params2    = {'alpha':[0,0.1,0.5,1,2,5],
                   'solver': ['svd','lsqr','saga']}
regdataPC2       = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[:i], regdata[notff3].iloc[i:i+1]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2 = regdata[list_of_responses].iloc[:i], regdata[list_of_responses].iloc[i:i+1]
    # Scaling
    scalefit    = scaler.fit(X_train) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_test.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    # Apply PCA transformation
    pca         = PCA(n_components=0.99, whiten=1)
    pca.fit(X_trainS) # Fit
    X_pca1      = pca.transform(X_trainS) # transform
    X_pca2      = pca.transform(X_testS) # transform
    X_pca1      = pd.DataFrame(X_pca1, index=X_train.index, columns=['PC' + str(i+1) for i in range(0,X_pca1.shape[1])])
    X_pca2      = pd.DataFrame(X_pca2, index=X_test.index, columns=X_pca1.columns)
    regdataPC2.append(X_pca2.values)
    for resp in list_of_responses:
        model        = {'Lasso': GridSearchCV(Lasso(fit_intercept=True, normalize=False, max_iter=1000), 
                                      param_grid=lasso_params2, 
                                      scoring='r2', 
                                      return_train_score=True,
                                      cv=tsplit.split(X_trainS.index), iid=True).fit(X_pca1, y_train2[resp]).best_estimator_, 
                        'Ridge': GridSearchCV(Ridge(fit_intercept=True, normalize=False, max_iter=1000), 
                                      param_grid=ridge_params2, 
                                      scoring='r2', 
                                      return_train_score=True,
                                      cv=tsplit.split(X_trainS.index), iid=True).fit(X_pca1, y_train2[resp]).best_estimator_}
        ## RIDGE
        ridge_reg2.append(model['Ridge'])
        y_predLinridge2.append(ridge_reg2[len(ridge_reg2)-1].predict(X_pca2))
        axLinridge2.append(np.array([y_test2[resp], y_predLinridge2[len(ridge_reg2)-1]]).T)
        ## LASSO
        lasso_reg2.append(model['Lasso'])
        y_predLinlasso2.append(lasso_reg2[len(lasso_reg2)-1].predict(X_pca2))
        axLinlasso2.append(np.array([y_test2[resp], y_predLinlasso2[len(lasso_reg2)-1]]).T)
t1 = timer()
print(t1-t0)

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinridge2  = np.array(axLinridge2).squeeze().reshape(trainmax-trainmin,12)
y_predLinlasso2  = np.array(axLinlasso2).squeeze().reshape(trainmax-trainmin,12)
ridgeresults2    = []
axLinridge2      = []
lassoresults2    = []
axLinlasso2      = []
# Get the results
for i in range(0,y_predLinridge2.shape[1]-1,2):
    ridgeresults2.append(r2_score(np.roll(y_predLinridge2[:,i],1,axis=0),y_predLinridge2[:,i+1]))
    axLinridge2.append(np.array([np.roll(y_predLinridge2[:,i],1,axis=0),y_predLinridge2[:,i+1]]).T)
    lassoresults2.append(r2_score(np.roll(y_predLinlasso2[:,i],1,axis=0),y_predLinlasso2[:,i+1]))
    axLinlasso2.append(np.array([np.roll(y_predLinlasso2[:,i],1,axis=0),y_predLinlasso2[:,i+1]]).T)

## Plot results, Ridge regression with pcs
plt.bar(np.arange(0,6,step=1), height=ridgeresults2, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')

## Plot results, Lasso regression with pcs
plt.bar(np.arange(0,6,step=1), height=lassoresults2, align='edge')
plt.xticks(ticks=np.arange(0,6,step=1))
plt.grid(b=None,axis='x')
###############################################################################


#%% RANDOM FOREST FEATURE SELECTION ###########################################

# Defining grid for Gridsearch cross validation

n_estimators      = [200]

#[int(x) for x in np.linspace(start = 50, stop = 200, num = 3)]

max_features      = ['auto'] # Number of features to consider at every split

max_depth         = [10, None] # Maximum number of levels in tree

#[int(x) for x in np.linspace(10, 100, num = 3)]
#max_depth.append(None)

min_samples_split = [2,10,20] # Minimum number of samples required to split a node

min_samples_leaf  = [2,10,20] # Minimum number of samples required at each leaf node
# Method of selecting samples for training each tree
bootstrap         = [True] # Create the random grid

random_grid       = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

# Define Random Forest Regressor as estimator for regression
RFR              = RandomForestRegressor() 
# Define Random Forest Classifier as estimator for classification
RFC              = RandomForestClassifier()

##### Simple Cross-Validation with no expanding window. 


## RANDOM FOREST ####
forest_reg       = []
axRFR            = []
oob_score_reg    = []
ypredforest      = []

forest_clas      = []
axRFC            = []
ypredforestclas  = []
oob_score_clas   = []

t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[train_index], regdata[notff3].iloc[test_index]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[train_index], regdata2[list_of_responses].iloc[test_index]
    y_train2, y_test2 = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
    # Scaling
    scalefit    = scaler.fit(X_train[notff3]) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_test.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    for resp in list_of_responses:
        # Set GridSearchCV 
        model   = {'RFR': GridSearchCV(RFR, param_grid=random_grid, scoring='r2',
                                       return_train_score=True, cv=tsplit.split(X_trainS.index),
                                       iid=True, n_jobs=2).fit(X_trainS, y_train2[resp]).best_estimator_,
                   'RFC': GridSearchCV(RFC, param_grid=random_grid, scoring='accuracy',
                                        return_train_score=True, cv=tsplit.split(X_trainS.index),
                                        iid=True, n_jobs=2).fit(X_trainS, y_train1[resp]).best_estimator_}
        ## Random Forest Regression
        forest_reg.append(model['RFR'])
        ypredforest.append(forest_reg[len(forest_reg)-1].predict(X_testS))
        axRFR.append(np.array([y_test2[resp], ypredforest[len(forest_reg)-1]]).T)
       
        ## Random Forest Classification
        forest_clas.append(model['RFC'])
        ypredforestclas.append(forest_clas[len(forest_clas)-1].predict(X_testS))
        axRFC.append(np.array([y_test1[resp], ypredforestclas[len(forest_clas)-1]]).T)
t1 = timer()
print(t1-t0)

en = []

for i in range(len(axRFR)):
    en.append(r2_score(axRFR[i][1:,0],axRFR[i][:-1,1]))

"""
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforest     = np.array(axRFR).squeeze().reshape(trainmax-trainmin,12)
ypredforestclas = np.array(axRFC).squeeze().reshape(trainmax-trainmin,12)
forestresults   = []
axRFR           = []
forestacc       = []
axRFC           = []
for i in range(0,ypredforest.shape[1]-1,2):
    forestresults.append(r2_score(np.roll(ypredforest[:,i],1,axis=0),ypredforest[:,i+1]))
    axRFR.append(np.array([np.roll(ypredforest[:,i],1,axis=0),ypredforest[:,i+1]]).T)
    forestacc.append(accuracy_score(np.roll(ypredforestclas[:,i],1,axis=0),ypredforestclas[:,i+1]))
    axRFC.append(np.array([np.roll(ypredforestclas[:,i],1,axis=0),ypredforestclas[:,i+1]]).T)
"""    
plt.barh(range(forest_reg[-1].n_features_), forest_reg[-1].feature_importances_, align='center')
plt.yticks(np.arange(n_features), notff3)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


#%% Random Forest with PCs

# Define Random Forest Classifier as estimator for classification
model           = {'RFR': GridSearchCV(RFR, param_grid=random_grid, scoring='r2',
                                       return_train_score=True, cv=tsplit.split(X_trainS.index),
                                       iid=True, n_jobs=1).fit(X_pca1, y_train2[resp]).best_estimator_,
                   'RFC': GridSearchCV(RFC, param_grid=random_grid, scoring='accuracy',
                                        return_train_score=True, cv=tsplit.split(X_trainS.index),
                                        iid=True, n_jobs=1).fit(X_pca1, y_train1[resp]).best_estimator_}

## RANDOM FOREST ####
forest_reg2       = []
axRFR2            = []
oob_score_reg    = []
ypredforest2      = []

forest_clas2      = []
axRFC2            = []
ypredforestclas2  = []
oob_score_clas   = []

t0 = timer()
for train_index,test_index in tsplit.split(X_trainS.index):
    #print(train_index,test_index)
    X_train, X_test   = regdata[notff3].iloc[train_index], regdata[notff3].iloc[test_index]
    y_train1, y_test1 = regdata2[list_of_responses].iloc[train_index], regdata2[list_of_responses].iloc[test_index]
    y_train2, y_test2 = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
    # Scaling
    scalefit    = scaler.fit(X_train[notff3]) # Standardized variables
    X_trainS    = pd.DataFrame(scalefit.transform(X_train), columns=X_train.columns,index=X_train.index)
    X_testS     = pd.DataFrame(scalefit.transform(X_test), columns=X_test.columns,index=X_test.index)
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
    # Apply PCA transformation
    pca         = PCA(n_components=0.99, whiten=1)
    pca.fit(X_trainS) # Fit
    X_pca1      = pca.transform(X_trainS) # transform
    X_pca2      = pca.transform(X_testS) # transform
    X_pca1      = pd.DataFrame(X_pca1, index=X_train.index, columns=['PC' + str(i+1) for i in range(0,X_pca1.shape[1])])
    X_pca2      = pd.DataFrame(X_pca2, index=X_test.index, columns=X_pca1.columns)
    for resp in list_of_responses:
        # Set GridSearchCV 

        ## Random Forest Regression
        forest_reg2.append(model['RFR'])
        ypredforest2.append(forest_reg2[len(forest_reg2)-1].predict(X_pca2))
        axRFR2.append(np.array([y_test2[resp], ypredforest2[len(forest_reg2)-1]]).T)
       
        ## Random Forest Classification
        forest_clas2.append(model['RFC'])
        ypredforestclas2.append(forest_clas2[len(forest_clas2)-1].predict(X_pca2))
        axRFC2.append(np.array([y_test1[resp], ypredforestclas2[len(forest_clas2)-1]]).T)
t1 = timer()
print(t1-t0)

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforest2     = np.array(axRFR2).squeeze().reshape(trainmax-trainmin,12)
ypredforestclas2 = np.array(axRFC2).squeeze().reshape(trainmax-trainmin,12)
forestresults   = []
axRFR2           = []
forestacc       = []
axRFC2           = []
for i in range(0,ypredforest2.shape[1]-1,2):
    forestresults.append(r2_score(np.roll(ypredforest2[:,i],1,axis=0),ypredforest2[:,i+1]))
    axRFR2.append(np.array([np.roll(ypredforest2[:,i],1,axis=0),ypredforest2[:,i+1]]).T)
    forestacc.append(accuracy_score(np.roll(ypredforestclas2[:,i],1,axis=0),ypredforestclas2[:,i+1]))
    axRFC2.append(np.array([np.roll(ypredforestclas2[:,i],1,axis=0),ypredforestclas2[:,i+1]]).T)

#%% Compare results ###########################################

np.mean(linresults)
np.mean(linresults2)
np.mean(ridgeresults)
np.mean(ridgeresults2)
np.mean(lassoresults)
np.mean(lassoresults2)
np.mean(forestresults)
np.mean(forestresults2)

