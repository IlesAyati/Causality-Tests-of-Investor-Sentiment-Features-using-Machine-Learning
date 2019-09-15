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
data         = pd.read_csv('./Data/letsdodis3.csv')
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
dfCOT           = pd.concat([pd.DataFrame(NONPNL),pd.DataFrame(CPNL),pd.DataFrame(NCPNL),pd.DataFrame(OI)],axis=1)
dfCOT.columns= range(4)


####    CEFD data
cefd            = pd.DataFrame(data.dscnt.interpolate('linear').values[1:])

####    Dickey Fuller test for unit root in variables ########################
adfuller        = pd.DataFrame()
dfall           = pd.concat([famafrench, cefd, dfCOT, vixret],axis=1)
dfall.columns   = np.arange(9)
for i in dfall.columns:
    adfuller[i] = sm.tsa.stattools.adfuller(dfall[i], regression="c", autolag='AIC')
    dfall[i]    = np.where(abs(dfall[i])>=5*np.std(dfall[i]), np.nan, dfall[i])
    dfall       = dfall.ffill()
    
ff3             = ['ret', 'smb', 'hml']
notff3          = ['cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret']
adfuller.columns= ff3 + notff3
dfall.columns   = adfuller.columns
dfall.index     = Sdata.index[1:]
# Stationarity assumption holds.

sixcolors       = ['darkcyan', 'teal', 'seagreen' , 'mediumseagreen' , 'lightseagreen' , 'mediumaquamarine' ]

dfall[notff3].corr()
# There seems to be multicollinearity in the features. Let's extract the PCs
smPC            = sm.PCA(dfall, standardize=1, method='svd')
smPCcorr        = smPC.scores.corr()
dfallPC         = smPC.scores

#!! MULTIVARIATE REGRESSIONS MUST USE PCS
# %% ## Printing section of data
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
# Using lagged values of responses and features to test for causality (individual)
var1        = []
varesults1  = []
varesults11 = []
var1_acorr  = []
var1aic     = []
var1yesno   = []
var11yesno  = []
var1pvals   = []
var11pvals  = []
for resp in list_of_responses:
    for exog in notff3:
        var1.append(VAR(regdata[[resp,exog]], dates=regdata.index).fit(maxlags=None, ic='aic', trend='c'))
        varesults1.append(var1[len(var1)-1].test_causality(caused=resp, causing=exog, kind='f', signif=0.1))
        varesults11.append(var1[len(var1)-1].test_causality(caused=exog,causing=resp, kind='f', signif=0.1))
        var1aic.append(var1[len(var1)-1].aic)
        var1pvals.append(varesults1[len(var1)-1].pvalue)
        var11pvals.append(varesults11[len(var1)-1].pvalue)
        var1yesno.append(varesults1[len(var1)-1].conclusion)
        var11yesno.append(varesults11[len(var1)-1].conclusion)
        var1_acorr.append(var1[len(var1)-1].acorr)
        print(varesults1[len(var1)-1].summary())

### Plot results
plt.figure()
plt.bar(range(0,len(var1pvals)),np.array(var1yesno), alpha=.5, label='Feat -> Resp', color='r', align='edge')
plt.bar(range(0,len(var11pvals)), np.array(var11yesno), alpha=.5, label= 'Resp -> Feat', color='b', align='edge')
plt.xticks(ticks=np.arange(0,len(var1pvals),step=6), labels=list_of_responses, fontsize=8)
plt.xlim(0,len(var1pvals))
plt.legend()
plt.grid(b=None,axis='x')

### Mean performance of each feature
var1pvals   = pd.DataFrame(np.split(np.array(var1pvals), 6), columns=list_of_responses, index=notff3) 
print('Mean of p-values per feature = ', var1pvals.mean(axis=0))
var11pvals   = pd.DataFrame(np.split(np.array(var11pvals), 6), columns=notff3, index=list_of_responses) 
print('Mean of p-values per feature = ', var11pvals.mean(axis=0))

# %% LOGIT ####################################################################

# Transform abnormal returns into logical: 1 iff \abs(response) > \sigma^2
Pdata.columns     = list_of_responses
Pdata3            = pd.DataFrame(Pdata[1::])
Pdata3.index      = regdata.index
retpos            = Pdata3.abs() > pd.core.window.Expanding(Pdata3, min_periods=12, axis=0).std().bfill()
retpos            = retpos.astype(int)
retpos            = pd.DataFrame(retpos)
retpos.columns    = list_of_responses
regdata2          = pd.concat([retpos,dfall],axis=1,ignore_index=False)

# X defined as [ones, cefd, ret]
# One regression per stock portfolio
regL1  = []
tvalsL1 = []
for resp in list_of_responses:
    for exog in notff3:
        formula = resp + " ~ " + exog
        regL1.append(sm.Logit.from_formula(formula, data=regdata2).fit())
        print(regL1[len(regL1)-1].summary2())
        print(regL1[len(regL1)-1].get_margeff(at='overall').summary())
        tvalsL1.append(regL1[len(regL1)-1].get_margeff(at='overall').tvalues)
tvalsL1copy = np.array(tvalsL1).reshape(-6,6)
print('Mean of pvalues mfx =', np.abs(tvalsL1copy).mean(axis=0))
sns.regplot(regdata2[['cefd']],regdata2[[list_of_responses[3]]], logistic=1)

# X defined as [ones, [dfCOT]]
# One regression per stock portfolio
regL2  = []
tvalsL2 = []
for resp in list_of_responses:
    formula = resp + " ~ NONPNL + CPNL + NCPNL + OI"
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
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
dflag             = pd.concat([
                               pd.DataFrame(np.roll(dfall,1,axis=0),  # First lag
                                            index=dfall.index, 
                                            columns=dfall.columns),
                               pd.DataFrame(np.roll(dfall,2,axis=0),  # Second lag
                                            index=dfall.index, 
                                            columns=dfall.columns)], axis=1).drop(index=dfall.index[:2])
#
responsesL        = [list_of_responses[i] + str('_L') for i in range(len(list_of_responses))]
#
dada              = pd.concat([Pdata2, 
                               pd.DataFrame(np.roll(Pdata2,1,axis=0), # First lag 
                                            index=dfall.index, 
                                            columns=responsesL), 
                               pd.DataFrame(np.roll(Pdata2,2,axis=0), # Second lag
                                            index=dfall.index, 
                                            columns=responsesL)], axis=1).drop(index=dfall.index[:2])
regdata           = pd.concat([dada,dflag],axis=1,ignore_index=False)
#
######             ######################################################## 
retpos            = Pdata3.abs() > pd.core.window.Expanding(Pdata3,axis=0).std().bfill()
retpos            = pd.DataFrame(retpos.astype(int))
retpos            = pd.concat([retpos, 
                               pd.DataFrame(np.roll(retpos,1,axis=0),
                                                    index=dfall.index,
                                                    columns=notff3),
                               pd.DataFrame(np.roll(retpos,2,axis=0),
                                                    index=dfall.index,
                                                    columns=notff3)], axis=1).drop(index=dfall.index[:2])
regdata2          = pd.concat([retpos,dflag],axis=1,ignore_index=False)
###########################################################################

# Define splits. The first 250 observations are used, expanded by one for each iteration
trainmin   = 270
trainmax   = len(regdata.values)

tsplit          = TimeSeriesSplit(n_splits=5, max_train_size=270)
scaler          = StandardScaler()
scaler2         = StandardScaler()
lin_regWO       = []
lin_regW        = []
y_predLinWO     = []
y_predLinW      = []
axLinWO         = []
axLinW          = []
trainscoreWO    = []
trainscoreW     = []
testscoreWO     = []
testscoreW      = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(range(i-trainmin,i))
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[responsesL + notff3].iloc[i-trainmin:i], regdata[responsesL + notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train sets
    scalefitwof    = scaler.fit(Xwof_train) # Standardize to fit train set WITHOUT FEATURE LAGS
    scalefitwf     = scaler2.fit(Xwf_train)  # Standardize to fit train set WITH FEATURE LAGS
    # Step3: Standardize train AND test sets WITHOUT FEATURES nor their lags
    Xwof_train     = pd.DataFrame(scalefitwof.transform(Xwof_train), columns=Xwof_train.columns,index=Xwof_train.index)
    Xwof_test      = pd.DataFrame(scalefitwof.transform(Xwof_test), columns=Xwof_test.columns,index=Xwof_test.index)
    # Standardize train AND test sets WITH FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses 
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for exog in notff3:
            ## Fit regressions
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False).fit(Xwof_train[respL],y_train2[resp]),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_train[[respL] + [exog]],y_train2[resp]),}
            ## Predictions
            lin_regWO.append(model['OLSwof'])
            lin_regW.append(model['OLSwf'])
            y_predLinWO.append(lin_regWO[len(lin_regWO)-1].predict(Xwof_test[respL]))
            y_predLinW.append(lin_regW[len(lin_regW)-1].predict(Xwf_test[[respL] + [exog]]))
            axLinWO.append(np.array([y_test2[resp], y_predLinWO[len(lin_regWO)-1]]).T)
            axLinW.append(np.array([y_test2[resp], y_predLinW[len(lin_regW)-1]]).T)    ## Performances
            ## Compare train set performance
            trainscoreWO.append(model['OLSwof'].score(Xwof_train[respL], y_train2[resp]))
            trainscoreW.append(model['OLSwf'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
t1 = timer()
print(t1-t0)

# Compare test set performance
cols = int(len(axLinW)*2/(trainmax-trainmin)) # Twice the number of predictions made
# We want to restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinW0  = np.array(axLinWO).squeeze().reshape(trainmax-trainmin,cols)
y_predLinW   = np.array(axLinW).squeeze().reshape(trainmax-trainmin,cols)
linresultsWO = []
axLinWO      = []
linresid     = []
linresultsW  = []
axLinW       = []
# Get the results
for i in range(0,y_predLinW0.shape[1]-1,2):
    linresultsWO.append(r2_score(y_predLinW0[1:,i],y_predLinW0[:-1,i+1]))
    axLinWO.append(np.array([y_predLinW0[:,i],y_predLinW0[:,i+1]]).T)
    #linresid.append(axLin[len(axLin)-1].T[0]-axLin[len(axLin)-1].T[1]) # Heteroscedasticity? 
    linresultsW.append(r2_score(y_predLinW[1:,i],y_predLinW[:-1,i+1]))
    axLinW.append(np.array([y_predLinW[:,i],y_predLinW[:,i+1]]).T)

### Mean performance of each feature
linresultsWO = pd.DataFrame(np.split(np.array(linresultsWO), 6), columns=list_of_responses, index=responsesL)
print('Mean R2 score without feature = ', linresultsWO.mean(axis=1))
linresultsW  = pd.DataFrame(np.split(np.array(linresultsW), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', linresultsW.mean(axis=1))

"""
## Plot train results, LinearRegression
plt.bar(np.arange(0,len(trainscoreWO),step=1), height=trainscoreWO, 
                  align='edge', alpha=0.5, label='Without Features')
plt.bar(np.arange(0,len(trainscoreW),step=1), height=trainscoreW, 
                  align='edge', alpha=0.5, label='With Features')
plt.xticks(ticks=np.arange(0,len(trainscoreWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')


"""

##############################################################################
##############################################################################

# %% PRINCIPAL COMPONENTS ANALYSIS ###########################################

# First, split the data, then scale, then apply SVD.
# Second, test significance of extracted pcs (LinearRegression)
#
pca                 = PCA(n_components='mle', whiten=1, random_state=42)
lin_regWPCA         = []
y_predLinWPCA       = []
axLinWPCA           = []
trainscoreWPCA      = []
testscoreWPCA       = []
#
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[notff3].iloc[i-trainmin:i], regdata[notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train sets
    scalefitwof    = scaler.fit(Xwof_train)
    scalefitwf     = scaler2.fit(Xwf_train)
    # Step3: Standardize train AND test sets WITHOUT FEATURES nor their lags
    Xwof_train     = pd.DataFrame(scalefitwof.transform(Xwof_train), columns=Xwof_train.columns,index=Xwof_train.index)
    Xwof_test      = pd.DataFrame(scalefitwof.transform(Xwof_test), columns=Xwof_test.columns,index=Xwof_test.index)
    # Standardize train AND test sets WITHOUT FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
######## Apply PCA transformation ############################################
    # Fit PCA to train set (with features)
    pca.fit(Xwf_train) 
    # Transform train and test sets 
    Xwf_trainPCA    = pca.transform(Xwf_train)
    Xwf_testPCA     = pca.transform(Xwf_test)  # Transformed test sets
    pclist          = ['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])]
    # Dataframe train set and enumerate PCs
    Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train.index, columns=pclist)
    # Dataframe test set and enumerate PCs
    Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test.index, columns=Xwf_trainPCA.columns)
    ####regdataPC.append(X_pca2.values) Ignore
    # Concatenate the response lags with all PCs
    Xwf_trainPCA = pd.concat([Xwof_train,Xwf_trainPCA], axis=1)
    Xwf_testPCA  = pd.concat([Xwof_test,Xwf_testPCA], axis=1)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for pc in pclist:
            #print(resp, respL, pc)
            ## Fit regression
            model      = {'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]),}
            ## Predictions
            lin_regWPCA.append(model['OLSwf'])
            y_predLinWPCA.append(lin_regWPCA[len(lin_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinWPCA.append(np.array([y_test2[resp], y_predLinWPCA[len(lin_regWPCA)-1]]).T)
            ## Performance
            ## Compare train set performance
            trainscoreWPCA.append(model['OLSwf'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
t1 = timer()
print(t1-t0)
#
cols = int(len(axLinWPCA)*2/(trainmax+1-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predited y_test
y_predLinWPCA   = np.array(axLinWPCA).squeeze().reshape(trainmax+1-trainmin,cols)
linresultsWPCA  = []
axLinWPCA       = []
# Get the results
for i in range(0,y_predLinWPCA.shape[1]-1,2):
    linresultsWPCA.append(r2_score(y_predLinWPCA[1:,i],y_predLinWPCA[:-1,i+1]))
    axLinWPCA.append(np.array([y_predLinWPCA[:,i],y_predLinWPCA[:,i+1]]).T)
#
### Mean performance of each feature
linresultsWPCA  = pd.DataFrame(np.split(np.array(linresultsWPCA), 6), columns=pclist, index=list_of_responses).T
print('Mean of R2 score with feature = ', linresultsWPCA.mean(axis=1))
#
sns.regplot(Xwf_trainPCA['PC6'], y_train2[list_of_responses[-1]])
"""
## Plot train results, LinearRegression
plt.bar(np.arange(0,len(trainscoreWPCA),step=1), height=trainscoreWPCA, 
                  align='edge', alpha=0.5, label='With PCs')
plt.xticks(ticks=np.arange(0,len(trainscoreWPCA),step=6))
plt.legend()
plt.grid(b=None,axis='x')

"""


############################################################################
############################################################################

#%%     Plotting Routine   ####################################################

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

#%%    ENSEMBLE ALGORITHMS    #################################################

# FEATURE SELECTION STARTS BELOW
# First two models are Ridge and Lasso.

#%%    RIDGE AND LASSO  REGRESSIONS  ##########################################
lasso_regWO       = []
y_predLinlassoWO  = []
axLinlassoWO      = []
ridge_regWO       = []
y_predLinridgeWO  = []
axLinridgeWO      = []
#
lasso_regW       = []
y_predLinlassoW  = []
axLinlassoW      = []
ridge_regW       = []
y_predLinridgeW  = []
axLinridgeW      = []
#
trainscoreRWO     = []
trainscoreRW      = []
trainscoreLWO     = []
trainscoreLW      = []
#
testscoreRWO     = []
testscoreRW      = []
testscoreLWO     = []
testscoreLW      = []
#
lasso_params    = {'alpha':[0.005, 0.01, 0.02, 0.05]}
ridge_params    = {'alpha':[5,10,20,50],
                   'solver': ['auto','svd','lsqr','saga']}
#
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[responsesL + notff3].iloc[i-trainmin:i], regdata[responsesL + notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train sets
    scalefitwof    = scaler.fit(Xwof_train) # Standardize to fit train set WITHOUT FEATURE LAGS
    scalefitwf     = scaler2.fit(Xwf_train)  # Standardize to fit train set WITH FEATURE LAGS
    # Step3: Standardize train AND test sets WITHOUT FEATURES nor their lags
    Xwof_train     = pd.DataFrame(scalefitwof.transform(Xwof_train), columns=Xwof_train.columns,index=Xwof_train.index)
    Xwof_test      = pd.DataFrame(scalefitwof.transform(Xwof_test), columns=Xwof_test.columns,index=Xwof_test.index)
    # Standardize train AND test sets WITHOUT FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for exog in notff3:
            model        = {'LassoWO': GridSearchCV(Lasso(fit_intercept=False, normalize=False, 
                                                          random_state=42, selection='random', 
                                                          max_iter=1000, tol=0.001), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error',
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train.index), iid=True).fit(Xwof_train[respL], y_train2[resp]).best_estimator_, 
                            'LassoW':  GridSearchCV(Lasso(fit_intercept=False, normalize=False, 
                                                          random_state=42, selection='random', 
                                                          max_iter=1000, tol=0.001), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train.index), iid=True).fit(Xwf_train[[respL] + [exog]], y_train2[resp]).best_estimator_,
                            'RidgeWO': GridSearchCV(Ridge(fit_intercept=False, normalize=False, 
                                                          random_state=42, 
                                                          max_iter=1000, tol=0.001), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train.index), iid=True).fit(Xwof_train[respL], y_train2[resp]).best_estimator_,
                            'RidgeW':  GridSearchCV(Ridge(fit_intercept=False, normalize=False, 
                                                          random_state=42, 
                                                          max_iter=1000, tol=0.001), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train.index), iid=True).fit(Xwf_train[[respL] + [exog]], y_train2[resp]).best_estimator_,
                            }
            ## RIDGE Predictions
            ridge_regWO.append(model['RidgeWO'])
            y_predLinridgeWO.append(ridge_regWO[len(ridge_regWO)-1].predict(Xwof_test[respL]))
            axLinridgeWO.append(np.array([y_test2[resp], y_predLinridgeWO[len(ridge_regWO)-1]]).T)
            ridge_regW.append(model['RidgeW'])
            y_predLinridgeW.append(ridge_regW[len(ridge_regW)-1].predict(Xwf_test[[respL] + [exog]]))
            axLinridgeW.append(np.array([y_test2[resp], y_predLinridgeW[len(ridge_regW)-1]]).T)
            ## LASSO Predictions
            lasso_regWO.append(model['LassoWO'])
            y_predLinlassoWO.append(lasso_regWO[len(lasso_regWO)-1].predict(Xwof_test[respL]))
            axLinlassoWO.append(np.array([y_test2[resp], y_predLinlassoWO[len(lasso_regWO)-1]]).T)
            lasso_regW.append(model['LassoW'])
            y_predLinlassoW.append(lasso_regW[len(lasso_regW)-1].predict(Xwf_test[[respL] + [exog]]))
            axLinlassoW.append(np.array([y_test2[resp], y_predLinlassoW[len(lasso_regW)-1]]).T)
            ## Performances
            ## Compare train set performance
            trainscoreRWO.append(model['RidgeWO'].score(Xwof_train[respL], y_train2[resp]))
            trainscoreLWO.append(model['LassoWO'].score(Xwof_train[respL], y_train2[resp]))
            trainscoreRW.append(model['RidgeW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
            trainscoreLW.append(model['LassoW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
t1 = timer()
print(t1-t0)

cols = int(len(axLinridgeWO)*2/(trainmax-trainmin))

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinridgeWO    = np.array(axLinridgeWO).squeeze().reshape(trainmax-trainmin,cols)
y_predLinridgeW     = np.array(axLinridgeW).squeeze().reshape(trainmax-trainmin,cols)
y_predLinlassoWO    = np.array(axLinlassoWO).squeeze().reshape(trainmax-trainmin,cols)
y_predLinlassoW     = np.array(axLinlassoW).squeeze().reshape(trainmax-trainmin,cols)
ridgeresultsWO      = []
axLinridgeWO        = []
lassoresultsWO      = []
axLinlassoWO        = []
ridgeresultsW       = []
axLinridgeW         = []
lassoresultsW       = []
axLinlassoW         = []
# Get the results
for i in range(0,y_predLinridgeWO.shape[1]-1,2):
    ridgeresultsWO.append(r2_score(y_predLinridgeWO[1:,i],y_predLinridgeWO[:-1,i+1]))
    ridgeresultsW.append(r2_score(y_predLinridgeW[1:,i],y_predLinridgeW[:-1,i+1]))
    lassoresultsWO.append(r2_score(y_predLinlassoWO[1:,i],y_predLinlassoWO[:-1,i+1]))
    lassoresultsW.append(r2_score(y_predLinlassoW[1:,i],y_predLinlassoW[:-1,i+1]))
    axLinridgeWO.append(np.array([y_predLinridgeWO[:,i],y_predLinridgeWO[:,i+1]]).T)
    axLinridgeW.append(np.array([y_predLinridgeW[:,i],y_predLinridgeW[:,i+1]]).T)
    axLinlassoWO.append(np.array([y_predLinlassoWO[:,i],y_predLinlassoWO[:,i+1]]).T)
    axLinlassoW.append(np.array([y_predLinlassoW[:,i],y_predLinlassoW[:,i+1]]).T)

### Mean performance of each feature
ridgeresultsWO = pd.DataFrame(np.split(np.array(ridgeresultsWO), 6), columns=list_of_responses, index=responsesL)
print('Mean R2 score without feature = ', ridgeresultsWO.mean(axis=1))
ridgeresultsW  = pd.DataFrame(np.split(np.array(ridgeresultsW), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', ridgeresultsW.mean(axis=1))
lassoresultsWO = pd.DataFrame(np.split(np.array(lassoresultsWO), 6), columns=list_of_responses, index=responsesL)
print('Mean R2 score without feature = ', lassoresultsWO.mean(axis=1))
lassoresultsW  = pd.DataFrame(np.split(np.array(lassoresultsW), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', lassoresultsW.mean(axis=1))

sns.regplot(Xwf_trainPCA['PC13'], y_train2[list_of_responses[-1]])

"""
## Plot train results,
plt.bar(np.arange(0,len(trainscoreRWO),step=1), height=trainscoreRWO, 
                  align='edge', alpha=0.25, label='Ridge without Features', linewidth=1)
plt.bar(np.arange(0,len(trainscoreRW),step=1), height=trainscoreRW, 
                  align='edge', alpha=0.25, label='Ridge with Features', linewidth=1)
plt.bar(np.arange(0,len(trainscoreLWO),step=1), height=trainscoreLWO, 
                  align='edge', alpha=0.25, label=' Lasso without Features', linewidth=1)
plt.bar(np.arange(0,len(trainscoreLW),step=1), height=trainscoreLW, 
                  align='edge', alpha=0.25, label='Lasso with Features', linewidth=1)
plt.xticks(ticks=np.arange(0,len(trainscoreLWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')

"""

###############################################################################


#%% RIDGE AND LASSO REGRESSIONS WITH PCS #####################################

#
lasso_regWPCA       = []
y_predLinlassoWPCA  = []
axLinlassoWPCA      = []
ridge_regWPCA       = []
y_predLinridgeWPCA  = []
axLinridgeWPCA      = []
#
trainscoreRWPCA     = []
trainscoreLWPCA     = []
#
testscoreRWPCA      = []
testscoreLWPCA      = []
#
regdataPC2       = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[notff3].iloc[i-trainmin:i], regdata[notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train set
    scalefitwf     = scaler2.fit(Xwf_train)
    # Step3: Standardize train AND test sets WITHOUT FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), 
                                 columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), 
                                 columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), 
                               columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), 
                               columns=list_of_responses,index=y_test2.index)
######## Apply PCA transformation ############################################
    pca.fit(Xwf_train) # Fit 1
    Xwf_trainPCA    = pca.transform(Xwf_train) # Transformed train set
    Xwf_testPCA     = pca.transform(Xwf_test)  # Transformed test sets
    pclist          = ['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])]
    # Dataframe train set
    Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train.index, columns=pclist)
    # Dataframe test set
    Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test.index, columns=Xwf_trainPCA.columns)
    ####regdataPC.append(X_pca2.values) Ignore
    # Concatenate the respective response lags with all PCs
    Xwf_trainPCA = pd.concat([Xwof_train,Xwf_trainPCA], axis=1)
    Xwf_testPCA  = pd.concat([Xwof_test,Xwf_testPCA], axis=1)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for pc in pclist:
            # Fit regressions
            model        = {'LassoWPCA':  GridSearchCV(Lasso(fit_intercept=False, normalize=False, random_state=42,
                                                          selection='random', max_iter=1000, 
                                                          tol=0.001), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error',
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train.index), iid=True).fit(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]).best_estimator_,
                            'RidgeWPCA':  GridSearchCV(Ridge(fit_intercept=False, normalize=False, random_state=42, 
                                                          max_iter=1000, tol=0.001), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error',
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwf_train.index), iid=True).fit(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]).best_estimator_,}
            ## RIDGE Predictions
            ridge_regWPCA.append(model['RidgeWPCA'])
            y_predLinridgeWPCA.append(ridge_regWPCA[len(ridge_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinridgeWPCA.append(np.array([y_test2[resp], y_predLinridgeWPCA[len(ridge_regWPCA)-1]]).T)
            ## LASSO Predictions
            lasso_regWPCA.append(model['LassoWPCA'])
            y_predLinlassoWPCA.append(lasso_regWPCA[len(lasso_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinlassoWPCA.append(np.array([y_test2[resp], y_predLinlassoWPCA[len(lasso_regWPCA)-1]]).T)
            ## Performances
            ## Compare train set performance
            trainscoreRWPCA.append(model['RidgeWPCA'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
            trainscoreLWPCA.append(model['LassoWPCA'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
t1 = timer()
print(t1-t0)

# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
cols                = int(np.size(axLinridgeWPCA)/(trainmax-trainmin))
y_predLinridgeWPCA  = np.array(axLinridgeWPCA).squeeze().reshape(trainmax-trainmin,cols)
y_predLinlassoWPCA  = np.array(axLinlassoWPCA).squeeze().reshape(trainmax-trainmin,cols)
ridgeresultsWPCA    = []
axLinridgeWPCA      = []
lassoresultsWPCA    = []
axLinlassoWPCA      = []
# Get the results
for i in range(0,y_predLinridgeWO.shape[1]-1,2):
    ridgeresultsWPCA.append(r2_score(y_predLinridgeW[1:,i],y_predLinridgeW[:-1,i+1]))
    lassoresultsWPCA.append(r2_score(y_predLinlassoW[1:,i],y_predLinlassoW[:-1,i+1]))
    axLinridgeWPCA.append(np.array([y_predLinridgeW[:,i],y_predLinridgeW[:,i+1]]).T)
    axLinlassoWPCA.append(np.array([y_predLinlassoW[:,i],y_predLinlassoW[:,i+1]]).T)

### Mean performance of each feature
ridgeresultsWPCA = pd.DataFrame(np.split(np.array(ridgeresultsWPCA), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', ridgeresultsWPCA.mean(axis=1))
lassoresultsWPCA = pd.DataFrame(np.split(np.array(lassoresultsWPCA), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', lassoresultsWPCA.mean(axis=1))

sns.regplot(Xwf_trainPCA['PC13'], y_train2[list_of_responses[-1]])
"""
## Plot train results,
plt.bar(np.arange(0,len(trainscoreRWO),step=1), height=trainscoreRWO, 
                  align='edge', alpha=0.25, label='Ridge without PCs', linewidth=1)
plt.bar(np.arange(0,len(trainscoreRWPCA),step=1), height=trainscoreRWPCA, 
                  align='edge', alpha=0.25, label='Ridge with PCs', linewidth=1)
plt.bar(np.arange(0,len(trainscoreLWO),step=1), height=trainscoreLWO, 
                  align='edge', alpha=0.25, label=' Lasso without PCs', linewidth=1)
plt.bar(np.arange(0,len(trainscoreRWPCA),step=1), height=trainscoreRWPCA, 
                  align='edge', alpha=0.25, label='Lasso with PCs', linewidth=1)
plt.xticks(ticks=np.arange(0,len(trainscoreLWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')
"""
###############################################################################


#%% RANDOM FOREST FEATURE SELECTION ###########################################

# Defining grid for Gridsearch cross validation

n_estimators      = [5]

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
forest_regWO     = []
axRFRWO          = []
ypredforestWO    = []

forest_regW      = []
axRFRW           = []
ypredforestW     = []

t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[responsesL + notff3].iloc[i-trainmin:i], regdata[responsesL + notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train sets
    scalefitwof    = scaler.fit(Xwof_train)
    scalefitwf     = scaler2.fit(Xwf_train)
    # Step3: Standardize train AND test sets WITHOUT FEATURES nor their lags
    Xwof_train     = pd.DataFrame(scalefitwof.transform(Xwof_train), columns=Xwof_train.columns,index=Xwof_train.index)
    Xwof_test      = pd.DataFrame(scalefitwof.transform(Xwof_test), columns=Xwof_test.columns,index=Xwof_test.index)
    # Standardize train AND test sets WITHOUT FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        # Set GridSearchCV 
        model   = {'RFRWO': GridSearchCV(RFR, param_grid=random_grid, 
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True, cv=tsplit.split(Xwof_train.index),
                                       iid=True, n_jobs=-1).fit(Xwof_train[respL], y_train2[resp]).best_estimator_,
                   'RFRW': GridSearchCV(RFR, param_grid=random_grid, scoring='neg_mean_squared_error',
                                        return_train_score=True, cv=tsplit.split(Xwf_train.index),
                                        iid=True, n_jobs=-1).fit(Xwf_train[[respL] + notff3], y_train2[resp]).best_estimator_}
        ## Random Forest Regression
        forest_regWO.append(model['RFRWO'])
        ypredforestWO.append(forest_regWO[len(forest_regWO)-1].predict(Xwof_test[respL]))
        axRFRWO.append(np.array([y_test2[resp], ypredforestWO[len(forest_regWO)-1]]).T)
        forest_regW.append(model['RFRW'])
        ypredforestW.append(forest_regW[len(forest_regW)-1].predict(Xwf_test[[respL] + notff3]))
        axRFRW.append(np.array([y_test2[resp], ypredforestW[len(forest_regW)-1]]).T)
        ## Random Forest Classification
        #forest_clas.append(model['RFRW'])
        #ypredforestclas.append(forest_clas[len(forest_clas)-1].predict(Xwf_test))
        #axRFC.append(np.array([y_test1[resp], ypredforestclas[len(forest_clas)-1]]).T)
t1 = timer()
print(t1-t0)

cols            = int(np.size(axRFRWO)/(trainmax-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforestWO   = np.array(axRFRWO).squeeze().reshape(trainmax-trainmin,cols)
ypredforestW    = np.array(axRFRW).squeeze().reshape(trainmax-trainmin,cols)
forestresultsWO = []
axRFRWO         = []
forestresultsW  = []
axRFRW          = []
for i in range(0,ypredforestWO.shape[1]-1,2):
    forestresultsWO.append(r2_score(ypredforestWO[1:,i],ypredforestWO[:-1,i+1]))
    axRFRWO.append(np.array([ypredforestWO[:,i],ypredforestWO[:,i+1]]).T)
    forestresultsW.append(r2_score(ypredforestW[1:,i],ypredforestW[:-1,i+1]))
    axRFRW.append(np.array([ypredforestW[:,i],ypredforestW[:,i+1]]).T)

FI              = []
FII = pd.DataFrame()
for i in range(len(forest_regW)):
    for respL in responsesL:
        FI.append(pd.DataFrame(forest_regW[i].feature_importances_, index = Xwf_train[[respL] + notff3].columns, columns=['Feature Importance']))
        FII[i] = forest_regW[i].feature_importances_
        FII.index = Xwf_train[[respL] + notff3].columns

FII.mean(axis = 1)
print('Mean of feature importance = ', FII.mean(axis = 1))

"""   
plt.barh(range(forest_regWPCA[48].n_features_), FIPCA[49], align='center')
plt.yticks(np.arange(forest_regW[48].n_features_), notff3)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
"""

#%% Random Forest with PCs ###################################################

## RANDOM FOREST WITH PCS ####
forest_regWPCA      = []
axRFRWPCA           = []
ypredforestWPCA     = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL].iloc[i-trainmin:i], regdata[responsesL].iloc[i:i+1]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[notff3].iloc[i-trainmin:i], regdata[notff3].iloc[i:i+1]
    #
    y_train1, y_test1     = regdata2[list_of_responses].iloc[i-trainmin:i], regdata2[list_of_responses].iloc[i:i+1]
    y_train2, y_test2     = regdata[list_of_responses].iloc[i-trainmin:i], regdata[list_of_responses].iloc[i:i+1]
    # Step2: Fit standardizer to train set
    scalefitwf     = scaler2.fit(Xwf_train)
    # Step3: Standardize train AND test sets WITHOUT FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), 
                                 columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), 
                                 columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), 
                               columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), 
                               columns=list_of_responses,index=y_test2.index)
######## Apply PCA transformation ############################################
    pca.fit(Xwf_train) # Fit 1
    Xwf_trainPCA    = pca.transform(Xwf_train) # Transformed train set
    Xwf_testPCA     = pca.transform(Xwf_test)  # Transformed test sets
    pclist          = ['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])]
    # Dataframe train set
    Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train.index, columns=pclist)
    # Dataframe test set
    Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test.index, columns=Xwf_trainPCA.columns)
    ####regdataPC.append(X_pca2.values) Ignore
    # Concatenate the respective response lags with all PCs
    Xwf_trainPCA = pd.concat([Xwof_train,Xwf_trainPCA], axis=1)
    Xwf_testPCA  = pd.concat([Xwof_test,Xwf_testPCA], axis=1)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        # Set GridSearchCV 
        model   = {'RFRWPCA': GridSearchCV(RFR, param_grid=random_grid, scoring='neg_mean_squared_error',
                                           return_train_score=True, cv=tsplit.split(Xwf_trainPCA.index),
                                           iid=True, n_jobs=-1).fit(Xwf_trainPCA[[respL] + pclist], y_train2[resp]).best_estimator_}
        ## Random Forest Regression
        forest_regWPCA.append(model['RFRWPCA'])
        ypredforestWPCA.append(forest_regWPCA[len(forest_regWPCA)-1].predict(Xwf_testPCA[[respL] + pclist]))
        axRFRWPCA.append(np.array([y_test2[resp], ypredforestWPCA[len(forest_regWPCA)-1]]).T)
        ## Random Forest Classification
        #forest_clas.append(model['RFRW'])
        #ypredforestclas.append(forest_clas[len(forest_clas)-1].predict(Xwf_test))
        #axRFC.append(np.array([y_test1[resp], ypredforestclas[len(forest_clas)-1]]).T)
t1 = timer()
print(t1-t0)

cols                = int(np.size(axRFRWPCA)/(trainmax-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforestWPCA     = np.array(axRFRWPCA).squeeze().reshape(trainmax-trainmin,cols)
forestresultsWPCA   = []
axRFRWPCA           = []
for i in range(0,ypredforestWPCA.shape[1]-1,2):
    forestresultsWPCA.append(r2_score(ypredforestWPCA[1:,i],ypredforestWPCA[:-1,i+1]))
    axRFRWPCA.append(np.array([ypredforestWPCA[1:,i],ypredforestWPCA[:-1,i+1]]).T)

FIPCA           = []
idxx            = []
idxl            = []
meanFIPCA       = []
FIPCAdf         = pd.DataFrame()
for i in range(len(forest_regWPCA)):
    for respL in responsesL:
        FIIPCA = np.array(forest_regWPCA[i].feature_importances_)
        idxx.append(np.arange(len(FIIPCA)-2))
        idxl.append([[respL] + [respL] + ['PC' + str(idxx[-1][x]+1) for x in range(len(idxx[-1]))]])
        FIPCA.append(pd.DataFrame(FIIPCA,index=idxl[-1], columns=['Feature Importance']))

print('Mean of feature importance (with PCs) = ', FIIPCA.mean(axis = 1))
###############################################################################

# Whathappenedtomycommentssdsdsdsd
