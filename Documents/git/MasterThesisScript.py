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
dfCOT           = pd.concat([pd.DataFrame(NONPNL),
                             pd.DataFrame(CPNL),
                             pd.DataFrame(NCPNL),
                             pd.DataFrame(OI)],axis=1)
dfCOT.columns= range(4)


####    CEFD data
cefd            = pd.DataFrame(data.dscnt.interpolate('linear').values[1:])

####  Augmented Dickey Fuller test for unit root in variables #################
adfuller        = pd.DataFrame()
dfall           = pd.concat([famafrench, cefd, dfCOT, vixret],axis=1)
dfall.columns   = np.arange(9)
for i in dfall.columns:
    adfuller[i] = sm.tsa.stattools.adfuller(dfall[i], maxlag=None, regression="c", autolag='AIC')
    dfall[i]    = np.where(abs(dfall[i])>=5*np.std(dfall[i]), np.nan, dfall[i])
    dfall       = dfall.ffill()

# Defining Columns and indexes  
ff3             = ['ret', 'smb', 'hml']
notff3          = ['cefd', 'NONPNL', 'CPNL', 'NCPNL', 'OI', 'vixret']
adfuller.columns= ff3 + notff3
dfall.columns   = adfuller.columns
dfall.index     = Sdata.index[1:]

# Correlation matrix of features
dfall[notff3].corr()
# There seems to be multicollinearity in the features. Extracting the PCs
smPC            = sm.PCA(dfall, standardize=1, ncomp=3, method='svd')
smPCcorr        = smPC.scores.corr()
dfallPC         = smPC.scores

# %% ## Printing section of data
"""
sixcolors       = ['darkcyan', 'teal', 'seagreen' ,
                   'mediumseagreen' , 'lightseagreen' , 'mediumaquamarine' ]
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
# %% GLS REGRESSIONS ##########################################################

######  Define regdata - The regression dataset ##############################
Pdata2            = pd.DataFrame(Pdata[1::])
Pdata2.index      = Pdata.index[1:]
list_of_responses = ["SMALLLoBM", "ME1BM2", "SMALLHiBM", "BIGLoBM", "ME2BM2", "BIGHiBM"]
Pdata2.columns    = list_of_responses
regdata           = pd.concat([Pdata2,dfall],axis=1,ignore_index=False)
regdata.columns   = np.append(list_of_responses,dfall.columns.values)
##############################################################################

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
        resid1.append(reg1[len(reg1)-1].resid)
        # White's test for heterosced
        white1ols.append(sm.stats.diagnostic.het_white(resid1[len(reg1)-1],
                                                 sm.add_constant(regdata[[exog]])))
        # Breusch Godfrey test for autocorr
        acorr1ols.append(sm.stats.diagnostic.acorr_breusch_godfrey(reg1[len(reg1)-1]))
        # Regress residual time series on its lag
        res_fit     = sm.OLS(resid1[len(reg1)-1].values[1:], resid1[len(reg1)-1].values[:-1]).fit()
        # Obtain parameter
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

#### Regression 2 - COT bundle
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

#### Regression 3 - All features
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

###### Cointegration and stationarity test ###################################
Pdata2Price        = pd.DataFrame(np.cumprod(np.exp(Pdata2.values),axis=0))
FF3Price           = np.cumprod(np.exp(famafrench),axis=0)
####    COT levels
COT2Price          = pd.concat([pd.DataFrame(nclong/ncshort),
                                pd.DataFrame(clong/cshort),
                                pd.DataFrame(nonreplong/nonrepshort),
                                pd.DataFrame(oi)],axis=1).drop(index=nclong.index[0])
COT2Price.columns     = range(4)
COT2Price.index       = FF3Price.index
vix2Price             = np.cumprod(np.exp(vixret))
data2Price            = pd.concat([cefd,FF3Price,COT2Price,vix2Price], 
                                  axis=1, ignore_index=True)
data2Price.columns    = dfall.columns
regdata2Price         = pd.concat([Pdata2Price,data2Price], 
                               axis=1)
regdata2Price.columns = regdata.columns
regdata2Price.index   = regdata.index
#
# Cointegration test
ERCO                  = []
for i in range(len(Pdata2Price.columns)):
    for j in range(len(data2Price.columns)):
        ERCO.append(sm.tsa.stattools.coint(Pdata2Price.loc[:,Pdata2Price.columns[i]],
                                           data2Price.loc[:,data2Price.columns[j]],
                                           maxlag=12))
# Unit root test on levels
ADF2                  = pd.DataFrame()
for i in regdata2Price.columns:
    ADF2[i] = sm.tsa.stattools.adfuller(regdata2Price[i].diff().bfill(), maxlag=12, regression="c", autolag='AIC')

# =============================================================================
# Even though the data seems to be stationary, the absence of cointegration means
# VAR inputs should be on differenced data. Thus, we use the same data set as in the GLS regressions.
# =============================================================================
##############################################################################

from statsmodels.tsa.vector_ar.var_model1 import VAR
#### VAR 1
# Using lagged values of responses and features to test for causality (per feature)
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
        varesults1.append(var1[len(var1)-1].test_causality(caused=resp, causing=exog, kind='wald', signif=0.05))
        varesults11.append(var1[len(var1)-1].test_causality(caused=exog,causing=resp, kind='wald', signif=0.05))
        var1aic.append(var1[len(var1)-1].aic)
        var1pvals.append(varesults1[len(var1)-1].pvalue)
        var11pvals.append(varesults11[len(var1)-1].pvalue)
        var1yesno.append(varesults1[len(var1)-1].conclusion)
        var11yesno.append(varesults11[len(var1)-1].conclusion)
        var1_resid.append(var1[len(var1)-1].resid)
        #var1_acorr.append(var1[len(var1)-1].plot_acorr()) # Takes some computer power to run
        print(var1[len(var1)-1].summary())
#
### Plot results
plt.figure()
plt.bar(range(0,len(var1pvals)),np.array(var1yesno), alpha=.5, label='Feat -> Resp', color='r', align='edge')
plt.bar(range(0,len(var11pvals)), np.array(var11yesno), alpha=.5, label= 'Resp -> Feat', color='b', align='edge')
plt.xticks(ticks=np.arange(0,len(var1pvals),step=6), labels=list_of_responses, fontsize=8)
plt.xlim(0,len(var1pvals))
plt.legend()
plt.grid(b=None,axis='x')
#
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

# %% MACHINE LEARNING  #######################################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoLarsIC, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
#from sklearn.feature_selection import f_regression
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
#from sklearn.tree.export import export_graphviz
from sklearn.model_selection import TimeSeriesSplit
#import mglearn
#import graphviz
from timeit import default_timer as timer
# =============================================================================
def aic(y, y_pred, k):
   resid = np.array([y - y_pred]).T
   rss   = np.sum(resid**2)
   AIC   = 2*k - 2*len(y)*np.log(rss/len(y))
   return AIC
# =============================================================================
tsplit          = TimeSeriesSplit(n_splits=5, max_train_size=270)
pca             = PCA(n_components='mle', whiten=1, random_state=42)
pca2            = PCA(n_components='mle', whiten=1, random_state=42)
scaler          = StandardScaler()
scaler2         = StandardScaler()
# =============================================================================
lin_regWO       = []
lin_regW        = []
lin_regcoefWO   = []
lin_regcoefW    = []
#
y_predLinWO     = []
y_predLinW      = []
axLinWO         = []
axLinW          = []
#
trainscoreWO    = []
trainscoreW     = []
testscoreWO     = []
testscoreW      = []
#
modelselectWO   = []
modelselectW    = []
#
CorrXwof_train_L= []
CorrXwf_train_L = []
CorrXwof_test_L = []
CorrXwf_test_L  = []
#
Hetlin_regWO    = []
Hetlin_regW     = []
acorrlin_regWO  = []
acorrlin_regW   = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[list_of_responses + ['ret']].iloc[train_index], regdata[list_of_responses + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata.iloc[train_index], regdata.iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
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
###### Model training and testing ############################################
    for resp in list_of_responses:
        for exog in notff3:
            ## Model Selection
            modelselectWO.append(np.max([1,VAR(Xwof_train[[resp] + ['ret']],
                                   dates=y_train2.index).select_order(maxlags=12).aic]))
            modelselectW.append(np.max([1,VAR(Xwf_train[[resp] + ['ret'] + [exog]],
                                   dates=y_train2.index).select_order(maxlags=12).aic]))
            # Define lagged X w.r.t AIC
            Xwof_train_L   = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],
                                                        maxlag0=modelselectWO[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:modelselectWO[-1]],
                                                                                        columns=Xwof_train[[resp] + ['ret']].columns[0])
            Xwf_train_L    = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + [exog]],
                                                        maxlag0=np.max([1,modelselectW[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + [exog]].index[:modelselectW[-1]],
                                                                                        columns=Xwf_train[[resp] + ['ret'] + [exog]].columns[0])
            Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']],
                                                        maxlag0=np.max([1,modelselectWO[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret']].index[:modelselectWO[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret']].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + [exog]],
                                                        maxlag0=np.max([1,modelselectW[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + [exog]].index[:modelselectW[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret'] + [exog]].columns[0])
            # Correlation matrices of lagged X
            CorrXwof_train_L.append(Xwof_train_L.corr())
            CorrXwf_train_L.append(Xwf_train_L.corr())
            CorrXwof_test_L.append(Xwof_test_L.corr())
            CorrXwf_test_L.append(Xwf_test_L.corr())
            ## Train models
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False).fit(Xwof_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)])}
            ## Predictions
            lin_regWO.append(model['OLSwof'])
            lin_regW.append(model['OLSwf'])
            lin_regcoefWO.append(lin_regWO[len(lin_regWO)-1].coef_)
            lin_regcoefW.append(lin_regW[len(lin_regW)-1].coef_)
            y_predLinWO.append(lin_regWO[len(lin_regWO)-1].predict(Xwof_test_L))
            y_predLinW.append(lin_regW[len(lin_regW)-1].predict(Xwf_test_L))
            axLinWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], y_predLinWO[len(lin_regWO)-1]]).T)
            axLinW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], y_predLinW[len(lin_regW)-1]]).T)
            ## Heteroscedasticity and autocorrelation tests
            Hetlin_regWO.append(sm.stats.diagnostic.het_white(np.array([axLinWO[-1][:,0] - axLinWO[-1][:,1]]).T, 
                                                              sm.add_constant(Xwof_test_L)))
            Hetlin_regW.append(sm.stats.diagnostic.het_white(np.array([axLinW[-1][:,0] - axLinW[-1][:,1]]).T, 
                                                             sm.add_constant(Xwf_test_L)))
            acorrlin_regWO.append(sm.stats.diagnostic.acorr_ljungbox(axLinWO[-1][:,0] - axLinWO[-1][:,1]))
            acorrlin_regW.append(sm.stats.diagnostic.acorr_ljungbox(axLinW[-1][:,0] - axLinW[-1][:,1]))
# =============================================================================
#             ## Performances
#             ## Compare train set performance
#             trainscoreWO.append(model['OLSwof'].score(Xwof_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
#             trainscoreW.append(model['OLSwf'].score(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
#             ## Compare test set performance
#             testscoreWO.append(model['OLSwof'].score(Xwof_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)]))
#             testscoreW.append(model['OLSwf'].score(Xwf_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
# =============================================================================
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
lin_regcoefWO= np.array(lin_regcoefWO)
lin_regcoefW = np.array(lin_regcoefW)
linresultsWO = []
linresidWO   = []
linresultsW  = []
linresidW    = []
# Get the results
for i in range(0,len(axLinWO)):
    linresultsWO.append(aic(axLinWO[i][:,0],axLinWO[i][:,1],lin_regcoefWO[i].shape[0]))
    linresidWO.append(axLinWO[i][:,0] - axLinWO[i][:,1])  
    linresultsW.append(aic(axLinW[i][:,0],axLinW[i][:,1],lin_regcoefW[i].shape[0]))
    linresidW.append(axLinW[i][:,0] - axLinW[i][:,1]) 
# =============================================================================
### Mean performance of each feature
linresultsWO = pd.DataFrame(np.split(np.array(linresultsWO), 6), columns=list_of_responses*5, index=[list_of_responses[i] + str('_L') for i in range(len(list_of_responses))])
print('Mean AIC without feature = ', linresultsWO.mean(axis=1))
linresultsW  = pd.DataFrame(np.split(np.array(linresultsW), 6), columns=list_of_responses*5, index=[notff3[i] + str('_L') for i in range(len(notff3))])
print('Mean AIC with feature = ', linresultsW.mean(axis=1))
# =============================================================================
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
#
lin_regWPCA       = []
lin_regcoefWPCA   = []
lin_regWPCA2      = []
lin_regcoefWPCA2  = []
#
y_predLinWPCA     = []
axLinWPCA         = []
y_predLinWPCA2    = []
axLinWPCA2        = []
#
trainscoreWPCA    = []
trainscoreWPCA2   = []
testscoreWPCA     = []
testscoreWPCA2    = []
#
countiter         = []
modelselectWPCA   = []
modelselectWPCA2  = []
#
Hetlin_regWPCA    = []
Hetlin_regWPCA2   = []
acorrlin_regWPCA  = []
acorrlin_regWPCA2 = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[list_of_responses + ['ret']].iloc[train_index], regdata[list_of_responses + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata.iloc[train_index], regdata.iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
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
###### Model training and testing ############################################
    for resp in list_of_responses:
        for exog in notff3:
            ## Model Selection
            countiter.append(y_test2.columns) # Just to count the iterations
            modelselectWPCA  = [np.max([1,modelselectWO[i]]) for i in range(len(countiter))]
            modelselectWPCA2 = [np.max([1,modelselectW[i]]) for i in range(len(countiter))]
            # Define lagged X w.r.t AIC from model selection, then trim the initial observations (with no input)
            Xwof_train_L   = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],
                                                        maxlag0=modelselectWPCA[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwof_train[[resp] + ['ret']].columns[0])
            Xwf_train_L    = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + [exog]],
                                                        maxlag0=modelselectWPCA2[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + [exog]].index[:modelselectWPCA2[-1]],
                                                                                        columns=Xwf_train[[resp] + ['ret'] + [exog]].columns[0])
            Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']],
                                                        maxlag0=np.max([1,modelselectWPCA[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret']].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret']].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + [exog]],
                                                        maxlag0=np.max([1,modelselectWPCA2[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + [exog]].index[:modelselectWPCA2[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret'] + [exog]].columns[0])
            ######## Apply PCA transformation ############################################
            # Fit PCA to train sets
            pca.fit(Xwof_train_L) 
            pca2.fit(Xwf_train_L) 
            # Transform train and test sets
            Xwof_trainPCA   = pca.transform(Xwof_train_L)
            Xwof_testPCA    = pca.transform(Xwof_test_L)
            Xwf_trainPCA    = pca2.transform(Xwf_train_L)
            Xwf_testPCA     = pca2.transform(Xwf_test_L)
            pclist          = ['PC'+ str(i+1) for i in range(Xwof_trainPCA.shape[1])]
            pclist2         = ['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])]
            # Dataframe train set and enumerate PCs
            Xwof_trainPCA   = pd.DataFrame(Xwof_trainPCA, index=Xwof_train_L.index, columns=pclist)
            Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train_L.index, columns=pclist2)
            # Dataframe test set and enumerate PCs
            Xwof_testPCA    = pd.DataFrame(Xwof_testPCA, index=Xwof_test_L.index, columns=Xwof_trainPCA.columns)
            Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test_L.index, columns=Xwf_trainPCA.columns)
            ## Train models
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False).fit(Xwof_trainPCA,y_train2[resp].iloc[y_train2.index.isin(Xwof_trainPCA.index)]),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_trainPCA,y_train2[resp].iloc[y_train2.index.isin(Xwf_trainPCA.index)])}
            ## Predictions
            lin_regWPCA.append(model['OLSwof'].fit(Xwof_trainPCA,
                                                   y_train2[resp].iloc[y_train2.index.isin(Xwof_trainPCA.index)]))
            lin_regcoefWPCA.append(lin_regWPCA[-1].coef_)
            y_predLinWPCA.append(lin_regWPCA[-1].predict(Xwof_testPCA))
            axLinWPCA.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_testPCA.index)], 
                                       y_predLinWPCA[-1]]).T)
            lin_regWPCA2.append(model['OLSwf'].fit(Xwof_trainPCA,
                                                   y_train2[resp].iloc[y_train2.index.isin(Xwof_trainPCA.index)]))
            lin_regcoefWPCA2.append(lin_regWPCA2[-1].coef_)
            y_predLinWPCA2.append(lin_regWPCA2[-1].predict(Xwf_testPCA))
            axLinWPCA2.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_testPCA.index)], 
                                       y_predLinWPCA2[-1]]).T)
            ## Heteroscedasticity and autocorrelation tests
            Hetlin_regWPCA.append(sm.stats.diagnostic.het_white(np.array([axLinWPCA[-1][:,0] - axLinWPCA[-1][:,1]]).T, 
                                                              sm.add_constant(Xwof_testPCA)))
            Hetlin_regWPCA2.append(sm.stats.diagnostic.het_white(np.array([axLinWPCA2[-1][:,0] - axLinWPCA2[-1][:,1]]).T, 
                                                             sm.add_constant(Xwf_testPCA)))
            acorrlin_regWPCA.append(sm.stats.diagnostic.acorr_ljungbox(axLinWPCA[-1][:,0] - axLinWPCA[-1][:,1]))
            acorrlin_regWPCA2.append(sm.stats.diagnostic.acorr_ljungbox(axLinWPCA2[-1][:,0] - axLinWPCA2[-1][:,1]))
# =============================================================================
#             ## Performances
#             ## Compare train set performance
#             trainscoreWPCA.append(model['OLSwof'].score(Xwof_trainPCA,y_train2[resp].iloc[y_train2.index.isin(Xwof_trainPCA.index)]))
#             trainscoreWPCA2.append(model['OLSwf'].score(Xwf_trainPCA,y_train2[resp].iloc[y_train2.index.isin(Xwf_trainPCA.index)]))
#             ## Compare test set performance
#             testscoreWPCA.append(model['OLSwof'].score(Xwof_testPCA,y_test2[resp].iloc[y_test2.index.isin(Xwof_testPCA.index)]))
#             testscoreWPCA2.append(model['OLSwf'].score(Xwf_testPCA,y_test2[resp].iloc[y_test2.index.isin(Xwf_testPCA.index)]))
# =============================================================================
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
lin_regcoefWPCA = np.array(lin_regcoefWPCA)
lin_regcoefWPCA2= np.array(lin_regcoefWPCA2)
linresultsWPCA  = []
linresidWPCA    = []
linresultsWPCA2 = []
linresidWPCA2   = []
# Get the results
for i in range(0,len(axLinWO)):
    linresultsWPCA.append(aic(axLinWPCA[i][:,0],axLinWPCA[i][:,1],lin_regcoefWPCA[i].shape[0]))
    linresidWPCA.append(axLinWPCA[i][:,0] - axLinWPCA[i][:,1])  
    linresultsWPCA2.append(aic(axLinWPCA2[i][:,0],axLinWPCA2[i][:,1],lin_regcoefWPCA2[i].shape[0]))
    linresidWPCA2.append(axLinWPCA2[i][:,0] - axLinWPCA2[i][:,1]) 
# =============================================================================
### Mean performance of each feature WRONG
linresultsWPCA = pd.DataFrame(np.split(np.array(linresultsWPCA), 6), 
                              columns=list_of_responses*5, 
                              index=[list_of_responses[i] + str('_L_PCs') for i in range(len(list_of_responses))])
print('Mean AIC without feature PCs = ', linresultsWPCA.mean(axis=1))
linresultsWPCA2 = pd.DataFrame(np.split(np.array(linresultsWPCA2), 6), 
                               columns=list_of_responses*5, 
                               index=[notff3[i] + str('_L_PCs') for i in range(len(notff3))])
print('Mean AIC with feature PCs = ', linresultsWPCA2.mean(axis=1))
# =============================================================================
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
#
lasso_regWO       = []
lasso_coef_WO     = []
y_predLinlassoWO  = []
axLinlassoWO      = []
ridge_regWO       = []
ridge_coef_WO     = []
y_predLinridgeWO  = []
axLinridgeWO      = []
#
lasso_regW       = []
lasso_coef_W     = []
y_predLinlassoW  = []
axLinlassoW      = []
ridge_regW       = []
ridge_coef_W     = []
y_predLinridgeW  = []
axLinridgeW      = []
#
trainscoreRWO    = []
trainscoreRW     = []
trainscoreLWO    = []
trainscoreLW     = []
#
testscoreRWO     = []
testscoreRW      = []
testscoreLWO     = []
testscoreLW      = []
#
lasso_params    = {'alpha':[0.005, 0.01, 0.02, 0.05]}
ridge_params    = {'alpha':[5,10,20,50],
                   'solver': ['svd','lsqr','saga']}
#
countiter       = []
modelselect2WO  = []
modelselect2W   = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[list_of_responses + ['ret']].iloc[train_index], regdata[list_of_responses + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata.iloc[train_index], regdata.iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
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
    for resp in list_of_responses:
        for exog in notff3:
            ## Model Selection
            countiter.append(y_test2.columns) # Just to count the iterations
            modelselectWO2  = [np.max([1,modelselectWO[i]]) for i in range(len(countiter))]
            modelselectW2   = [np.max([1,modelselectW[i]]) for i in range(len(countiter))]
            # Define lagged X w.r.t AIC
            Xwof_train_L    = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],
                                                        maxlag0=modelselectWO2[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:modelselectWO2[-1]],
                                                                                        columns=Xwof_train[[resp] + ['ret']].columns[0])
            Xwf_train_L     = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + [exog]],
                                                        maxlag0=modelselectW2[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + [exog]].index[:modelselectW2[-1]],
                                                                                        columns=Xwf_train[[resp] + ['ret'] + [exog]].columns[0])
            Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']],
                                                        maxlag0=modelselectWO2[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret']].index[:modelselectWO2[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret']].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + [exog]],
                                                        maxlag0=modelselectW2[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + [exog]].index[:modelselectW2[-1]],
                                                                                        columns=Xwf_test[[resp] + ['ret'] + [exog]].columns[0])
            # Train models
            model        = {'LassoWO': GridSearchCV(Lasso(fit_intercept=False, normalize=False, 
                                                          random_state=42, selection='random', 
                                                          max_iter=1000), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error',
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train_L.index), iid=True).fit(Xwof_train_L, y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]).best_estimator_, 
                            'LassoW':  GridSearchCV(Lasso(fit_intercept=False, normalize=False, 
                                                          random_state=42, selection='random', 
                                                          max_iter=1000), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwf_train_L.index), iid=True).fit(Xwf_train_L, y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_,
                            'RidgeWO': GridSearchCV(Ridge(fit_intercept=False, normalize=False, 
                                                          random_state=42, 
                                                          max_iter=1000), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwof_train_L.index), iid=True).fit(Xwof_train_L, y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]).best_estimator_,
                            'RidgeW':  GridSearchCV(Ridge(fit_intercept=False, normalize=False, 
                                                          random_state=42, 
                                                          max_iter=1000), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwf_train_L.index), iid=True).fit(Xwf_train_L, y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_,
                            }
            ## RIDGE Predictions
            ridge_regWO.append(model['RidgeWO'])
            ridge_coef_WO.append(ridge_regWO[-1].coef_)
            y_predLinridgeWO.append(ridge_regWO[-1].predict(Xwof_test_L))
            axLinridgeWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)],
                                          y_predLinridgeWO[-1]]).T)
            ridge_regW.append(model['RidgeW'])
            ridge_coef_W.append(ridge_regW[-1].coef_)
            y_predLinridgeW.append(ridge_regW[-1].predict(Xwf_test_L))
            axLinridgeW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                         y_predLinridgeW[-1]]).T)
            ## LASSO Predictions
            lasso_regWO.append(model['LassoWO'])
            lasso_coef_WO.append(lasso_regWO[-1].coef_)
            y_predLinlassoWO.append(lasso_regWO[-1].predict(Xwof_test_L))
            axLinlassoWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], 
                                          y_predLinlassoWO[-1]]).T)
            lasso_regW.append(model['LassoW'])
            lasso_coef_W.append(lasso_regW[-1].coef_)
            y_predLinlassoW.append(lasso_regW[-1].predict(Xwf_test_L))
            axLinlassoW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                         y_predLinlassoW[-1]]).T)
# =============================================================================
#             ## Performances
#             ## Compare train set performance
#             trainscoreRWO.append(model['RidgeWO'].score(Xwof_train[[respL]], y_train2[resp]))
#             trainscoreLWO.append(model['LassoWO'].score(Xwof_train[[respL]], y_train2[resp]))
#             trainscoreRW.append(model['RidgeW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
#             trainscoreLW.append(model['LassoW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
#             ## Compare test set performance
#             testscoreRWO.append(model['RidgeWO'].score(Xwof_test[[respL]], y_test2[resp]))
#             testscoreLWO.append(model['LassoWO'].score(Xwof_test[[respL]], y_test2[resp]))
#             testscoreRW.append(model['RidgeW'].score(Xwf_test[[respL] + [exog]], y_test2[resp]))
#             testscoreLW.append(model['LassoW'].score(Xwf_test[[respL] + [exog]], y_test2[resp]))
# =============================================================================
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ridge_coef_WO       = np.array(ridge_coef_WO)
ridge_coef_W        = np.array(ridge_coef_W)
lasso_coef_WO       = np.array(lasso_coef_WO)
lasso_coef_W        = np.array(lasso_coef_W)
ridgeresultsWO      = []
ridgeresidWO        = []
lassoresultsWO      = []
lassoresidWO        = []
ridgeresultsW       = []
ridgeresidW         = []
lassoresultsW       = []
lassoresidW         = []
# Get the results
for i in range(0,len(y_predLinridgeW)):
    ridgeresultsWO.append(aic(axLinridgeWO[i][:,0],axLinridgeWO[i][:,1],ridge_coef_WO[i].shape[0]))
    ridgeresidWO.append(axLinridgeWO[i][:,0] - axLinridgeWO[i][:,1])
    ridgeresultsW.append(aic(axLinridgeW[i][:,0],axLinridgeW[i][:,1],ridge_coef_W[i].shape[0]))
    ridgeresidW.append(axLinridgeW[i][:,0] - axLinridgeW[i][:,1])
    lassoresultsWO.append(aic(axLinlassoWO[i][:,0],axLinlassoWO[i][:,1],lasso_coef_WO[i].shape[0]))
    lassoresidWO.append(axLinlassoWO[i][:,0] - axLinlassoWO[i][:,1])
    lassoresultsW.append(aic(axLinlassoW[i][:,0],axLinlassoW[i][:,1],lasso_coef_W[i].shape[0]))
    lassoresidW.append(axLinlassoW[i][:,0] - axLinlassoW[i][:,1])
# =============================================================================
### Mean performance of each feature
ridgeresultsWO = pd.DataFrame(np.split(np.array(ridgeresultsWO), 6), columns=list_of_responses*5, 
                              index=[list_of_responses[i] + str('_L') for i in range(len(list_of_responses))])
print('RIDGEWO: Mean AIC without feature = ', ridgeresultsWO.mean(axis=1))
ridgeresultsW  = pd.DataFrame(np.split(np.array(ridgeresultsW), 6), columns=list_of_responses*5, 
                              index=[notff3[i] + str('_L') for i in range(len(notff3))])
print('RIDGEW: Mean AIC with feature = ', ridgeresultsW.mean(axis=1))
lassoresultsWO = pd.DataFrame(np.split(np.array(lassoresultsWO), 6), columns=list_of_responses*5, 
                              index=[list_of_responses[i] + str('_L') for i in range(len(list_of_responses))])
print('LASSOWO: Mean AIC without feature = ', lassoresultsWO.mean(axis=1))
lassoresultsW  = pd.DataFrame(np.split(np.array(lassoresultsW), 6), columns=list_of_responses*5, 
                              index=[notff3[i] + str('_L') for i in range(len(notff3))])
print('LASSOW: Mean AIC with feature = ', lassoresultsW.mean(axis=1))
# =============================================================================

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
#
lasso_regWPCA       = []
lasso_coef_WPCA     = []
y_predLinlassoWPCA  = []
axLinlassoWPCA      = []
#
ridge_regWPCA       = []
ridge_coef_WPCA     = []
y_predLinridgeWPCA  = []
axLinridgeWPCA      = []
#
trainscoreRWPCA     = []
trainscoreLWPCA     = []
#
testscoreRWPCA      = []
testscoreLWPCA      = []
#
countiter           = []
modelselect2WO      = []
modelselect2W       = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[responsesL + ['ret']].iloc[train_index], regdata[responsesL + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[responsesL + ['ret'] + notff3].iloc[train_index], regdata[responsesL + ['ret'] + notff3].iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
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
            ridge_coef_WPCA.append(ridge_regWPCA[len(ridge_regWPCA)-1].coef_)
            y_predLinridgeWPCA.append(ridge_regWPCA[len(ridge_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinridgeWPCA.append(np.array([y_test2[resp], y_predLinridgeWPCA[len(ridge_regWPCA)-1]]).T)
            ## LASSO Predictions
            lasso_regWPCA.append(model['LassoWPCA'])
            lasso_coef_WPCA.append(lasso_regWPCA[len(lasso_regWPCA)-1].coef_)
            y_predLinlassoWPCA.append(lasso_regWPCA[len(lasso_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinlassoWPCA.append(np.array([y_test2[resp], y_predLinlassoWPCA[len(lasso_regWPCA)-1]]).T)
            ## Performances
            ## Compare train set performance
            trainscoreRWPCA.append(model['RidgeWPCA'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
            trainscoreLWPCA.append(model['LassoWPCA'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
            ## Compare test set performance
            testscoreRW.append(model['RidgeWPCA'].score(Xwf_testPCA[[respL] + [pc]], y_test2[resp]))
            testscoreLW.append(model['LassoWPCA'].score(Xwf_testPCA[[respL] + [pc]], y_test2[resp]))
t1 = timer()
print(t1-t0)

## Alternative scoring on test set: Adjusted R-squared
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinridgeWPCA  = np.array(axLinridgeWPCA)
y_predLinlassoWPCA  = np.array(axLinlassoWPCA)
ridge_coef_WPCA     = np.array(ridge_coef_WPCA)
lasso_coef_WPCA     = np.array(lasso_coef_WPCA)
ridgeresultsWPCA    = []
ridgeresidWPCA      = []
lassoresultsWPCA    = []
lassoresidWPCA      = []
# Get the results
for i in range(0,y_predLinridgeWPCA.shape[0]):
    ridgeresultsWPCA.append(adj_r2_score(y_predLinridgeWPCA[i,:,0],y_predLinridgeWPCA[i,:,1],ridge_coef_WPCA.shape[1]))
    ridgeresidWPCA.append(y_predLinridgeWPCA[i,:,0] - y_predLinridgeWPCA[i,:,1])
    lassoresultsWPCA.append(adj_r2_score(y_predLinlassoWPCA[i,:,0],y_predLinlassoWPCA[i,:,1],lasso_coef_WPCA.shape[1]))
    lassoresidWPCA.append(y_predLinlassoWPCA[i,:,0] - y_predLinlassoWPCA[i,:,1])
#
### Mean performance of each feature
ridgeresultsWPCA = pd.DataFrame(np.split(np.array(ridgeresultsWPCA), 6), columns=list_of_responses*5, index=notff3)
print('RIDGEWPCA: Mean of R2 score with feature = ', ridgeresultsWPCA.mean(axis=1))
lassoresultsWPCA = pd.DataFrame(np.split(np.array(lassoresultsWPCA), 6), columns=list_of_responses*5, index=notff3)
print('LASSOWPCA: Mean of R2 score with feature = ', lassoresultsWPCA.mean(axis=1))
# =============================================================================
# cols                = int(np.size(axLinridgeWPCA)/(trainmax-trainmin))
# y_predLinridgeWPCA  = np.array(axLinridgeWPCA).squeeze().reshape(trainmax-trainmin,cols)
# y_predLinlassoWPCA  = np.array(axLinlassoWPCA).squeeze().reshape(trainmax-trainmin,cols)
# ridgeresultsWPCA    = []
# axLinridgeWPCA      = []
# lassoresultsWPCA    = []
# axLinlassoWPCA      = []
# # Get the results
# for i in range(0,y_predLinridgeWO.shape[1],2):
#     ridgeresultsWPCA.append(r2_score(y_predLinridgeW[1:,i],y_predLinridgeW[:-1,i+1]))
#     lassoresultsWPCA.append(r2_score(y_predLinlassoW[1:,i],y_predLinlassoW[:-1,i+1]))
#     axLinridgeWPCA.append(np.array([y_predLinridgeW[:,i],y_predLinridgeW[:,i+1]]).T)
#     axLinlassoWPCA.append(np.array([y_predLinlassoW[:,i],y_predLinlassoW[:,i+1]]).T)
# =============================================================================
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

## Defining grid for Gridsearch cross validation ##

n_estimators      = [200]
# Server execution uses:
#[int(x) for x in np.linspace(start = 50, stop = 500, num = 3)]

# Number of features to consider at every split
max_features      = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth         = [10, None] 
# Server execution uses:
# [int(x) for x in np.linspace(10, 100, num = 3)]
# max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2,10,20] 
# Minimum number of samples required at each leaf node
min_samples_leaf  = [2,10,20] 
# Method of selecting samples for training each tree
bootstrap         = [True] 
# Create the random grid
random_grid       = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}
# Define Random Forest Regressor as estimator for regression
RFR              = RandomForestRegressor(oob_score=True) 

## RANDOM FOREST ####
#
forest_regWO     = []
axRFRWO          = []
ypredforestWO    = []
#
forest_regW      = []
axRFRW           = []
ypredforestW     = []
#
trainscore_forestWO = []
trainscore_forestW  = []
testscore_forestWO  = []
testscore_forestW   = []
#
countiter         = []
modelselectWORFR  = []
modelselectWRFR   = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(range(i-trainmin,i))
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[list_of_responses + ['ret']].iloc[train_index], regdata[list_of_responses + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[list_of_responses + ['ret'] + notff3].iloc[train_index], regdata[list_of_responses + ['ret'] + notff3].iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
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
    for resp in list_of_responses:
        ## Model Selection
        #countiter.append(y_test2.columns) # Just to count the iterations
        #modelselectWORFR  = [np.max([1,modelselectWO[i]]) for i in range(len(countiter))]
        #modelselectWRFR   = [np.max([1,modelselectW[i]]) for i in range(len(countiter))]
        # Define lagged X w.r.t AIC
        Xwof_train_L    = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],
                                                    maxlag0=1,trim='forward', 
                                                    dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:1],
                                                                                    columns=Xwof_train[[resp] + ['ret']].columns[0])
        Xwf_train_L     = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + notff3],
                                                    maxlag0=1,trim='forward', 
                                                    dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + notff3].index[:1],
                                                                                    columns=Xwf_train[[resp] + ['ret'] + notff3].columns[0])
        Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']],
                                                    maxlag0=1,trim='forward', 
                                                    dropex=1, use_pandas=True).drop(index=Xwof_test[[resp] + ['ret']].index[:1],
                                                                                    columns=Xwof_test[[resp] + ['ret']].columns[0])
        Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + notff3],
                                                    maxlag0=1,trim='forward', 
                                                    dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + notff3].index[:1],
                                                                                    columns=Xwf_test[[resp] + ['ret'] + notff3].columns[0])
        # Train models
        model   = {'RFRWO': GridSearchCV(RFR, param_grid=random_grid, 
                                       scoring='neg_mean_squared_error', 
                                       cv=tsplit.split(Xwof_train_L.index),
                                       iid=True, n_jobs=-1).fit(Xwof_train_L, 
                                                                y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]).best_estimator_,
                   'RFRW': GridSearchCV(RFR, param_grid=random_grid, 
                                        scoring='neg_mean_squared_error',
                                        cv=tsplit.split(Xwf_train_L.index),
                                        iid=True, n_jobs=-1).fit(Xwf_train_L, 
                                                                 y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_}
        ## Random Forest Regression
        forest_regWO.append(model['RFRWO'].fit(Xwof_train_L, 
                            y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
        ypredforestWO.append(forest_regWO[len(forest_regWO)-1].predict(Xwof_test_L))
        axRFRWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], 
                                 ypredforestWO[len(forest_regWO)-1]]).T)
        forest_regW.append(model['RFRW'].fit(Xwf_train_L, 
                            y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ypredforestW.append(forest_regW[len(forest_regW)-1].predict(Xwf_test_L))
        axRFRW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                ypredforestW[len(forest_regW)-1]]).T)
        ## Performances
        ## Compare train set performance
        trainscore_forestWO.append(model['RFRWO'].score(Xwof_train_L, 
                                                        y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
        trainscore_forestW.append(model['RFRW'].score(Xwf_train_L, 
                                                      y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ## Compare test set performance
        testscore_forestWO.append(model['RFRWO'].score(Xwof_test_L, 
                                                       y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)]))
        testscore_forestW.append(model['RFRW'].score(Xwf_test_L, 
                                                     y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
RFRresultsWO = []
RFRresidWO   = []
RFRresultsW  = []
RFRresidW    = []
# Get the results
for i in range(0,len(axRFRWO)):
    RFRresultsWO.append(aic(axRFRWO[i][:,0],axRFRWO[i][:,1], forest_regWO[i].n_features_))
    RFRresidWO.append(axRFRWO[i][:,0] - axRFRWO[i][:,1])  
    RFRresultsW.append(aic(axRFRW[i][:,0],axRFRW[i][:,1],forest_regW[i].n_features_))
    RFRresidW.append(axRFRW[i][:,0] - axRFRW[i][:,1]) 
# =============================================================================
### Mean performance of each feature
RFRresultsWO = pd.DataFrame(np.split(np.array(RFRresultsWO), 6), columns=list_of_responses*5, index=[list_of_responses[i] + str('_L') for i in range(len(list_of_responses))])
print('Mean AIC without feature = ', linresultsWO.mean(axis=1))
RFRresultsW  = pd.DataFrame(np.split(np.array(RFRresultsW), 6), columns=list_of_responses*5, index=[notff3[i] + str('_L') for i in range(len(notff3))])
print('Mean AIC with feature = ', linresultsW.mean(axis=1))
# =============================================================================
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
for i in range(0,ypredforestWPCA.shape[1],2):
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
