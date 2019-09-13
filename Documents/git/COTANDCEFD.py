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
%matplotlib inline
import seaborn as sns

#### Loading data ############################################################
data         = pd.read_csv('letsdodis2.csv')
data.index   = pd.to_datetime(data.date)
data2        = pd.read_csv('FUT86_18all.txt',sep='\t', header=0)
data2.index  = pd.to_datetime(data2.values[:,2])
data2        = data2.groupby(pd.Grouper(freq='M')).last()
Sdata        = pd.read_csv('FamaFrench5Factors2.csv')
Sdata        = Sdata.drop(Sdata.index[0:288])
vixdata      = pd.read_csv('VIX94_18.csv', header=1)
vixdata.index= pd.to_datetime(vixdata.values[:,0])
vixdata      = vixdata.groupby(pd.Grouper(freq='M')).last()

# Each month of the CEFD data represents the mean of discounts across funds
# Each month of the COT data represents the last aggregate figure of the month
# since COT reports are published weekly. 
# Each month of the SP500 data is just the monthly return (value-weighted index)
##############################################################################

#### Data Management #########################################################

#   S&P 500 Index Value Weighted Returns
ret          = np.array(Sdata.values[:,1],dtype=float)
hml          = np.array(Sdata.values[:,3],dtype=float)

#   VIX Price to Returns
vixret       = vixdata.values[:,-1]
vixret       = np.array(vixret[1:]/vixret[:-1]-1,dtype=float)

#### Slicing COT dataset by variables
oi           = data2.values[:,7] # Open Interest
nclong       = data2.values[:,8] # Non-commercial long
ncshort      = data2.values[:,9] # -----||------- short
ncspread     = data2.values[:,10] # -----||-------spread
clong        = data2.values[:,11] # Commercial long
cshort       = data2.values[:,12] # Commercial short
totreplong   = data2.values[:,13] # Total reportables long
totrepshort  = data2.values[:,14] # Total reportables short
nonreplong   = data2.values[:,15] # Total Non-reportables long
nonrepshort  = data2.values[:,16] # Total Non-reportables short

#   Creating the Pressure Net Long Variables
OI           = oi[1:]/oi[:-1]-1 # Percentage change in Open Interest

NCPNL        = (nclong-ncshort)/(nclong+ncshort) # Non-Commercial Pressure Net
NCPNL        = np.array(NCPNL[1:]/NCPNL[:-1]-1,dtype=float) # Change in NCPNL
CPNL         = (clong-cshort)/(clong+cshort) # Commercial Pressure Net Long
CPNL         = np.array(CPNL[1:]/CPNL[:-1]-1,dtype=float) # Change in CPNL
TOTPNL       = (totreplong-totrepshort)/(totreplong+totrepshort) #TotRep PNL
TOTPNL       = np.array(TOTPNL[1:]/TOTPNL[:-1]-1,dtype=float) # Change in TotRep PNL


####    Slicing CEFD dataset
a            = data.navm.interpolate(method='linear')
b            = data.discount.interpolate(method='linear')
cefd          = np.array(-b/a,dtype=float) # Closed-end fund discount

####    Dickey Fuller test for unit root in variables

adfuller = pd.DataFrame()
X        = np.array([cefd[1:],CPNL,NCPNL,OI,TOTPNL,vixret,hml[1:]], dtype=float).T

for i in range(6):
    adfuller[i] = sm.tsa.stattools.adfuller(X[:-1,i], regression="c")
    i           = i + 1
    
# Since all variables are stationary, we can perform OLS tests. Stationarity assumption holds.

#### OLS REGRESSIONS ##########################################################
####                 ##########################################################
#### Regression 1   
    
#       Creating X matrix
X           = np.array([cefd], dtype=float).T

#       Add constant
X           = sm.add_constant(X)
X           = X.astype(float)

#       Regress
reg0        = sm.OLS(ret[1:],X[:-1], missing='drop')
results0    = reg0.fit()

print(results0.summary(yname='ret', 
                         xname=['const','cefd']))

####    Breusch Pagan
ols_resid0   = results0.resid
bp0          = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp0          = sm.stats.diagnostic.het_breuschpagan(ols_resid0,bp0)

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results0.summary(yname='ret', 
                         xname=['const','cefd'])), 
{'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('results0.png')
"""

#### Regression 2

#       Recreating X matrix
X           = np.array([CPNL,NCPNL,OI,TOTPNL,vixret, hml[1:]], dtype=float).T
Xcorr1      = np.corrcoef(X,rowvar=0) # Correlation matrix 
#       Add constant
X           = sm.add_constant(X)
X           = X.astype(float)

#       Regress
reg         = sm.OLS(ret[2:],X[:-1])
results     = reg.fit(cov_type='HC0')

print(results.summary(yname='ret', 
                         xname=['const','CPNL','NCPNL','OI','TOTPNL','vixret', 'hml']))

####    Breusch Pagan
ols_resid    = results.resid
bp           = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp           = sm.stats.diagnostic.het_breuschpagan(ols_resid,bp)

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results.summary(yname='ret', 
                         xname=['const','CPNL','NCPNL','OI','TOTPNL','vixret'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('results.png')
"""
#### Regression 3
#       Recreating X matrix

X            = np.array([cefd[1:],CPNL,NCPNL,OI,TOTPNL,vixret, hml[1:]], dtype=float).T
Xcorr2       = np.corrcoef(X[np.nonzero(~pd.isnull(X[:,0]))],rowvar=0) 

#       Add constant
X            = sm.add_constant(X)
X            = X.astype(float)

#       Regress
reg2         = sm.OLS(ret[2:],X[:-1], missing='drop')
results2     = reg2.fit()

print(results2.summary(yname='ret', 
                         xname=['const','cefd','CPNL','NCPNL', 'OI', 'TOTPNL','vixret', 'hml']))

####    Breusch Pagan
ols_resid2   = results2.resid
bp2          = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp2          = sm.stats.diagnostic.het_breuschpagan(ols_resid2,bp2)

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results2.summary(yname='ret', 
                         xname=['const','cefd','CPNL','NCPNL','OI','TOTPNL','vixret'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('results2.png')
"""
#       Plots
#sns.regplot(X[:,3],ret[1:])
##############################################################################
##############################################################################

#### GLS MODEL ###############################################################
ols_resid0  = results2.resid
res_fit     = sm.OLS(ols_resid0[1:], ols_resid0[:-1]).fit()
rho         = res_fit.params
# rho is a consistent estimator of the correlation of the residuals from
# an OLS fit of the longley data.  It is assumed that this is the true rho
# of the AR process data.

from scipy.linalg import toeplitz

nbyn        = np.arange(X.shape[0]-1,dtype=np.int32)
order       = np.array(toeplitz(nbyn),dtype=np.int32) 
sigma       = np.array(rho**order,dtype=np.float32)
# sigma is an n x n matrix of the autocorrelation structure of the data.

gls_model   = sm.GLS(ret[2:], X[:-1], missing='drop', sigma=sigma)
gls_results = gls_model.fit(cov_type='HC0')
print(gls_results.summary(yname='ret', 
                         xname=['const','cefd','CPNL','NCPNL','OI','TOTPNL','vixret']))
##############################################################################
##############################################################################

### LOGIT ####################################################################

# Transform positive returns --> logical
retpos  = ret>0
retpos  = retpos.astype(np.int)
retpos  = retpos[1:]
# X defined as [ones, cefd]
logit0    = sm.Logit( retpos, X[:-1], missing='drop') # Only cefd
logitres0 = logit0.fit()
logit0mfx = logitres0.get_margeff(at='overall')

print(logitres0.summary2(yname='retpos', 
                         xname=['const','cefd']))
print(logit0mfx.summary())


# X defined as [ones, CPNL, ...]
logit  = sm.Logit(retpos[1:],X[:-1]) # joint, excluding cefd
logitres = logit.fit()
logitmfx = logitres.get_margeff(at='overall')

print(logitres.summary2(yname='retpos', 
                         xname=['const','CPNL','NCPNL','OI','TOTPNL','vixret']))

print(logitmfx.summary())

# X defined as [ones, cefd, CPNL, ...]
logit2    = sm.Logit(retpos[1:],X[:-1], missing='drop') # joint, all
logitres2 = logit2.fit()
logit2mfx = logitres2.get_margeff(at='overall')

print(logitres2.summary2(yname='retpos', 
                         xname=['const','cefd','CPNL','NCPNL','OI','TOTPNL','vixret','hml']))

print(logit2mfx.summary())
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

#### MACHINE LEARNING LOGISTIC REGRESSION #####################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X[:,1:], retpos[1:], 
                                                    stratify=None, shuffle=0, 
                                                    random_state=42)

log_reg    = LogisticRegression(fit_intercept=1)
logresults = log_reg.fit(X_train[:-1], y_train[1:])
logresults.score(X_test[:-1],y_test[1:])
##############################################################################
##############################################################################

#### PRINCIPAL COMPONENTS ANALYSIS ###########################################
#scaler      = StandardScaler()
X_pca       = normalize(X[:,1:])
#X_pca       = scaler.fit_transform(X[:,1:]) # Standardize variables
pca         = PCA(n_components='mle') # PCA Model

#pca.fit(X_pca) # Fit 
#X_pca       = np.array(pca.transform(X_pca), dtype=float) # transform

################## 
X_train, X_test, y_train, y_test = train_test_split(X_pca, retpos[1:], 
                                                    stratify=None, shuffle=0, 
                                                    random_state=42)

pca.fit(X_train) # Fit
X_pca1      = np.array(pca.transform(X_train), dtype=float) # transform
X_pca2      = np.array(pca.transform(X_test), dtype=float) # transform
log_reg2    = LogisticRegression(fit_intercept=1, solver='lbfgs')
logresults2 = log_reg2.fit(X_pca1[:-1], y_train[1:])
logresults2.score(X_pca2[:-1],y_test[1:])
#################

####    logit using pcs
#X_pca3      = sm.add_constant(X_pca2)
logit3      = sm.Logit(y_test,X_pca3) # joint, all
logitres3   = logit3.fit()
print(logitres3.summary2())
logit3mfx   = logitres3.get_margeff(at='overall')
print(logit3mfx.summary())
pca_ratios  = plt.bar(range(logitres3.params.shape[0]-1),pca.explained_variance_ratio_)

sns.regplot(xx[:-1,1],retpos[2:], logistic=1)
sns.regplot(X_pca3[:-1,1],y_test[1:], logistic=1)
plt.scatter(X_pca3[:,1],xx[:,1])


"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(logitres3.summary2()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logitrespca.png')
"""
"""
plt.rc('figure', figsize=(7, 3))
plt.text(0.01, 0.05, str(logit3mfx.summary()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logit3mfx.png')
"""
#       Regress OLS
X_pca3       = pd.DataFrame(X_pca3)
X_pca1       = pd.DataFrame(sm.add_constant(X_pca1))
X_pcaOLS     = pd.concat([X_pca3,X_pca1], axis=0)
reg3         = sm.OLS(ret[2:],smPCA.scores[:-1])
results3     = reg3.fit()

print(results3.summary(yname='ret', 
                         xname=['const','pc1','pc2','pc3','pc4','pc5','pc6']))

####    Breusch Pagan
ols_resid3   = results3.resid
bp3          = sm.stats.diagnostic.het_breuschpagan(ols_resid3,smPCA.scores[:-1])
##############################################################################
##############################################################################
smPCA = sm.PCA(X[:,1:], standardize=1, method='svd')

X_pca4      = sm.add_constant(smPCA.scores)
logit4      = sm.Logit(retpos[2:],smPCA.scores[:-1]) # joint, all
logitres4   = logit4.fit()
print(logitres4.summary2())

sns.regplot(smPCA.scores[:-1,1],retpos[2:], logistic=1)
plt.scatter(smPCA.scores[:,0], xx[:,1])