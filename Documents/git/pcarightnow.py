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

# %% Loading data ############################################################
data         = pd.read_csv('./Data/letsdodis2.csv')
data.index   = pd.to_datetime(data.date)
data2        = pd.read_csv('./Data/FUT86_18all.txt',sep='\t', header=0)
data2.index  = pd.to_datetime(data2.values[:,2])
data2        = data2.groupby(pd.Grouper(freq='M')).last()
Sdata        = pd.read_csv('./Data/FamaFrench5Factors2.csv')
Sdata.index  = Sdata.values[:,0]
Pdata        = pd.read_csv('./Data/6_Portfolios_2x3_Wout_Div.csv', index_col=0)/100
vixdata      = pd.read_csv('./Data/VIX94_18.csv', header=1)
vixdata.index= pd.to_datetime(vixdata.values[:,0])
vixdata      = vixdata.groupby(pd.Grouper(freq='M')).last()

# Each month of the CEFD data represents the mean of discounts across funds
# Each month of the COT data represents the last aggregate figure of the month
# since COT reports are published weekly. 
# Each month of the SP500 data is just the monthly return (value-weighted index)
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
# Small Cap Portfolio returns
Pret        = np.array(Pdata.values[:,0],dtype=float)
Pret = Pret[1:]

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
cefd         = np.array(-b/a,dtype=float) # Closed-end fund discount
cefd         = cefd[1:] - cefd[:-1]

####    DataFraming by investment measure category
dfCOT        = pd.concat([pd.DataFrame(CPNL),pd.DataFrame(NCPNL),pd.DataFrame(TOTPNL)],axis=1)
dfCOT.columns= range(3)
for i in range(3):
    dfCOT[i] = np.where(abs(dfCOT[i])>3*np.std(dfCOT[i]), np.nan, dfCOT[i])
    dfCOT    = dfCOT.ffill()

cefd        = pd.DataFrame(cefd)
vixret      = pd.DataFrame(vixret)
famafrench  = pd.DataFrame([Pret, ret, smb, hml]).T



####    Dickey Fuller test for unit root in variables ########################
adfuller    = pd.DataFrame()
dfall       = pd.concat([famafrench, cefd, dfCOT, vixret],axis=1)
dfall.columns   = np.arange(9)
for i in dfall.columns:
    adfuller[i] = sm.tsa.stattools.adfuller(dfall[i], regression="c")
adfuller.columns=['Pret','ret','smb','hml','cefd','CPNL','NCPNL','TOTPNL','vixret']
dfall.columns=['Pret','ret','smb','hml','cefd','CPNL','NCPNL','TOTPNL','vixret']
# Since all variables are stationary, we can perform OLS tests. Stationarity assumption holds.


##############################################################################
##############################################################################    
# %% OLS REGRESSIONS ##########################################################

#### Regression 1   
Pdata2            = pd.DataFrame(data=np.roll(Pdata.drop(index=199401), -1, axis=0))
Pdata2.index      = range(len(Pdata2))
list_of_responses = ["SMALLLoBM", "ME1BM2", "SMALLHiBM", "BIGLoBM", "ME2BM2", "BIGHiBM"]
Pdata2.columns    = list_of_responses
regdata           = pd.concat([Pdata2,dfall],axis=1,ignore_index=False)
regdata.columns   = np.append(list_of_responses,dfall.columns.values)

# list of models
reg1 = []

for resp in list_of_responses:
    formula = resp + " ~ cefd + ret"
    reg1.append(sm.OLS.from_formula(formula, data=regdata).fit())
    print(reg1[reg1.count(resp).summary())

#       Creating X matrix
X           = np.array(cefd, dtype=float)

#       Add constant
X           = sm.add_constant(X)
X           = X.astype(float)

#       Regress
reg0        = sm.OLS(Pret[1:],X[:-1], missing='drop')
results0    = reg0.fit()

print(results0.summary(yname='ret', 
                         xname=['const','cefd']))

####    Breusch Pagan test for heteroscedasticity
ols_resid0   = results0.resid
#bp0          = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp0          = sm.stats.diagnostic.het_breuschpagan(ols_resid0,X[:-1])

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results0.summary(yname='Pret', 
                         xname=['const','cefd'])), 
{'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('results0.png')
"""

#### Regression 2

#       Recreating X matrix
X           = np.array(dfCOT, dtype=float)
Xcorr1      = np.corrcoef(X,rowvar=0) # Correlation matrix 
#       Add constant
X           = sm.add_constant(X)
X           = X.astype(float)

#       Regress
reg         = sm.OLS(Pret[1:],X[:-1])
results     = reg.fit()

print(results.summary(yname='Pret', 
                         xname=['const','CPNL','NCPNL','TOTPNL']))

####    Breusch Pagan test for heteroscedasticity
ols_resid    = results.resid
bp           = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp           = sm.stats.diagnostic.het_breuschpagan(ols_resid,bp)

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results.summary(yname='Pret', 
                         xname=['const','CPNL','NCPNL','OI','TOTPNL','vixret'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('results.png')
"""
#### Regression 3
#       Recreating X matrix

X            = np.array(pd.concat([cefd,dfCOT,vixret],axis=1), dtype=float)
Xcorr2       = np.corrcoef(X[np.nonzero(~pd.isnull(X[:,0]))],rowvar=0) 

#       Add constant
X            = sm.add_constant(X)
X            = X.astype(float)

#       Regress
reg2         = sm.OLS(Pret[1:],X[:-1])
results2     = reg2.fit()

print(results2.summary(yname='Pret', 
                       xname=['const','cefd','CPNL','NCPNL','TOTPNL','vixret']))
                      # xname=['const','cefd', 'hml','CPNL','NCPNL','vixret']))

####    Breusch Pagan test for heteroscedasticity
ols_resid2   = results2.resid
bp2          = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bp2          = sm.stats.diagnostic.het_breuschpagan(ols_resid2,bp2)

"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(results2.summary(yname='Pret', 
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

# %% GLS MODEL ###############################################################
ols_resid2  = results2.resid
res_fit     = sm.OLS(ols_resid2[1:], ols_resid2[:-1]).fit()
rho         = res_fit.params
# rho is a consistent estimator of the correlation of the residuals from
# an OLS fit of the longley data.  It is assumed that this is the true rho
# of the AR process data.

from scipy.linalg import toeplitz

nbyn        = np.arange(X.shape[0]-1,dtype=np.int32)
order       = np.array(toeplitz(nbyn),dtype=np.int32) 
sigma       = np.array(rho**order,dtype=np.float32)
# sigma is an n x n matrix of the autocorrelation structure of the data.

gls_model   = sm.GLS(Pret[1:], X[:-1], missing='drop', sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary(yname='Pret', 
                         xname=['const','cefd','CPNL','NCPNL','TOTPNL','vixret']))

####    Breusch Pagan test for heteroscedasticity
gls_resid    = gls_results.resid
bpG          = X[np.nonzero(~pd.isnull(X[:-1,1]))]
bpG          = sm.stats.diagnostic.het_breuschpagan(gls_resid,bpG)
##############################################################################
##############################################################################

from statsmodels.tsa.api import VAR, DynamicVAR

model = VAR(dfall)
resss = model.fit(2)
print(resss.summary())

# %% LOGIT ####################################################################

# Transform positive returns --> logical
retpos  = Pret > 0
retpos  = retpos.astype(np.int)

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
                         xname=['const','CPNL','NCPNL','TOTPNL']))

print(logitmfx.summary())

# X defined as [ones, cefd, CPNL, ...]
logit2    = sm.Logit(retpos[1:],X[:-1], missing='drop') # joint, all
logitres2 = logit2.fit()
logit2mfx = logitres2.get_margeff(at='overall')

print(logitres2.summary2(yname='retpos', 
                         xname=['const','cefd','CPNL','NCPNL','TOTPNL','vixret']))

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

# %% MACHINE LEARNING LOGISTIC REGRESSION #####################################
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate, \
                                    TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mglearn
import graphviz
from sklearn.tree.export import export_graphviz

# Define X using variables subject to pca. Include dependent variables.
Xdf                = np.array(X)
Xdf                = np.append(Xdf,famafrench.values[:,1].reshape(-1,1), axis=1)
#Xdf                = famafrench.values[:,1:]

Xdf                = pd.DataFrame(Xdf)
Xdf                = Xdf.drop(columns=6)

#Xdf                = pd.DataFrame(X)

# TimeSeriesSplit --> convert into dataframe to concatenate
retposdf           = pd.DataFrame(retpos)
retdf              = pd.DataFrame(Pret)
Xdf                = pd.concat([Xdf,retposdf, retdf],axis=1)
Xdf.index          = range(299)
Xdf.columns        = np.arange(Xdf.values.shape[1])
#tsplit             = TimeSeriesSplit(n_splits=3, max_train_size=200)
kf                 = KFold(n_splits=10,shuffle=False, random_state=42)

# Split into training and test subsets

for train_index, test_index in kf.split(Xdf.index):
    X_train, X_test   = Xdf.values[train_index,:-2], Xdf.values[test_index,:-2]
    y_train1, y_test1 = Xdf.values[train_index,-2], Xdf.values[test_index,-2]
    y_train2, y_test2 = Xdf.values[train_index,-1], Xdf.values[test_index,-1]
    
# Split into training and test subsets
for train_index, test_index in tsplit.split(Xdf.index):
    print(train_index,test_index)
    X_train, X_test   = Xdf.values[train_index,:-2], Xdf.values[test_index,:-2]
    y_train1, y_test1 = Xdf.values[train_index,-2], Xdf.values[test_index,-2]
    y_train2, y_test2 = Xdf.values[train_index,-1], Xdf.values[test_index,-1]

########## LogisticRegression
log_reg    = LogisticRegression(fit_intercept=1, solver='lbfgs')
logresults = log_reg.fit(X_train, y_train1)
y_pred1    = logresults.predict(X_test)
logresults.score(X_test[:], y_test1[:]) 
plt.scatter(y_pred1,y_test1)
# Mean accuracy is 71.4%

########## LinearRegression
lin_reg     = LinearRegression(fit_intercept=True)
linresults2 = lin_reg.fit(X_train, y_train2)
y_pred2     = linresults2.predict(X_test)
r2_score(y_test2,y_pred2)
plt.plot(dates[test_index],np.array([y_test2,y_pred2]).T)

cross_val_score(LinearRegression(fit_intercept=1), Xdf.values[:,:-2], 
                                  Pret, cv=10, scoring='r2').mean()
cross_validate(LinearRegression(fit_intercept=1), Xdf.values[:,:-2], 
                                  Pret, cv=10, scoring='r2')
# Mean Rsquared is 24%. 
#################

##############################################################################
##############################################################################

# %% PRINCIPAL COMPONENTS ANALYSIS ###########################################
#X_pca       = normalize(Xdf.values[:,:-2]) # Normalize regressors
scaler      = StandardScaler()
scalefit    = scaler.fit(X_train[:,1:]) # Standardized variables
X_trainS    = scalefit.transform(X_train[:,1:])
X_testS     = scalefit.transform(X_test[:,1:])
pca         = PCA(n_components=5, whiten=1) # PCA Model

X_stand     = pd.concat([pd.DataFrame(X_trainS),
                         pd.DataFrame(X_testS)],axis=0)

pca.fit(X_trainS) # Fit
X_pca1      = pca.transform(X_trainS) # transform
X_pca2      = pca.transform(X_testS) # transform
X_pca1      = sm.add_constant(X_pca1)
X_pca2      = sm.add_constant(X_pca2)
########## LogisticRegression
log_reg     = LogisticRegression(fit_intercept=1, solver='liblinear')
logresults2 = log_reg.fit(X_pca1, y_train1)
logresults2.score(X_pca2, y_test1)
# Mean accuracy is 75%

########## LinearRegression
lin_reg2    = LinearRegression(fit_intercept=1)
linresults2 = lin_reg2.fit(X_pca1, y_train2)
linresults2.score(X_pca2, y_test2)
y_pred2     = linresults2.predict(X_pca2)
r2_score(y_test2,y_pred2)
plt.plot(dates[test_index], np.array([y_test2,y_pred2]).T)
sns.regplot(y_pred2,y_test2)

cross_val_score(LinearRegression(fit_intercept=1), X_pca3.values, 
                                  Pret, cv=10, scoring='r2').mean()
# Rsquared is 6.1%. 



############################################################################

####    Regress LOGIT using sklearn pcs
X_pca3          = pd.concat([pd.DataFrame(X_pca1),
                             pd.DataFrame(X_pca2)], axis=0)
X_pca3.index    = Xdf.index
X_pca3.columns  = np.arange(X_pca3.values.shape[1])
X_pca3corr      = np.corrcoef(X_pca3.values[:,1:],rowvar=0)


logit3          = sm.Logit(retpos[1:],X_pca3[:-1],missing='drop') # joint, all
logitres3       = logit3.fit()
print(logitres3.summary2())
logit3mfx       = logitres3.get_margeff(at='overall')
print(logit3mfx.summary())

####    Regress OLS using sklearn pcs
reg4         = sm.OLS(Pret[1:], X_pca3[:-1])
results4     = reg4.fit()

print(results4.summary())

####    Breusch Pagan
ols_resid4   = results4.resid
bp4          = sm.stats.diagnostic.het_breuschpagan(ols_resid4,
                                                    X_pca3.values[:-1,1:])

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

#%%    STATSMODELS PRINCIPAL COMPONENTS ANALYSIS #############################
smPCA       = sm.PCA(X[:,1:], standardize=1)

X_pca5      = sm.add_constant(smPCA.scores)
X_pca5corr  = np.corrcoef(X_pca5[:,1:],rowvar=0) 
sm_pca      = sm.OLS(Pret[1:],X_pca5[:-1]) # joint, all
sm_pcares   = sm_pca.fit()
print(sm_pcares.summary())

####    Breusch Pagan
ols_resid5   = sm_pcares.resid
bp5          = sm.stats.diagnostic.het_breuschpagan(ols_resid5,X_pca5[:-1,1:])

smPCA.plot_scree()
smPCA.plot_rsquare()
sns.regplot(smPCA.scores[:,0],Pret[:])
sns.regplot(smPCA.scores[:,0], X[:,3])
##############################################################################
##############################################################################

#%%    COMPARING STATSMODELS vs. SKLEARN PCA     #############################
X_pca6       = X_pca3

for i in range(6,11):
    X_pca6   = pd.concat([X_pca6, pd.DataFrame(smPCA.scores[:,i-6], index=X_pca6.index)], axis=1)
    i        = i + 1 

X_pca6.columns = np.arange(X_pca6.values.shape[1])

#       Add constant 
X_pca6corr   = np.corrcoef(np.array(X_pca6.values[:,1:] , dtype=float),rowvar=0) 
#       Regress
sm_pca2      = sm.OLS(Pret[1:],X_pca6.values[:-1])
sm_pcares2   = sm_pca2.fit()

print(sm_pcares2.summary())

###############################################################################
smpca_resid = sm_pcares2.resid
res_fit     = sm.OLS(smpca_resid[1:], smpca_resid[:-1]).fit()
rho         = res_fit.params

from scipy.linalg import toeplitz

nbyn        = np.arange(X_pca6.values.shape[0]-1,dtype=np.int32)
order       = np.array(toeplitz(nbyn),dtype=np.int32) 
sigma       = np.array(rho**order,dtype=np.float32)
# sigma is an n x n matrix of the autocorrelation structure of the data.

gls_model   = sm.GLS(ret[1:], X_pca6.values[:-1], missing='drop', sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary(yname='ret', 
                          xname=['const','pc1','pc2','pc3','pc4', 'pc5']))
###############################################################################
###############################################################################

#%% RECURSIVE FEATURE ELIMINATION #############################################
from mpl_toolkits.mplot3d import Axes3D
Axes3D.plot_surface(Xna[:,1:])
for train_index, test_index in kf.split(X_pca6.index):
    X_train, X_test   = X_pca6.values[train_index,1:], X_pca6.values[test_index,1:]

model        = LinearRegression(fit_intercept=1)
rfe          = RFE(model,3,verbose=1)

fit          = rfe.fit(X_train,y_train2)
fit.ranking_
fit.support_

Xnatrain     = fit.transform(X_train)
Xnatrain     = sm.add_constant(Xnatrain)
Xna          = fit.transform(X_test)
Xna          = sm.add_constant(Xna)
#X_nacorr     = np.corrcoef(Xna[:,1:],rowvar=0)
reg4         = sm.OLS(y_test2[1:], Xna[:-1])
results4     = reg4.fit()

print(results4.summary())

########## Breusch Pagan
ols_resid5   = results4.resid
bp5          = sm.stats.diagnostic.het_breuschpagan(ols_resid5,Xna[:-1])

########## LinearRegression

lin_reg               = LinearRegression(fit_intercept=1)
linresultsrfe         = lin_reg.fit(Xnatrain, y_train2)
linresultsrfe.score(Xna, y_test2)
# Rsquared is 3.2%

log_reg              = LogisticRegression(fit_intercept=1, solver='lbfgs')
logresultsrfe        = log_reg.fit(Xnatrain, y_train1)
logresultsrfe.score(Xna, y_test1)
# Mean accuracy is 72.4%
#%% RANDOM FOREST FEATURE SELECTION ###########################################
##DESICION TREE

tree = DecisionTreeClassifier(random_state=0, max_depth=None)
tree.fit(X_train, y_train1)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train1)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test1)))
# Since the score is <1 the model is not overfitting..


export_graphviz(tree, out_file='retpostree.dot', class_names=['Enter', 'Hold'],
                feature_names=['cefd','CPNL','NCPNL','TOTPNL','vixret'],
                impurity=False, filled=True)

n_features = X.shape[1]-1
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), ['cefd','CPNL','NCPNL','TOTPNL','vixret'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

## RANDOM FOREST

forest = RandomForestClassifier(n_estimators=10,random_state=0)
forest.fit(X_train,y_train1)
print('Accuracy on the training subset: {:.3f}'.format(forest.score(X_train, y_train1)))
print('Accuracy on the test subset: {:.3f}'.format(forest.score(X_test, y_test1)))

# Again, not overfitting 

plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), ['cefd','CPNL','NCPNL','OI','TOTPNL','vixret'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Now more features play a role (nonzero importance) in the decision making!

# Visualization, 100 trees:
i_tree = 0
for tree_in_forest in forest.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as retpostree2:
        retpostree2 = export_graphviz(tree_in_forest, out_file = 'retpostree2.dot')
    i_tree = i_tree + 1
    
