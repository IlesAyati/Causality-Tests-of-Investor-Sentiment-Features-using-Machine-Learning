# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:59:20 2019

@author: yeeya
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
smPC            = sm.PCA(dfall, standardize=1, method='svd')
smPCcorr        = smPC.scores.corr()
dfallPC         = smPC.scores