# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:32:38 2019

@author: yeeya
"""

# %% MACHINE LEARNING  ######################################################
from statsmodels.tsa.vector_ar.var_model1 import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import f_regression
from scipy.stats import normaltest, yeojohnson
from timeit import default_timer as timer
#from sklearn.tree.tree import DecisionTreeClassifier
#from sklearn.tree.export import export_graphviz
#import mglearn
#import graphviz
# =============================================================================
# AIC to measure the forecasts
def aic(y, y_pred, k):
   resid = np.array([y - y_pred]).T
   rss   = np.sum(resid**2)
   AIC   = 2*k - 2*len(y)*np.log(rss/len(y))
   return AIC
# F-test to compare restricted and unrestricted models
def F(y1,y1_pred,y2, y2_pred,p1,p2):
    resid1 = np.array([y1 - y1_pred]).T
    rss1   = np.sum(resid1**2)
    resid2 = np.array([y2 - y2_pred]).T
    rss2   = np.sum(resid2**2)
    F_stat = ((rss1-rss2)/(p2-p1)/(rss2/(len(y2)-p2)))
    return F_stat
# =============================================================================
tsplit          = TimeSeriesSplit(n_splits=5, max_train_size=250)
tsplit2         = TimeSeriesSplit(n_splits=3)
pca             = PCA(n_components=3, whiten=1, random_state=42)
scaler          = StandardScaler()
scaler2         = StandardScaler()
# =============================================================================