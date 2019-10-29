# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:32:38 2019

@author: yeeya
"""

# %% MACHINE LEARNING  ######################################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
#from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
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
pca             = PCA(n_components=0.95, whiten=1, random_state=42)
scaler          = StandardScaler()
# =============================================================================
