# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:32:38 2019

@author: yeeya
"""

# %% MACHINE LEARNING  ######################################################
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import ttest_ind, f_oneway 
from skits.pipeline import ForecasterPipeline, FeatureUnion
from skits.preprocessing import ReversibleImputer
from skits.feature_extraction import AutoregressiveTransformer                     
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
######             ########################################################### 
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

# Define splits. 
# Cross validation uses a 5 fold TimeSeriesSplit
# For testing, the first 270 observations are used, expanded by one for each iteration
trainmin          = 270
trainmax          = len(regdata.values)
tsplit            = TimeSeriesSplit(n_splits=5, max_train_size=270)
##############################################################################
