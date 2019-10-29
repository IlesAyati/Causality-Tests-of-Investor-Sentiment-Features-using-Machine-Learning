# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:44:22 2019

@author: yeeya
"""

#%%    RIDGE AND LASSO  REGRESSIONS  ##########################################
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
            # Pick the same amount of lags as in OLS:
            # Counts iteration number
            countiter.append(y_test2.columns) 
            # Add number of lags corresponding to iteration number from OLS
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
            ridge_regWO.append(model['RidgeWO'].fit(Xwof_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
            ridge_coef_WO.append(ridge_regWO[-1].coef_)
            y_predLinridgeWO.append(ridge_regWO[-1].predict(Xwof_test_L))
            axLinridgeWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)],
                                          y_predLinridgeWO[-1]]).T)
            ridge_regW.append(model['RidgeW'].fit(Xwf_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            ridge_coef_W.append(ridge_regW[-1].coef_)
            y_predLinridgeW.append(ridge_regW[-1].predict(Xwf_test_L))
            axLinridgeW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                         y_predLinridgeW[-1]]).T)
            ## LASSO Predictions
            lasso_regWO.append(model['LassoWO'].fit(Xwof_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
            lasso_coef_WO.append(lasso_regWO[-1].coef_)
            y_predLinlassoWO.append(lasso_regWO[-1].predict(Xwof_test_L))
            axLinlassoWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], 
                                          y_predLinlassoWO[-1]]).T)
            lasso_regW.append(model['LassoW'].fit(Xwf_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            lasso_coef_W.append(lasso_regW[-1].coef_)
            y_predLinlassoW.append(lasso_regW[-1].predict(Xwf_test_L))
            axLinlassoW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                         y_predLinlassoW[-1]]).T)
#           ## Performances
#           ## Compare train set performance
#           trainscoreRWO.append(model['RidgeWO'].score(Xwof_train[[respL]], y_train2[resp]))
#           trainscoreLWO.append(model['LassoWO'].score(Xwof_train[[respL]], y_train2[resp]))
#           trainscoreRW.append(model['RidgeW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
#           trainscoreLW.append(model['LassoW'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
#           ## Compare test set performance
#           testscoreRWO.append(model['RidgeWO'].score(Xwof_test[[respL]], y_test2[resp]))
#           testscoreLWO.append(model['LassoWO'].score(Xwof_test[[respL]], y_test2[resp]))
#           testscoreRW.append(model['RidgeW'].score(Xwf_test[[respL] + [exog]], y_test2[resp]))
#           testscoreLW.append(model['LassoW'].score(Xwf_test[[respL] + [exog]], y_test2[resp]))
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
"""
## Plot train results,
plt.bar(np.arange(0,len(ridgeresultsWO),step=1), height=ridgeresultsWO, 
                  align='edge', alpha=0.25, label='Ridge without Features', linewidth=1)
plt.bar(np.arange(0,len(ridgeresultsW),step=1), height=ridgeresultsW, 
                  align='edge', alpha=0.25, label='Ridge with Features', linewidth=1)
plt.bar(np.arange(0,len(lassoresultsWO),step=1), height=lassoresultsWO, 
                  align='edge', alpha=0.25, label=' Lasso without Features', linewidth=1)
plt.bar(np.arange(0,len(lassoresultsW),step=1), height=lassoresultsW, 
                  align='edge', alpha=0.25, label='Lasso with Features', linewidth=1)
plt.xticks(ticks=np.arange(0,len(lassoresultsWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')
"""
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
modelselectWPCA     = []
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
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]    # Step2: Fit standardizer to train set
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
    for resp in list_of_responses:
        for pc in pclist:
            ## Model Selection
            # Only one lag
            modelselectWPCA  = [1]
            # Define lagged X then trim the initial observations (with no input)
            Xwf_train_L    = sm.tsa.tsatools.lagmat2ds(Xwf_trainPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=modelselectWPCA[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_trainPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwf_trainPCA[[resp] + ['ret'] + [pc]].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_testPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=np.max([1,modelselectWPCA[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_testPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwf_testPCA[[resp] + ['ret'] + [pc]].columns[0])
            ## Train model
                        # Train models
            model        = {'LassoW':  GridSearchCV(Lasso(fit_intercept=False, normalize=False, 
                                                          random_state=42, selection='random', 
                                                          max_iter=1000), 
                                          param_grid=lasso_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwf_train_L.index), iid=True).fit(Xwf_train_L, y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_,
                            'RidgeW':  GridSearchCV(Ridge(fit_intercept=False, normalize=False, 
                                                          random_state=42, 
                                                          max_iter=1000), 
                                          param_grid=ridge_params, 
                                          scoring='neg_mean_squared_error', 
                                          return_train_score=True, 
                                          cv=tsplit.split(Xwf_train_L.index), iid=True).fit(Xwf_train_L, 
                                                                                            y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_}            
            ## RIDGE Predictions
            ridge_regWPCA.append(model['RidgeW'].fit(Xwf_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            ridge_coef_WPCA.append(ridge_regWPCA[-1].coef_)
            y_predLinridgeWPCA.append(ridge_regWPCA[-1].predict(Xwf_test_L))
            axLinridgeWPCA.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                           y_predLinridgeWPCA[-1]]).T)
            ## LASSO Predictions
            lasso_regWPCA.append(model['LassoW'].fit(Xwf_train_L, 
                                                    y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            lasso_coef_WPCA.append(lasso_regWPCA[-1].coef_)
            y_predLinlassoWPCA.append(lasso_regWPCA[-1].predict(Xwf_test_L))
            axLinlassoWPCA.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], 
                                           y_predLinlassoWPCA[-1]]).T)
            ## Performances
            ## Compare train set performance
            trainscoreRWPCA.append(model['RidgeW'].score(Xwf_train_L, 
                                                      y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            trainscoreLWPCA.append(model['LassoW'].score(Xwf_train_L, 
                                                      y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            ## Compare test set performance
            testscoreRWPCA.append(model['RidgeW'].score(Xwf_test_L, 
                                                     y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
            testscoreLWPCA.append(model['LassoW'].score(Xwf_test_L, 
                                                     y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ridge_coef_WPCA        = np.array(ridge_coef_WPCA)
lasso_coef_WPCA        = np.array(lasso_coef_WPCA)
ridgeresultsWPCA       = []
ridgeresidWPCA         = []
lassoresultsWPCA       = []
lassoresidWPCA         = []
# Get the results
for i in range(0,len(y_predLinridgeWPCA)):
    ridgeresultsWPCA.append(aic(axLinridgeWPCA[i][:,0],axLinridgeWPCA[i][:,1],ridge_coef_WPCA[i].shape[0]))
    ridgeresidWPCA.append(axLinridgeWPCA[i][:,0] - axLinridgeWPCA[i][:,1])
    lassoresultsWPCA.append(aic(axLinlassoWPCA[i][:,0],axLinlassoWPCA[i][:,1],lasso_coef_WPCA[i].shape[0]))
    lassoresidWPCA.append(axLinlassoWPCA[i][:,0] - axLinlassoWPCA[i][:,1])
# =============================================================================
"""
## Plot train results,
plt.bar(np.arange(0,len(ridgeresultsWPCA),step=1), height=ridgeresultsWPCA, 
                  align='edge', alpha=0.5, label='Ridge with PCs', linewidth=1)
plt.bar(np.arange(0,len(lassoresultsWPCA),step=1), height=lassoresultsWPCA, 
                  align='edge', alpha=0.5, label='Lasso with PCs', linewidth=1)
plt.xticks(ticks=np.arange(0,len(lassoresultsWPCA),step=6))
plt.legend()
plt.grid(b=None,axis='x')
"""
# =============================================================================
### Mean performance of each feature
ridgeresultsWPCA  = pd.DataFrame(np.split(np.array(ridgeresultsWPCA), len(pclist)), columns=list_of_responses*5, index=pclist)
print('RIDGEWPCA: Mean AIC with feature = ', ridgeresultsWPCA.mean(axis=1))
lassoresultsWPCA  = pd.DataFrame(np.split(np.array(lassoresultsWPCA), len(pclist)), columns=list_of_responses*5, index=pclist)
print('LASSOWPCA: Mean AIC with feature = ', lassoresultsWPCA.mean(axis=1))
# =============================================================================
