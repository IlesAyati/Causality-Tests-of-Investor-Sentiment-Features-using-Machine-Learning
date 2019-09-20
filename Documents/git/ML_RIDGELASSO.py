# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:44:22 2019

@author: yeeya
"""

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