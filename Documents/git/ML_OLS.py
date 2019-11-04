# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:38:40 2019

@author: yeeya
"""
# %% OLS regressions    ######################################################
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
ftestWO         = []
ftestW          = []
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
                                   dates=y_train2.index).select_order(maxlags=None).aic]))
            modelselectW.append(np.max([1,VAR(Xwf_train[[resp] + ['ret'] + [exog]],
                                   dates=y_train2.index).select_order(maxlags=None).aic]))
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
            ## Train models: Not really relevant for LinearRegression (no hyperparameters)
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False)}
            ## Predictions
            lin_regWO.append(model['OLSwof'].fit(Xwof_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
            lin_regW.append(model['OLSwf'].fit(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            lin_regcoefWO.append(lin_regWO[-1].coef_)
            lin_regcoefW.append(lin_regW[-1].coef_)
            y_predLinWO.append(lin_regWO[-1].predict(Xwof_test_L))
            y_predLinW.append(lin_regW[-1].predict(Xwf_test_L))
            ftestWO.append([Xwof_test_L.columns.values,f_regression(Xwof_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)])[0]])
            ftestW.append([Xwf_test_L.columns.values,f_regression(Xwf_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)])[0]])
            axLinWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], y_predLinWO[-1]]).T)
            axLinW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], y_predLinW[-1]]).T)
            ## Heteroscedasticity and autocorrelation tests
            Hetlin_regWO.append(sm.stats.diagnostic.het_white(np.array([axLinWO[-1][:,0] - axLinWO[-1][:,1]]).T, 
                                                              sm.add_constant(Xwof_test_L)))
            Hetlin_regW.append(sm.stats.diagnostic.het_white(np.array([axLinW[-1][:,0] - axLinW[-1][:,1]]).T, 
                                                             sm.add_constant(Xwf_test_L)))
            acorrlin_regWO.append(sm.stats.diagnostic.acorr_ljungbox(axLinWO[-1][:,0] - axLinWO[-1][:,1]))
            acorrlin_regW.append(sm.stats.diagnostic.acorr_ljungbox(axLinW[-1][:,0] - axLinW[-1][:,1]))
            ## Performances - R squared - Inaccurate due to violated assumptions
            ## Compare train set performance
            trainscoreWO.append(model['OLSwof'].score(Xwof_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
            trainscoreW.append(model['OLSwf'].score(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            ## Compare test set performance
            testscoreWO.append(model['OLSwof'].score(Xwof_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)]))
            testscoreW.append(model['OLSwf'].score(Xwf_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
# =============================================================================
lin_regcoefWO= np.array(lin_regcoefWO)
lin_regcoefW = np.array(lin_regcoefW)
linresultsWO = []
linresultsW  = []
linresidWO   = []
linresidW    = []
Ftest        = []
# Get the results
for i in range(0,len(axLinWO)):
    linresultsWO.append(aic(axLinWO[i][:,0],axLinWO[i][:,1],lin_regcoefWO[i].shape[0]))
    linresultsW.append(aic(axLinW[i][:,0],axLinW[i][:,1],lin_regcoefW[i].shape[0]))
    Ftest.append(F(axLinWO[i][:,0],
                   axLinWO[i][:,1],
                   axLinW[i][:,0],
                   axLinW[i][:,1],
                   lin_regcoefWO[i].shape[0], 
                   lin_regcoefW[i].shape[0]))
    linresidWO.append(axLinWO[i][:,0] - axLinWO[i][:,1]) 
    linresidW.append(axLinW[i][:,0] - axLinW[i][:,1])  
### Structuring and dataframing results
linresultsWO        = pd.DataFrame(np.split(np.array(linresultsWO), 5), 
                                   columns=np.array([list_of_responses[i] + str('_L') for i in range(len(list_of_responses))]).repeat(6),
                                   index=['Split' + str(i+1) for i in range(5)]).T                      
print('Mean AIC without feature: ', linresultsWO.mean(axis=1))

linresultsW  = pd.DataFrame(np.split(np.array(linresultsW), 5), 
                            columns=[notff3[i] + str('_L') for i in range(len(notff3))]*6, 
                            index=['Split' + str(i+1) for i in range(5)]).T
print('Mean AIC with feature:', linresultsW.mean(axis=1))

### F-tests in concetenated dataframe
ftestWO      = pd.concat([pd.DataFrame(zip(ftestWO[i][0],ftestWO[i][1])) for i in range(len(ftestWO))],axis=1)
ftestW       = pd.concat([pd.DataFrame(zip(ftestW[i][0],ftestW[i][1])) for i in range(len(ftestW))],axis=1)
# Residuals
linresidWO = pd.DataFrame(linresidWO)
linresidW  = pd.DataFrame(linresidW)
# =============================================================================
# %% OLS: PRINCIPAL COMPONENTS ###############################################
#
# First, split the data, then scale, then apply SVD.
# Second, test significance of extracted pcs (LinearRegression)
#
pclist            = []
lin_regWPCA       = []
lin_regcoefWPCA   = []
lin_regWPCA2      = []
lin_regcoefWPCA2  = []
#
y_predLinWPCA     = []
axLinWPCA         = []
y_predLinWPCA2    = []
axLinWPCA2        = []
ftestWPCA         = []
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
    Xwf_train, Xwf_test   = regdata[notff3].iloc[train_index], regdata[notff3].iloc[test_index]
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
######## Apply PCA transformation ############################################
    # Fit PCA to train set (with features)
    pca.fit(Xwf_train) 
    # Transform train and test sets 
    Xwf_trainPCA    = pca.transform(Xwf_train)
    Xwf_testPCA     = pca.transform(Xwf_test)  # Transformed test sets
    pclist.append(['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])])
    # Dataframe train set and enumerate PCs
    Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train.index, columns=pclist[-1])
    # Dataframe test set and enumerate PCs
    Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test.index, columns=Xwf_trainPCA.columns)
    ####regdataPC.append(X_pca2.values) Ignore
    # Concatenate the response lags with all PCs
    Xwf_trainPCA = pd.concat([Xwof_train,Xwf_trainPCA], axis=1)
    Xwf_testPCA  = pd.concat([Xwof_test,Xwf_testPCA], axis=1)
##############################################################################
    for resp in list_of_responses:
        for pc in pclist[-1]:
            ## Model Selection
            modelselectWPCA  = [1] # Only one lag for PCs
            # Define lagged X then trim the initial observations (with no input)
            Xwf_train_L    = sm.tsa.tsatools.lagmat2ds(Xwf_trainPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=modelselectWPCA[-1],trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_trainPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwf_trainPCA[[resp] + ['ret'] + [pc]].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_testPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=np.max([1,modelselectWPCA[-1]]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_testPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1]],
                                                                                        columns=Xwf_testPCA[[resp] + ['ret'] + [pc]].columns[0])
            # Correlation matrices of lagged X
            CorrXwf_train_L.append(Xwf_train_L.corr())
            CorrXwf_test_L.append(Xwf_test_L.corr())
            ## Train model: Not really relevant for LinearRegression (no hyperparameters)
            model      = {'OLSwf': LinearRegression(fit_intercept=False, normalize=False)}
            ## Predictions
            lin_regWPCA.append(model['OLSwf'].fit(Xwf_train_L, \
                                                  y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            lin_regcoefWPCA.append(lin_regWPCA[-1].coef_)
            y_predLinWPCA.append(lin_regWPCA[-1].predict(Xwf_test_L))
            ftestWPCA.append([Xwf_test_L.columns.values,f_regression(Xwf_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)])[0]])
            axLinWPCA.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], \
                                       y_predLinWPCA[-1]]).T)
            ## Heteroscedasticity and autocorrelation tests
            Hetlin_regWPCA.append(sm.stats.diagnostic.het_white(np.array([axLinWPCA[-1][:,0] - axLinWPCA[-1][:,1]]).T, \
                                                                sm.add_constant(Xwf_test_L)))
            acorrlin_regWPCA.append(sm.stats.diagnostic.acorr_ljungbox(axLinWPCA[-1][:,0] - axLinWPCA[-1][:,1]))
            ## Performances - R squared
            ## Compare train set performance
            trainscoreWPCA.append(model['OLSwf'].score(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
            ## Compare test set performance
            testscoreWPCA.append(model['OLSwf'].score(Xwf_test_L,y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: AIC
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
lin_regcoefWPCA = np.array(lin_regcoefWPCA)
linresultsWPCA  = []
linresidWPCA    = []
# Get the results
for i in range(0,len(axLinWPCA)):
    linresultsWPCA.append(aic(axLinWPCA[i][:,0],axLinWPCA[i][:,1],lin_regcoefWPCA[i].shape[0]))
    linresidWPCA.append(axLinWPCA[i][:,0] - axLinWPCA[i][:,1])  
#  
## Mean performance of each feature
linresultsWPCA  = pd.DataFrame(np.split(np.array(linresultsWPCA), 5), 
                               columns=[np.array(notff3).repeat(5)[i] + '_PC' + str(i+1) for i in range(30)], 
                               index=['Split' + str(i+1) for i in range(5)]).T
print('Mean AIC with PC = ',
      linresultsWPCA.mean(axis=1))
#
### F-tests in concetenated dataframe
ftestWPCA    = pd.concat([pd.DataFrame(zip(ftestWPCA[i][0],ftestWPCA[i][1])) for i in range(len(ftestWPCA))],axis=1)
# Residuals
linresidWPCA = pd.DataFrame(linresidWPCA)
# =============================================================================