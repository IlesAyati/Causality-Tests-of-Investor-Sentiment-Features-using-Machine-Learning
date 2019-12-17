#%% RANDOM FOREST FEATURE SELECTION ###########################################

## Defining grid for Gridsearch cross validation ##
n_estimators      = [100]
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
min_samples_split = [5,10,20,50] 
# Minimum number of samples required at each leaf node
min_samples_leaf  = [5,10,20,50] 
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
RFR              = RandomForestRegressor(oob_score=True, warm_start= True, 
                                         criterion='mse') 

## RANDOM FOREST ####
#
forest_regWO        = []
axRFRWO             = []
ypredforestWO       = []
#
forest_regW         = []
axRFRW              = []
ypredforestW        = []
#
trainscore_forestWO = []
trainscore_forestW  = []
testscore_forestWO  = []
testscore_forestW   = []
#
countiter           = []
modelselectWORFR    = []
modelselectWRFR     = []
#
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(train_index,test_index)
######## Data Splitting and Scaling ##########################################
    # Step1: Split time series into train and test sets
    Xwof_train, Xwof_test = regdata[list_of_responses + ['ret']].iloc[train_index], regdata[list_of_responses + ['ret']].iloc[test_index]
    #                              !-------------------! Here we include everything but y_t
    Xwf_train, Xwf_test   = regdata[list_of_responses + ['ret'] + notff3].iloc[train_index], regdata[list_of_responses + ['ret'] + notff3].iloc[test_index]
    #
    y_train2, y_test2     = regdata[list_of_responses].iloc[train_index], regdata[list_of_responses].iloc[test_index]
# =============================================================================
#     # Step2: Fit standardizer to train sets
#     scalefitwof    = scaler.fit(Xwof_train) # Standardize to fit train set WITHOUT FEATURE LAGS
#     scalefitwf     = scaler2.fit(Xwf_train)  # Standardize to fit train set WITH FEATURE LAGS
#     # Step3: Standardize train AND test sets WITHOUT FEATURES nor their lags
#     Xwof_train     = pd.DataFrame(scalefitwof.transform(Xwof_train), columns=Xwof_train.columns,index=Xwof_train.index)
#     Xwof_test      = pd.DataFrame(scalefitwof.transform(Xwof_test), columns=Xwof_test.columns,index=Xwof_test.index)
#     # Standardize train AND test sets WITHOUT FEATURES and their lags
#     Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
#     Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
#     # Scale and fit responses
#     scalefity   = scaler.fit(y_train2) 
#     y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
#     y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
# =============================================================================
##############################################################################
    for resp in list_of_responses:
        ## Model Selection
        # Only 1 lag
        modelselectWORFR= [1]
        modelselectWRFR = [1]
        # Define lagged X w.r.t AIC
        Xwof_train_L    = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],\
                                                    maxlag0=modelselectWORFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:modelselectWORFR[-1]], \
                                                                                    columns=Xwof_train[[resp] + ['ret']].columns[0])
        Xwf_train_L     = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + notff3], \
                                                    maxlag0=modelselectWRFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + notff3].index[:modelselectWRFR[-1]], \
                                                                                    columns=Xwf_train[[resp] + ['ret'] + notff3].columns[0])
        Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']], \
                                                    maxlag0=modelselectWORFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwof_test[[resp] + ['ret']].index[:modelselectWORFR[-1]], \
                                                                                    columns=Xwof_test[[resp] + ['ret']].columns[0])
        Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + notff3],\
                                                    maxlag0=modelselectWRFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + notff3].index[:modelselectWRFR[-1]], \
                                                                                    columns=Xwf_test[[resp] + ['ret'] + notff3].columns[0])
        # Train models
        model   = {'RFRWO': GridSearchCV(RFR, param_grid=random_grid, \
                                       scoring='neg_mean_squared_error', \
                                       cv=tsplit.split(Xwof_train_L.index),\
                                       iid=True, n_jobs=-1).fit(Xwof_train_L, \
                                                                y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]).best_estimator_,
                   'RFRW': GridSearchCV(RFR, param_grid=random_grid, \
                                        scoring='neg_mean_squared_error',\
                                        cv=tsplit.split(Xwf_train_L.index),\
                                        iid=True, n_jobs=-1).fit(Xwf_train_L, 
                                                                 y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_}
        ## Random Forest Regression
        forest_regWO.append(model['RFRWO'].fit(Xwof_train_L, \
                            y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
        ypredforestWO.append(forest_regWO[-1].predict(Xwof_test_L))
        axRFRWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], \
                                 ypredforestWO[-1]]).T)
        forest_regW.append(model['RFRW'].fit(Xwf_train_L, \
                            y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ypredforestW.append(forest_regW[-1].predict(Xwf_test_L))
        axRFRW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], \
                                ypredforestW[-1]]).T)
        ## Performances
        ## Compare train set performance
        trainscore_forestWO.append(model['RFRWO'].score(Xwof_train_L, \
                                                        y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]))
        trainscore_forestW.append(model['RFRW'].score(Xwf_train_L, \
                                                      y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ## Compare test set performance
        testscore_forestWO.append(model['RFRWO'].score(Xwof_test_L, \
                                                       y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)]))
        testscore_forestW.append(model['RFRW'].score(Xwf_test_L, \
                                                     y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: Mean Squared Error
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
RFRresultsWO = []
RFRresidWO   = []
RFRresultsW  = []
RFRresidW    = []
# Get the results
for i in range(0,len(axRFRWO)):
    RFRresultsWO.append(mean_squared_error(axRFRWO[i][:,0],axRFRWO[i][:,1]))
    RFRresidWO.append(axRFRWO[i][:,0] - axRFRWO[i][:,1])  
    RFRresultsW.append(mean_squared_error(axRFRW[i][:,0],axRFRW[i][:,1]))
    RFRresidW.append(axRFRW[i][:,0] - axRFRW[i][:,1]) 
# =============================================================================
### Mean performance of each feature
RFRresultsWO = pd.DataFrame(np.split(np.array(RFRresultsWO), 5),  
                                   columns=np.array([list_of_responses[i] + str('_L') for i in range(len(list_of_responses))]),
                                   index=['Split' + str(i+1) for i in range(5)]).T
print('Mean MSE without features = ', RFRresultsWO.mean(axis=1))
RFRresultsW  = pd.DataFrame(np.split(np.array(RFRresultsW), 5), 
                            columns=np.array([list_of_responses[i] + str('_L_features') for i in range(len(list_of_responses))]), 
                            index=['Split' + str(i+1) for i in range(5)]).T
print('Mean MSE with features = ', RFRresultsW.mean(axis=1))
# =============================================================================
FIWO = pd.DataFrame()
FIW = pd.DataFrame()
for i in range(len(forest_regWO)):
    FIWO[i] = forest_regWO[i].feature_importances_
    FIWO.index = Xwof_train_L.columns
    FIW[i] = forest_regW[i].feature_importances_
    FIW.index = Xwf_train_L.columns
FIWO.columns = list_of_responses*5
FIW.columns = list_of_responses*5
print('Mean of feature importance without features = ', FIWO.mean(axis = 1))
print('Mean of feature importance with features= ', FIW.mean(axis = 1))
# For plots, the mean feature importance of the last split is used:
FIWOsplit5        = FIWO.iloc[:,-6:]
FIWOsplit5.index  = np.insert(FIWOsplit5.index,1,'response.L.1')[1:]
ranked1           = pd.DataFrame([np.argsort(FIWOsplit5[i]) for i in FIWOsplit5.columns])
FIWsplit5         = FIW.iloc[:,-6:]
FIWsplit5.index   = np.insert(FIWsplit5.index,1,'response.L.1')[1:]
ranked2           = pd.DataFrame([np.argsort(FIWsplit5[i]) for i in FIWsplit5.columns])
# For plot:
rankedWO          = np.argsort(FIWOsplit5.mean(axis=1))
rankedW           = np.argsort(FIWsplit5.mean(axis=1))
# =============================================================================
#%% Random Forest with PCs ###################################################
#
forest_regWPCA          = []
axRFRWPCA               = []
ypredforestWPCA         = []
#
trainscore_forestWPCA   = []
testscore_forestWPCA    = []
#
modelselectWRFR         = []
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
# =============================================================================
#     scalefitwf     = scaler2.fit(Xwf_train)
#     # Step3: Standardize train AND test sets WITHOUT FEATURES and their lags
#     Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), 
#                                  columns=Xwf_train.columns,index=Xwf_train.index)
#     Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), 
#                                  columns=Xwf_test.columns,index=Xwf_test.index)
#     # Scale and fit responses
#     scalefity   = scaler.fit(y_train2) 
#     y_train2    = pd.DataFrame(scalefity.transform(y_train2), 
#                                columns=list_of_responses,index=y_train2.index)
#     y_test2     = pd.DataFrame(scalefity.transform(y_test2), 
#                                columns=list_of_responses,index=y_test2.index)
# =============================================================================
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
        ## Model Selection
        # Pick the same amount of lags as in OLS:
        # Counts iteration number
        countiter.append(y_test2.columns) 
        # Add number of lags corresponding to iteration number from OLS with features
        modelselectWRFR = [1]
        # Define lagged X w.r.t AIC
        Xwf_train_L     = sm.tsa.tsatools.lagmat2ds(Xwf_trainPCA[[resp] + ['ret'] + pclist],\
                                                    maxlag0=modelselectWRFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwf_trainPCA[[resp] + ['ret'] + pclist].index[:modelselectWRFR[-1]],
                                                                                    columns=Xwf_trainPCA[[resp] + ['ret'] + pclist].columns[0])
        Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_testPCA[[resp] + ['ret'] + pclist],\
                                                    maxlag0=modelselectWRFR[-1],trim='forward', \
                                                    dropex=1, use_pandas=True).drop(index=Xwf_testPCA[[resp] + ['ret'] + pclist].index[:modelselectWRFR[-1]],
                                                                                    columns=Xwf_testPCA[[resp] + ['ret'] + pclist].columns[0])
        # Train models
        model   = { \
                   'RFRWPCA': GridSearchCV(RFR, param_grid=random_grid, \
                                        scoring='neg_mean_squared_error', \
                                        cv=tsplit.split(Xwf_train_L.index), \
                                        iid=True, n_jobs=-1).fit(Xwf_train_L, \
                                                                 y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]).best_estimator_}
        ## Random Forest Regression
        forest_regWPCA.append(model['RFRWPCA'].fit(Xwf_train_L, \
                            y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ypredforestWPCA.append(forest_regWPCA[-1].predict(Xwf_test_L))
        axRFRWPCA.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], \
                                ypredforestWPCA[-1]]).T)
        ## Performances
        ## Compare train set performance
        trainscore_forestWPCA.append(model['RFRWPCA'].score(Xwf_train_L, \
                                                      y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)]))
        ## Compare test set performance
        testscore_forestWPCA.append(model['RFRWPCA'].score(Xwf_test_L, \
                                                     y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)]))
t1 = timer()
print(t1-t0)
# =============================================================================
## Alternative scoring on test set: Mean Squared Error
# We restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
RFRresultsWPCA  = []
RFRresidWPCA    = []
# Get the results
for i in range(0,len(axRFRWPCA)):
    RFRresultsWPCA.append(mean_squared_error(axRFRWPCA[i][:,0],axRFRWPCA[i][:,1]))
    RFRresidWPCA.append(axRFRWPCA[i][:,0] - axRFRWPCA[i][:,1]) 
# =============================================================================
### Mean performance of each feature
RFRresultsWPCA  = pd.DataFrame(np.split(np.array(RFRresultsWPCA), 5),
                               columns=np.array([list_of_responses[i] + str('_L+PCs') for i in range(len(list_of_responses))])).T
print('Mean MSE with PC lags = ', RFRresultsWPCA.mean(axis=1))
# =============================================================================
FIWPCA = pd.DataFrame()
for i in range(len(forest_regWPCA)):
    FIWPCA[i]    = forest_regWPCA[i].feature_importances_
    FIWPCA.index = Xwf_train_L.columns
FIWPCA.columns      = list_of_responses*5
print('Mean of relative importance with PCs= ', FIWPCA.mean(axis = 1))
# For plots, the mean feature importance of the last split is used:
FIWPCAsplit5        = np.mean(FIWPCA.iloc[:,-6:],axis=1)
FIWPCAsplit5.index  = np.insert(FIWPCAsplit5.index,1,'response.L.1')[1:]
ranked3             = np.argsort(FIWPCAsplit5)
FIWPCAsplit5        = FIWPCA.iloc[:,-6:]
FIWPCAsplit5.index  = np.insert(FIWPCAsplit5.index,1,'response.L.1')[1:]
# =============================================================================
