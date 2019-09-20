# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:45:26 2019

@author: yeeya
"""

#%% RANDOM FOREST FEATURE SELECTION ###########################################

# Defining grid for Gridsearch cross validation ##############################
n_estimators      = [5]

# Server execution:
# [int(x) for x in np.linspace(start = 50, stop = 200, num = 3)]

max_features      = ['auto'] # Number of features to consider at every split

max_depth         = [10, None] # Maximum number of levels in tree

# Server execution:
#[int(x) for x in np.linspace(10, 100, num = 3)]
#max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2,10,20] 
# Minimum number of samples required at each leaf node
min_samples_leaf  = [2,10,20] 
# Method of selecting samples for training each tree
bootstrap         = [True] # Create the random grid

random_grid       = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

# Define Random Forest Regressor as estimator for regression
RFR              = RandomForestRegressor() 
#
## RANDOM FOREST #############################################################
forest_regWO     = []
axRFRWO          = []
ypredforestWO    = []
#
forest_regW      = []
axRFRW           = []
ypredforestW     = []
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
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        # Set GridSearchCV 
        model   = {'RFRWO': GridSearchCV(RFR, param_grid=random_grid, 
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True, cv=tsplit.split(Xwof_train.index),
                                       iid=True, n_jobs=-1).fit(Xwof_train[respL], y_train2[resp]).best_estimator_,
                   'RFRW': GridSearchCV(RFR, param_grid=random_grid, scoring='neg_mean_squared_error',
                                        return_train_score=True, cv=tsplit.split(Xwf_train.index),
                                        iid=True, n_jobs=-1).fit(Xwf_train[[respL] + notff3], y_train2[resp]).best_estimator_}
        ## Random Forest Regression
        forest_regWO.append(model['RFRWO'])
        ypredforestWO.append(forest_regWO[len(forest_regWO)-1].predict(Xwof_test[respL]))
        axRFRWO.append(np.array([y_test2[resp], ypredforestWO[len(forest_regWO)-1]]).T)
        forest_regW.append(model['RFRW'])
        ypredforestW.append(forest_regW[len(forest_regW)-1].predict(Xwf_test[[respL] + notff3]))
        axRFRW.append(np.array([y_test2[resp], ypredforestW[len(forest_regW)-1]]).T)
        ## Random Forest Classification
        #forest_clas.append(model['RFRW'])
        #ypredforestclas.append(forest_clas[len(forest_clas)-1].predict(Xwf_test))
        #axRFC.append(np.array([y_test1[resp], ypredforestclas[len(forest_clas)-1]]).T)
t1 = timer()
print(t1-t0)
#
cols            = int(np.size(axRFRWO)/(trainmax-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforestWO   = np.array(axRFRWO).squeeze().reshape(trainmax-trainmin,cols)
ypredforestW    = np.array(axRFRW).squeeze().reshape(trainmax-trainmin,cols)
forestresultsWO = []
axRFRWO         = []
forestresultsW  = []
axRFRW          = []
for i in range(0,ypredforestWO.shape[1]-1,2):
    forestresultsWO.append(r2_score(ypredforestWO[1:,i],ypredforestWO[:-1,i+1]))
    axRFRWO.append(np.array([ypredforestWO[:,i],ypredforestWO[:,i+1]]).T)
    forestresultsW.append(r2_score(ypredforestW[1:,i],ypredforestW[:-1,i+1]))
    axRFRW.append(np.array([ypredforestW[:,i],ypredforestW[:,i+1]]).T)
#
FI              = []
FII = pd.DataFrame()
for i in range(len(forest_regW)):
    for respL in responsesL:
        FI.append(pd.DataFrame(forest_regW[i].feature_importances_, index = Xwf_train[[respL] + notff3].columns, columns=['Feature Importance']))
        FII[i] = forest_regW[i].feature_importances_
        FII.index = Xwf_train[[respL] + notff3].columns
#
FII.mean(axis = 1)
print('Mean of feature importance = ', FII.mean(axis = 1))
#
"""   
plt.barh(range(forest_regWPCA[48].n_features_), FIPCA[49], align='center')
plt.yticks(np.arange(forest_regW[48].n_features_), notff3)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
"""
#
#%% Random Forest with PCs ###################################################
#
forest_regWPCA      = []
axRFRWPCA           = []
ypredforestWPCA     = []
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
        # Set GridSearchCV 
        model   = {'RFRWPCA': GridSearchCV(RFR, param_grid=random_grid, scoring='neg_mean_squared_error',
                                           return_train_score=True, cv=tsplit.split(Xwf_trainPCA.index),
                                           iid=True, n_jobs=-1).fit(Xwf_trainPCA[[respL] + pclist], y_train2[resp]).best_estimator_}
        ## Random Forest Regression
        forest_regWPCA.append(model['RFRWPCA'])
        ypredforestWPCA.append(forest_regWPCA[len(forest_regWPCA)-1].predict(Xwf_testPCA[[respL] + pclist]))
        axRFRWPCA.append(np.array([y_test2[resp], ypredforestWPCA[len(forest_regWPCA)-1]]).T)
        ## Random Forest Classification
        #forest_clas.append(model['RFRW'])
        #ypredforestclas.append(forest_clas[len(forest_clas)-1].predict(Xwf_test))
        #axRFC.append(np.array([y_test1[resp], ypredforestclas[len(forest_clas)-1]]).T)
t1 = timer()
print(t1-t0)
#
cols                = int(np.size(axRFRWPCA)/(trainmax-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
ypredforestWPCA     = np.array(axRFRWPCA).squeeze().reshape(trainmax-trainmin,cols)
forestresultsWPCA   = []
axRFRWPCA           = []
for i in range(0,ypredforestWPCA.shape[1]-1,2):
    forestresultsWPCA.append(r2_score(ypredforestWPCA[1:,i],ypredforestWPCA[:-1,i+1]))
    axRFRWPCA.append(np.array([ypredforestWPCA[1:,i],ypredforestWPCA[:-1,i+1]]).T)
#
FIPCA           = []
idxx            = []
idxl            = []
meanFIPCA       = []
FIPCAdf         = pd.DataFrame()
for i in range(len(forest_regWPCA)):
    for respL in responsesL:
        FIIPCA = np.array(forest_regWPCA[i].feature_importances_)
        idxx.append(np.arange(len(FIIPCA)-2))
        idxl.append([[respL] + [respL] + ['PC' + str(idxx[-1][x]+1) for x in range(len(idxx[-1]))]])
        FIPCA.append(pd.DataFrame(FIIPCA,index=idxl[-1], columns=['Feature Importance']))
#
print('Mean of feature importance (with PCs) = ', FIIPCA.mean(axis = 1))
###############################################################################