# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:38:40 2019

@author: yeeya
"""

# %% OLS regressions    ######################################################
scaler          = StandardScaler()
scaler2         = StandardScaler()
lin_regWO       = []
lin_regW        = []
y_predLinWO     = []
y_predLinW      = []
axLinWO         = []
axLinW          = []
trainscoreWO    = []
trainscoreW     = []
testscoreWO     = []
testscoreW      = []
t0 = timer()
for i in range(trainmin,trainmax):
    #print(range(i-trainmin,i))
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
    # Standardize train AND test sets WITH FEATURES and their lags
    Xwf_train     = pd.DataFrame(scalefitwf.transform(Xwf_train), columns=Xwf_train.columns,index=Xwf_train.index)
    Xwf_test      = pd.DataFrame(scalefitwf.transform(Xwf_test), columns=Xwf_test.columns,index=Xwf_test.index)
    # Scale and fit responses 
    scalefity   = scaler.fit(y_train2) 
    y_train2    = pd.DataFrame(scalefity.transform(y_train2), columns=list_of_responses,index=y_train2.index)
    y_test2     = pd.DataFrame(scalefity.transform(y_test2), columns=list_of_responses,index=y_test2.index)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for exog in notff3:
            ## Fit regressions
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False).fit(Xwof_train[respL],y_train2[resp]),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_train[[respL] + [exog]],y_train2[resp]),}
            ## Predictions
            lin_regWO.append(model['OLSwof'])
            lin_regW.append(model['OLSwf'])
            y_predLinWO.append(lin_regWO[len(lin_regWO)-1].predict(Xwof_test[respL]))
            y_predLinW.append(lin_regW[len(lin_regW)-1].predict(Xwf_test[[respL] + [exog]]))
            axLinWO.append(np.array([y_test2[resp], y_predLinWO[len(lin_regWO)-1]]).T)
            axLinW.append(np.array([y_test2[resp], y_predLinW[len(lin_regW)-1]]).T)    ## Performances
            ## Compare train set performance
            trainscoreWO.append(model['OLSwof'].score(Xwof_train[respL], y_train2[resp]))
            trainscoreW.append(model['OLSwf'].score(Xwf_train[[respL] + [exog]], y_train2[resp]))
t1 = timer()
print(t1-t0)

# Compare test set performance
cols = int(len(axLinW)*2/(trainmax-trainmin)) # Twice the number of predictions made
# We want to restructure (y, y_prediction) pairs so that all of them are in one matrix. 
# Each columns alternates between true y_test and predicted y_test
y_predLinW0  = np.array(axLinWO).squeeze().reshape(trainmax-trainmin,cols)
y_predLinW   = np.array(axLinW).squeeze().reshape(trainmax-trainmin,cols)
linresultsWO = []
axLinWO      = []
linresid     = []
linresultsW  = []
axLinW       = []
# Get the results
for i in range(0,y_predLinW0.shape[1]-1,2):
    linresultsWO.append(r2_score(y_predLinW0[1:,i],y_predLinW0[:-1,i+1]))
    axLinWO.append(np.array([y_predLinW0[:,i],y_predLinW0[:,i+1]]).T)
    #linresid.append(axLin[len(axLin)-1].T[0]-axLin[len(axLin)-1].T[1]) # Heteroscedasticity? 
    linresultsW.append(r2_score(y_predLinW[1:,i],y_predLinW[:-1,i+1]))
    axLinW.append(np.array([y_predLinW[:,i],y_predLinW[:,i+1]]).T)

### Mean performance of each feature
linresultsWO = pd.DataFrame(np.split(np.array(linresultsWO), 6), columns=list_of_responses, index=responsesL)
print('Mean R2 score without feature = ', linresultsWO.mean(axis=1))
linresultsW  = pd.DataFrame(np.split(np.array(linresultsW), 6), columns=notff3, index=list_of_responses).T
print('Mean of R2 score with feature = ', linresultsW.mean(axis=1))

"""
## Plot train results, LinearRegression
plt.bar(np.arange(0,len(trainscoreWO),step=1), height=trainscoreWO, 
                  align='edge', alpha=0.5, label='Without Features')
plt.bar(np.arange(0,len(trainscoreW),step=1), height=trainscoreW, 
                  align='edge', alpha=0.5, label='With Features')
plt.xticks(ticks=np.arange(0,len(trainscoreWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')
"""

##############################################################################
# %% OLS: PRINCIPAL COMPONENTS ###############################################

# First, split the data, then scale, then apply SVD.
# Second, test significance of extracted pcs (LinearRegression)
#
pca                 = PCA(n_components='mle', whiten=1, random_state=42)
lin_regWPCA         = []
y_predLinWPCA       = []
axLinWPCA           = []
trainscoreWPCA      = []
testscoreWPCA       = []
#
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
    pclist          = ['PC'+ str(i+1) for i in range(Xwf_trainPCA.shape[1])]
    # Dataframe train set and enumerate PCs
    Xwf_trainPCA    = pd.DataFrame(Xwf_trainPCA, index=Xwf_train.index, columns=pclist)
    # Dataframe test set and enumerate PCs
    Xwf_testPCA     = pd.DataFrame(Xwf_testPCA, index=Xwf_test.index, columns=Xwf_trainPCA.columns)
    ####regdataPC.append(X_pca2.values) Ignore
    # Concatenate the response lags with all PCs
    Xwf_trainPCA = pd.concat([Xwof_train,Xwf_trainPCA], axis=1)
    Xwf_testPCA  = pd.concat([Xwof_test,Xwf_testPCA], axis=1)
##############################################################################
    for resp, respL in zip(list_of_responses, responsesL):
        for pc in pclist:
            #print(resp, respL, pc)
            ## Fit regression
            model      = {'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]),}
            ## Predictions
            lin_regWPCA.append(model['OLSwf'])
            y_predLinWPCA.append(lin_regWPCA[len(lin_regWPCA)-1].predict(Xwf_testPCA[[respL] + [pc]]))
            axLinWPCA.append(np.array([y_test2[resp], y_predLinWPCA[len(lin_regWPCA)-1]]).T)
            ## Performance
            ## Compare train set performance
            trainscoreWPCA.append(model['OLSwf'].score(Xwf_trainPCA[[respL] + [pc]], y_train2[resp]))
t1 = timer()
print(t1-t0)
#
cols = int(len(axLinWPCA)*2/(trainmax+1-trainmin))
# Restructure predictions so that all of them are in one matrix. 
# Each columns alternates between true y_test and predited y_test
y_predLinWPCA   = np.array(axLinWPCA).squeeze().reshape(trainmax+1-trainmin,cols)
linresultsWPCA  = []
axLinWPCA       = []
# Get the results
for i in range(0,y_predLinWPCA.shape[1]-1,2):
    linresultsWPCA.append(r2_score(y_predLinWPCA[1:,i],y_predLinWPCA[:-1,i+1]))
    axLinWPCA.append(np.array([y_predLinWPCA[:,i],y_predLinWPCA[:,i+1]]).T)
#
### Mean performance of each feature
linresultsWPCA  = pd.DataFrame(np.split(np.array(linresultsWPCA), 6), columns=pclist, index=list_of_responses).T
print('Mean of R2 score with feature = ', linresultsWPCA.mean(axis=1))
#
"""
## Plot train results, LinearRegression
plt.bar(np.arange(0,len(trainscoreWPCA),step=1), height=trainscoreWPCA, 
                  align='edge', alpha=0.5, label='With PCs')
plt.xticks(ticks=np.arange(0,len(trainscoreWPCA),step=6))
plt.legend()
plt.grid(b=None,axis='x')

"""
############################################################################