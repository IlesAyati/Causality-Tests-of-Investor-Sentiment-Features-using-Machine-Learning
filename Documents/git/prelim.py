# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:44:45 2019

@author: yeeya
"""
#%%
tsplit          = TimeSeriesSplit(n_splits=5, max_train_size=270)
scaler          = StandardScaler()
scaler2         = StandardScaler()
lin_regWO       = []
lin_regW        = []
lin_regcoefWO   = []
lin_regcoefW    = []
y_predLinWO     = []
y_predLinW      = []
axLinWO         = []
axLinW          = []
trainscoreWO    = []
trainscoreW     = []
testscoreWO     = []
testscoreW      = []
modelselectWO   = []
modelselectW    = []
CorrXwof_train_L= []
CorrXwf_train_L = []
CorrXwof_test_L = []
CorrXwf_test_L  = []
t0 = timer()
for train_index,test_index in tsplit.split(regdata.index):
    #print(range(i-trainmin,i))
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
##############################################################################
    for resp in list_of_responses:
        for exog in notff3:
            ## Model Selection
            modelselectWO.append(VAR(Xwof_train[[resp] + ['ret']],
                                   dates=y_train2.index).select_order(maxlags=12))
            modelselectW.append(VAR(Xwf_train[[resp] + ['ret'] + [exog]],
                                   dates=y_train2.index).select_order(maxlags=12))
            # Define lagged X w.r.t AIC
            Xwof_train_L    = sm.tsa.tsatools.lagmat2ds(Xwof_train[[resp] + ['ret']],
                                                        maxlag0=np.max([1,modelselectWO[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwof_train[[resp] + ['ret']].index[:modelselectWO[-1].aic],
                                                                                        columns=Xwof_train[[resp] + ['ret']].columns[0])
            Xwf_train_L     = sm.tsa.tsatools.lagmat2ds(Xwf_train[[resp] + ['ret'] + [exog]],
                                                        maxlag0=np.max([1,modelselectW[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_train[[resp] + ['ret'] + [exog]].index[:modelselectW[-1].aic],
                                                                                        columns=Xwf_train[[resp] + ['ret'] + [exog]].columns[0])
            Xwof_test_L    = sm.tsa.tsatools.lagmat2ds(Xwof_test[[resp] + ['ret']],
                                                        maxlag0=np.max([1,modelselectWO[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret']].index[:modelselectWO[-1].aic],
                                                                                        columns=Xwf_test[[resp] + ['ret']].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_test[[resp] + ['ret'] + [exog]],
                                                        maxlag0=np.max([1,modelselectW[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_test[[resp] + ['ret'] + [exog]].index[:modelselectW[-1].aic],
                                                                                        columns=Xwf_test[[resp] + ['ret'] + [exog]].columns[0])
            # Correlation matrices of lagged X
            CorrXwof_train_L.append(Xwof_train_L.corr())
            CorrXwf_train_L.append(Xwf_train_L.corr())
            CorrXwof_test_L.append(Xwof_test_L.corr())
            CorrXwf_test_L.append(Xwf_test_L.corr())
            ## Fit regressions
            model      = {'OLSwof': LinearRegression(fit_intercept=False, normalize=False).fit(Xwof_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwof_train_L.index)]),
                          'OLSwf': LinearRegression(fit_intercept=False, normalize=False).fit(Xwf_train_L,y_train2[resp].iloc[y_train2.index.isin(Xwf_train_L.index)])}
            ## Predictions
            lin_regWO.append(model['OLSwof'])
            lin_regW.append(model['OLSwf'])
            lin_regcoefWO.append(lin_regWO[len(lin_regWO)-1].coef_)
            lin_regcoefW.append(lin_regW[len(lin_regW)-1].coef_)
            y_predLinWO.append(lin_regWO[len(lin_regWO)-1].predict(Xwof_test_L))
            y_predLinW.append(lin_regW[len(lin_regW)-1].predict(Xwf_test_L))
            axLinWO.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwof_test_L.index)], y_predLinWO[len(lin_regWO)-1]]).T)
            axLinW.append(np.array([y_test2[resp].iloc[y_test2.index.isin(Xwf_test_L.index)], y_predLinW[len(lin_regW)-1]]).T)
t1 = timer()
print(t1-t0)


#################### COPY PASTE
            ## Model Selection
            modelselectWPCA.append(VAR(Xwf_trainPCA[[resp] + ['ret'] + [pc]],
                                       dates=y_train2.index).select_order(maxlags=12))
            # Define lagged X w.r.t AIC
            Xwf_train_L    = sm.tsa.tsatools.lagmat2ds(Xwf_trainPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=np.max([1,modelselectWPCA[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_trainPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1].aic],
                                                                                        columns=Xwf_trainPCA[[resp] + ['ret'] + [pc]].columns[0])
            Xwf_test_L     = sm.tsa.tsatools.lagmat2ds(Xwf_testPCA[[resp] + ['ret'] + [pc]],
                                                        maxlag0=np.max([1,modelselectWPCA[-1].aic]),trim='forward', 
                                                        dropex=1, use_pandas=True).drop(index=Xwf_testPCA[[resp] + ['ret'] + [pc]].index[:modelselectWPCA[-1].aic],
                                                                                        columns=Xwf_testPCA[[resp] + ['ret'] + [pc]].columns[0])