# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:09:13 2019

@author: yeeya
"""
#from sklearn.tree.export import export_graphviz
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# %% ## Printing section of data
## Plot of time series
sixcolors       = ['darkcyan', 'teal', 'seagreen' ,
                   'mediumseagreen' , 'lightseagreen' , 'mediumaquamarine' ]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(len(notff3),sharex=True)
fig.set_figheight(6)
fig.set_figwidth(8)
fig.set_label("")
#fig.suptitle('Timeseries - All features', fontsize=12)   
for exog, i, color in zip(notff3,range(len(notff3)), sixcolors):
    dfall[exog].plot(ax=axs[i], color=[color], legend=exog)
    axs[i].legend(loc='lower left')
    axs[i].set(xlabel="")
#fig.savefig('C:/Users/yeeya/Figures/FeaturesSeries.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()
##############################################################################
# %% GLS results ##############################################################
### Univariate
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals1copy[notff3].plot.barh(width=0.8, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stat Overview for Univariate regressions}', fontsize=11)
plt.xlabel(r'\textbf{T-statistic}', fontsize=10)
#plt.ylabel(r'\textbf{Regression}', fontsize=10)
plt.yticks(ticks=np.arange(0, 35, step=6), labels=list_of_responses)
reg1plot = plt.gcf()
reg1plot.set_figwidth(8)
reg1plot.set_figheight(5)
#reg1plot.savefig('C:/Users/yeeya/Figures/reg1plot.pdf')
plt.show()
##############################################################################
### Multivariate - COT
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals2copy[['NONPNL','CPNL','NCPNL', 'OI']].plot.barh(width=0.5, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stats - Multivariate regressions: COT}', fontsize=11)
plt.xlabel(r'\textbf{T-stat}', fontsize=10)
#plt.ylabel(r'\textbf{Regression}', fontsize=10)
plt.yticks(ticks=range(6), labels=list_of_responses)
reg2plot = plt.gcf()
#reg2plot.savefig('C:/Users/yeeya/Figures/reg2plot.pdf')
plt.show()
#
### Multivariate - ALL
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals3copy[notff3].plot.barh(width=0.5, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stats - Multivariate regressions: All features}', fontsize=11)
plt.xlabel(r'\textbf{T-stat}', fontsize=10)
plt.yticks(ticks=range(6), labels=list_of_responses)
reg3plot = plt.gcf()
#reg3plot.savefig('C:/Users/yeeya/Figures/reg3plot.pdf')
plt.show()
##############################################################################
# %% VAR results ##############################################################
# Plot conclusion of Granger causality tests
plt.figure()
plt.bar(range(0,len(var1yesno)),np.array(var1yesno), alpha=.5, label='Feat -> Resp', color='r', align='edge')
plt.bar(range(0,len(var11yesno)), np.array(var11yesno), alpha=.5, label= 'Resp -> Feat', color='b', align='edge')
plt.xticks(ticks=np.arange(0,len(var1yesno),step=6), labels=list_of_responses, fontsize=8)
plt.xlim(0,len(var1yesno))
plt.legend()
plt.grid(axis='x')
##############################################################################
#LinearRegression
## Plot some random draws of LinearRegression predictions (orange) vs reality (blue)
idx = np.arange(0,180,6)
idx = choice(idx) # Pick random prediction
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig2, axs = plt.subplots(7)
fig2.set_figheight(8)
fig2.set_figwidth(7)
fig2.set_label("")
#fig2.suptitle('LinearRegression predictions vs reality', fontsize=12)   
for pred in range(6):
    pd.DataFrame(axLinWO[idx]).plot(ax=axs[0], legend='') # Without
    pd.DataFrame(axLinW[idx + pred]).plot(ax=axs[pred+1]) # With feature
    axs[pred+1].legend('',loc='lower left')
    #axs[0].set_xlim([0,len(axLinWO[idx])])
    #axs[pred+1].set_xlim([0,len(axLinWO[idx])])
fig2.savefig('C:/Users/yeeya/Figures/LinRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# AIC plot LinReg
fig3 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(0,72,2),linresultsWO.iloc[:,-1], align='edge', alpha=0.5, label='OLS: Without Feature')
plt.bar(range(1,73,2),linresultsW.iloc[:,-1], align='edge', alpha=0.5, label='OLS: With Feature')
plt.xticks(ticks=np.arange(0,len(linresultsWO)*2,step=12), labels=list_of_responses)
plt.xlim(0,len(linresultsWO)*2)
plt.legend()
plt.grid(axis='x')
plt.ylabel('AIC')
fig3.savefig('C:/Users/yeeya/Figures/LinRegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)
##############################################################################
# LASSORIDGE
## Plot some random draws of Random Forest predictions (orange) vs reality (blue)
idx = np.arange(0,180,6)
idx = choice(idx) # Pick random prediction .. 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig2, axs = plt.subplots(7)
fig2.set_figheight(8)
fig2.set_figwidth(7)
fig2.set_label("")
#fig2.suptitle('LinearRegression predictions vs reality', fontsize=12)   
for pred in range(6):
    pd.DataFrame(axLinWO[idx]).plot(ax=axs[0], legend='') # Without
    pd.DataFrame(axLinW[idx + pred]).plot(ax=axs[pred+1]) # With feature
    axs[pred+1].legend('',loc='lower left')
    axs[0].set_xlim([0,len(axLinWO[idx])])
    axs[pred+1].set_xlim([0,len(axLinWO[idx])])
fig2.savefig('C:/Users/yeeya/Figures/LRRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# AIC plot Ridge,Lasso regs
fig3 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(0,144,4),ridgeresultsWO.iloc[:,-1], align='edge', alpha=0.5, label='Ridge: Without Feature')
plt.bar(range(1,145,4),ridgeresultsW.iloc[:,-1], align='edge', alpha=0.5, label='Ridge: With Feature')
plt.bar(range(2,146,4),lassoresultsWO.iloc[:,-1], align='edge', alpha=0.5, label='Lasso: Without Feature')
plt.bar(range(3,147,4),lassoresultsW.iloc[:,-1], align='edge', alpha=0.5, label='Lasso: With Feature')
plt.xticks(ticks=np.arange(0,len(ridgeresultsWO)*4,step=24), labels=list_of_responses)
plt.xlim(0,len(ridgeresultsWO)*4)
plt.legend(loc='lower right')
plt.grid(axis='x')
plt.ylabel('AIC')
fig3.savefig('C:/Users/yeeya/Figures/RLRegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

# AIC with PCs: OLS, Ridge and Lasso
fig4 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(0,18,3),linresultsWPCA.iloc[:,-1], align='edge', alpha=0.5, label='OLS PC')
plt.bar(range(1,19,3),ridgeresultsWPCA.iloc[:,-1], align='edge', alpha=0.5, label='Ridge PC')
plt.bar(range(2,20,3),lassoresultsWPCA.iloc[:,-1], align='edge', alpha=0.5, label='Lasso PC')
plt.xticks(ticks=np.arange(0,len(linresultsWPCA)*3,step=3), labels=list_of_responses)
plt.xlim(0,len(linresultsWPCA)*3)
plt.legend()
plt.grid(axis='x')
plt.ylabel('AIC')
fig4.savefig('C:/Users/yeeya/Figures/PCRegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

# MSE plot Forest
fig5 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(0,18,3),RFRresultsWO.iloc[:,-1], align='edge', alpha=0.5, label='Forest: Without Features')
plt.bar(range(1,19,3),RFRresultsW.iloc[:,-1], align='edge', alpha=0.5, label='Forest: With Features')
plt.bar(range(2,20,3),RFRresultsWPCA.iloc[:,-1], align='edge', alpha=0.5, label='Forest: With PCs')
plt.xticks(ticks=np.arange(0,len(RFRresultsWO)*3,step=3), labels=list_of_responses)
plt.xlim(0,len(RFRresultsWO)*3)
plt.legend(loc='lower left')
plt.grid(axis='x')
plt.ylabel('Mean Squared Error')
fig5.savefig('C:/Users/yeeya/Figures/RFRresults.pdf', bbox_inches = 'tight', pad_inches = 0)

# MSE plot Forest
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig6, axs6 = plt.subplots(3, sharex=True)
fig6.set_figheight(8)
fig6.set_figwidth(7)
fig6.set_label("")
FIWOsplit5[ranked1].plot(ax=axs6[0], kind='barh', alpha=0.5, 
       color= 'red',
       label='Feature Importances: Without features')
FIWsplit5[ranked2].plot(ax=axs6[1], kind='barh',  alpha=0.5, 
       color= 'blue',
       label='Feature Importances: With features')
FIWPCAsplit5[ranked3].plot(ax=axs6[2], kind='barh', alpha=0.5, 
       color= 'purple',
       label='Feature Importances: PCs')
fig6.legend(loc='lower left')
plt.xlabel('Relative Importance')
fig6.savefig('C:/Users/yeeya/Figures/RFRFI.pdf', bbox_inches = 'tight', pad_inches = 0)
# %%   TABLES 
"""
## Export COT table to latex
with open('dfCOT.tex','w') as tf:
    tf.write(dfCOT.head().to_latex(float_format='%.3f'))

## Export all features table to latex
with open('dfall[notff3].tex','w') as tf:
    tf.write(dfall[notff3].head().to_latex(float_format='%.3f'))
    
## Export portfolio table to latex
with open('Pdata2.tex','w') as tf:
    tf.write(Pdata2.head().to_latex(float_format='%.3f'))
    
## Export table unigls to latex
with open('params1na.tex','w') as tf:
    tf.write(params1na.to_latex())
    
## Export table glsfeatures to latex
with open('params2copy.tex','w') as tf:
    tf.write(params2copy.to_latex())
    
## Export table glspcs to latex
with open('params3copy.tex','w') as tf:
    tf.write(params3copy.to_latex())
    
## Export table corrmatrix to latex
with open('dfall[notff3].corr().tex','w') as tf:
    tf.write(dfall[notff3].corr().to_latex(float_format='%.3f'))
    
## Export tables VARpvalues1 and VARpvalues11 to latex
with open('var1pvals.tex','w') as tf:
    tf.write(var1pvals.to_latex())
with open('var11pvals.tex','w') as tf:
    tf.write(var11pvals.to_latex())
"""
##############################################################################