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

### PC feature relation
fig4 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
corrplot = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0, 
                       cmap=sns.diverging_palette(20, 220, n=200), square=True)
corrplot.set_xticklabels(corrplot.get_xticklabels(), rotation=45, 
                         horizontalalignment='right')
fig4 = plt.gcf()
fig4.savefig('C:/Users/yeeya/Figures/corrplot.pdf', bbox_inches = 'tight', pad_inches = 0)
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
for i in range(6): 
    plt.bar(range(0,36,1),np.array(linresultsW.iloc[:,-1].values-linresultsWO.iloc[:,-1].values), 
            align='edge', alpha=1, color=['C'+str(i) for i in range(6)])
    plt.bar(range(0,36,1), range(0,1,1), label=notff3[i])
    plt.xticks(ticks=np.arange(0,len(linresultsWO),step=6), labels=list_of_responses)
    plt.xlim(0,len(linresultsWO))
plt.legend(ncol=1, bbox_to_anchor=[1, 0], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('$\Delta$ AIC')
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
for i in range(6): 
    plt.bar(range(0,36,1),np.array(ridgeresultsW.iloc[:,-1].values-ridgeresultsWO.iloc[:,-1].values), 
            align='edge', alpha=1, color=['C'+str(i) for i in range(6)])
    plt.bar(range(0,36,1), range(0,1,1), label=notff3[i])
    plt.xticks(ticks=np.arange(0,len(ridgeresultsW),step=6), labels=list_of_responses)
    plt.xlim(0,len(ridgeresultsW))
plt.legend(ncol=1, bbox_to_anchor=[1, 0], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('$\Delta$ AIC')
fig3.savefig('C:/Users/yeeya/Figures/R_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

fig4 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(6): 
    plt.bar(range(0,36,1),np.array(lassoresultsW.iloc[:,-1].values-lassoresultsWO.iloc[:,-1].values), 
            align='edge', alpha=1, color=['C'+str(i) for i in range(6)])
    plt.bar(range(0,36,1), range(0,1,1), label=notff3[i])
    plt.xticks(ticks=np.arange(0,len(lassoresultsW),step=6), labels=list_of_responses)
    plt.xlim(0,len(lassoresultsW))
plt.legend(ncol=1, bbox_to_anchor=[1, 0], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('$\Delta$ AIC')
fig4.savefig('C:/Users/yeeya/Figures/L_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)
#
# AIC with PCs: OLS, Ridge and Lasso
dAIC1 = linresultsWPCA.iloc[:,-1].values-[linresultsWO.iloc[i,-1] for i in range(0,36,6)]
dAIC2 = ridgeresultsWPCA.iloc[:,-1].values-[ridgeresultsWO.iloc[i,-1] for i in range(0,36,6)]
dAIC3 = lassoresultsWPCA.iloc[:,-1].values-[lassoresultsWO.iloc[i,-1] for i in range(0,36,6)]
fig4 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(6): 
    plt.bar(range(0,18,3),dAIC1,
            align='edge', alpha=0.2, color=['C'+str(i) for i in range(6)])
    plt.bar(range(1,19,3),dAIC2,
            align='edge', alpha=0.4, color=['C'+str(i) for i in range(6)])
    plt.bar(range(2,20,3),dAIC3,
            align='edge', alpha=0.75, color=['C'+str(i) for i in range(6)])
    plt.xticks(ticks=np.arange(0,18,step=3), labels=list_of_responses)
    plt.xlim(0,18)
plt.bar(range(0,18,3),range(0,1,1),
        align='edge', alpha=0.3, color=['C0'], label='OLS')
plt.bar(range(0,18,3),range(0,1,1),
        align='edge', alpha=0.4, color=['C0'], label='Ridge')
plt.bar(range(0,18,3),range(0,1,1),
        align='edge', alpha=0.8, color=['C0'], label='Lasso')
plt.legend(ncol=1)
plt.grid(axis='x')
plt.ylabel('$\Delta$ AIC')
fig4.savefig('C:/Users/yeeya/Figures/PC_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

# MSE plot Forest
fig5 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(6): 
    plt.bar(range(0,12,2),np.array(RFRresultsW.iloc[:,-1].values-RFRresultsWO.iloc[:,-1].values), 
            align='edge', alpha=0.25, color='C0')
    plt.bar(range(1,13,2),np.array(RFRresultsWPCA.iloc[:,-1].values-RFRresultsWO.iloc[:,-1].values), 
            align='edge', alpha=0.25, color='C1')
    plt.xticks(ticks=np.arange(0,len(RFRresultsW)*2,step=2), labels=list_of_responses)
    plt.xlim(0,len(RFRresultsW)*2)
plt.bar(range(0,12,2), range(0,1,1),align='edge', alpha=0.6, 
        color=['C0'], label='Forest: $MSE_{X}-MSE_{X\'}$')
plt.bar(range(1,13,2), range(0,1,1),align='edge', alpha=0.6, 
        color=['C1'], label='Forest: $MSE_{X_{PC}}-MSE_{X\'}$')
plt.legend(ncol=2, bbox_to_anchor=[0, -0.25], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('$\Delta$ Mean Squared Error')
fig5.savefig('C:/Users/yeeya/Figures/RFRresults.pdf', bbox_inches = 'tight', pad_inches = 0)

# Feature importance plot Forest
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig6, axs6 = plt.subplots(3, sharex=True)
fig6.set_figheight(7)
fig6.set_figwidth(7)
for i in range(0,6,1): 
    FIWOsplit5.mean(axis=1)[rankedWO].plot(ax=axs6[0], kind='barh', alpha=0.5, 
            color= 'mediumaquamarine', edgecolor = 'black')
    FIWsplit5.mean(axis=1)[rankedW].plot(ax=axs6[1], kind='barh',  alpha=0.5, 
           color= 'blue', edgecolor = 'black')
    FIWPCAsplit5[ranked3].plot(ax=axs6[2], kind='barh', alpha=0.5, 
           color= 'seagreen', edgecolor = 'black')
fig6.legend('')
plt.xlabel('Mean Relative Importance')
fig6.savefig('C:/Users/yeeya/Figures/RFRFI.pdf', bbox_inches = 'tight', pad_inches = 0)

# Visualization, 100 trees:
i_tree = 0
for tree_in_forest in forest_regWO[-1].estimators_:
    with open('tree_' + str(i_tree) + '.png', 'w') as retpostree2:
        retpostree2 = export_graphviz(tree_in_forest, out_file = 'retpostree2.png')
    i_tree = i_tree + 1
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
    
## Export table corrmatrix to latex
with open('corrmat.tex','w') as tf:
    tf.write(corrmat.to_latex(float_format='%.3f'))
    
## Export tables VARpvalues1 and VARpvalues11 to latex
with open('var1pvals.tex','w') as tf:
    tf.write(var1pvals.to_latex())
with open('var11pvals.tex','w') as tf:
    tf.write(var11pvals.to_latex())
    
## Export table corrmatrix to latex
with open('RFRresultsWO.tex','w') as tf:
    tf.write(RFRresultsWO.to_latex(float_format='%.3f'))
    
## Export table Feature Importance table to latex
with open('FIWPCAsplit5.tex','w') as tf:
    tf.write(FIWPCAsplit5.to_latex(float_format='%.3f'))
"""
##############################################################################