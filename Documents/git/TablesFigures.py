# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:09:13 2019

@author: iles_
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
fig1, axs1 = plt.subplots(len(notff3),sharex=True)
fig1.set_figheight(6)
fig1.set_figwidth(8)
fig1.set_label("")
#fig1.suptitle('Timeseries - All features', fontsize=12)   
for exog, i, color in zip(notff3,range(len(notff3)), sixcolors):
    dfall[exog].plot(ax=axs1[i], color=[color], legend=exog)
    axs1[i].legend(loc='lower left')
    axs1[i].set(xlabel="")
#fig1.savefig('C:/Users/iles_/Figures/FeaturesSeries.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# Descriptive statistics
DStats       = regdata.describe().append([regdata.skew(), regdata.kurt()],ignore_index=True)
DStats.index = ['obs', 'mean', 'std', 'min', '25%', '50%', '75%', 'max','skew','kurt']

# Plot example of region splits for a decion tree
fig3d = plt.figure()
ax3d  = fig3d.add_subplot(111)
r1    = 0.084
ax3d.scatter(regdata.cefd, regdata.SMALLHiBM)
ax3d.axvline(x=r1)
ax3d.axhline(y=0.028, ls='--', xmax=0.585, c = 'k')
ax3d.axhline(y=0.0, ls='--', xmin=0.585, c = 'k')
plt.xlim(-0.025,0.16)
plt.ylim(-0.22,0.2)
ax3d.axvspan(-0.025,r1, alpha = 0.3, color='C0')
ax3d.axvspan(r1,0.16, alpha = 0.3, color='C1')
ax3d.set_xlabel('cefd')
ax3d.set_ylabel('SMALLHiBM')
fig3d.savefig('C:/Users/iles_/Figures/fig3d.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()
###############################################################################
# %% GLS results ##############################################################
### Univariate - T stats
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
#reg1plot.savefig('C:/Users/iles_/Figures/reg1plot.pdf')
plt.show()
##############################################################################
### Multivariate - T stats
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tvals3copy[notff3].plot.barh(width=0.5, grid=True, align='edge' , zorder=3)
plt.title(r'\textbf{T-stats - Multivariate regressions: All features}', fontsize=11)
plt.xlabel(r'\textbf{T-stat}', fontsize=10)
plt.yticks(ticks=range(6), labels=list_of_responses)
reg3plot = plt.gcf()
#reg3plot.savefig('C:/Users/iles_/Figures/reg3plot.pdf')
plt.show()

### PC feature relation
fig2 = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
corrplot = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0, 
                       cmap=sns.diverging_palette(20, 220, n=200), square=True)
corrplot.set_xticklabels(corrplot.get_xticklabels(), rotation=45, 
                         horizontalalignment='right')
fig2 = plt.gcf()
fig2.savefig('C:/Users/iles_/Figures/corrplot.pdf', bbox_inches = 'tight', pad_inches = 0)
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
# LinearRegression
## Plot some random draws of LinearRegression predictions (orange) vs reality (blue)
idx = np.arange(0,180,6)
idx = choice(idx) # Pick random prediction
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig3, axs3 = plt.subplots(7)
fig3.set_figheight(8)
fig3.set_figwidth(7)
fig3.set_label("")
#fig3.suptitle('LinearRegression predictions vs reality', fontsize=12)   
for pred in range(6):
    pd.DataFrame(axLinWO[idx]).plot(ax=axs3[0], legend='') # Without
    pd.DataFrame(axLinW[idx + pred]).plot(ax=axs3[pred+1]) # With feature
    axs3[pred+1].legend('',loc='lower left')
    #axs[0].set_xlim([0,len(axLinWO[idx])])
    #axs[pred+1].set_xlim([0,len(axLinWO[idx])])
fig3.savefig('C:/Users/iles_/Figures/LinRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# AIC plot LinReg
fig4 = plt.figure()
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
fig4.savefig('C:/Users/iles_/Figures/LinRegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

Linprog = []
for i in range(6): 
    Linprog.append(np.array(np.mean(linresultsW[i::6])-np.mean(linresultsWO[i::6])))
Linprog = pd.DataFrame(Linprog, columns = linresultsW.columns, index=notff3)

# Mean AIC Progression LinReg
Progfig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(6): 
    plt.plot(Linprog.iloc[i,:], alpha=0.8)
    plt.axhline(y=0, alpha=0.3, color='black', ls='--')
    plt.bar(range(0,6,1), range(0,1,1), label=notff3[i])
    plt.xlim(0,4)
plt.legend(ncol=1, bbox_to_anchor=[1, 0], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('Mean $\Delta$ AIC')
Progfig.savefig('C:/Users/iles_/Figures/Progfig.pdf', bbox_inches = 'tight', pad_inches = 0)
##############################################################################
# LASSORIDGE
## Plot some random draws of Random Forest predictions (orange) vs reality (blue)
idx = np.arange(0,180,6)
idx = choice(idx) # Pick random prediction .. 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig5, axs5 = plt.subplots(7)
fig5.set_figheight(8)
fig5.set_figwidth(7)
fig5.set_label("")
#fig5.suptitle('LinearRegression predictions vs reality', fontsize=12)   
for pred in range(6):
    pd.DataFrame(axLinWO[idx]).plot(ax=axs5[0], legend='') # Without
    pd.DataFrame(axLinW[idx + pred]).plot(ax=axs5[pred+1]) # With feature
    axs5[pred+1].legend('',loc='lower left')
    axs5[0].set_xlim([0,len(axLinWO[idx])])
    axs5[pred+1].set_xlim([0,len(axLinWO[idx])])
fig5.savefig('C:/Users/iles_/Figures/LRRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# AIC plot Ridge,Lasso regs
fig6 = plt.figure()
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
fig6.savefig('C:/Users/iles_/Figures/R_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

fig7 = plt.figure()
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
fig7.savefig('C:/Users/iles_/Figures/L_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)
#
# AIC with PCs: OLS, Ridge and Lasso
dAIC1 = linresultsWPCA.iloc[:,-1].values-[linresultsWO.iloc[i,-1] for i in range(0,36,6)]
dAIC2 = ridgeresultsWPCA.iloc[:,-1].values-[ridgeresultsWO.iloc[i,-1] for i in range(0,36,6)]
dAIC3 = lassoresultsWPCA.iloc[:,-1].values-[lassoresultsWO.iloc[i,-1] for i in range(0,36,6)]
fig8 = plt.figure()
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
fig8.savefig('C:/Users/iles_/Figures/PC_RegAIC.pdf', bbox_inches = 'tight', pad_inches = 0)

# MSE plot Forest
fig9 = plt.figure()
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
        color=['C0'], label='$ \Delta MSE_{Feat}$')
plt.bar(range(1,13,2), range(0,1,1),align='edge', alpha=0.6, 
        color=['C1'], label='$ \Delta MSE_{PC}$')
plt.legend(ncol=1, bbox_to_anchor=[1, 0], loc='lower left', 
           fontsize='small', fancybox=True, shadow=True)
plt.grid(axis='x')
plt.ylabel('$ \Delta$ Mean Squared Error')
fig9.savefig('C:/Users/iles_/Figures/RFRresults2.pdf', bbox_inches = 'tight', pad_inches = 0)

# Feature importance plot Forest
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig10, axs10 = plt.subplots(3, sharex=True)
fig10.set_figheight(7)
fig10.set_figwidth(7)
for i in range(0,6,1): 
    FIWOsplit5.mean(axis=1)[rankedWO].plot(ax=axs10[0], kind='barh', alpha=0.5, 
            color= 'mediumaquamarine', edgecolor = 'black')
    FIWsplit5.mean(axis=1)[rankedW].plot(ax=axs10[1], kind='barh',  alpha=0.5, 
           color= 'blue', edgecolor = 'black')
    FIWPCAsplit5.mean(axis=1)[ranked3].plot(ax=axs10[2], kind='barh', alpha=0.5, 
           color= 'seagreen', edgecolor = 'black')
fig10.legend('')
plt.xlabel('Mean Relative Importance')
fig10.savefig('C:/Users/iles_/Figures/RFRFI2.pdf', bbox_inches = 'tight', pad_inches = 0)

# Visualization, 100 trees:
i_tree = 0
for tree_in_forest in forest_regWO[-1].estimators_:
    with open('tree_' + str(i_tree) + '.png', 'w') as retpostree2:
        retpostree2 = export_graphviz(tree_in_forest, out_file = 'retpostree2.png')
    i_tree = i_tree + 1


# Gradient descent illustration:

regdataa = pd.DataFrame()
for i in range(15):
    regdataa[i] = yeojohnson(regdata.iloc[:,i])[0]

fig = plt.figure()
ax  = fig.add_subplot(111)
for i in range(15):
    yeojohnson_normplot(regdata.iloc[:,i], -5, 5, plot=ax, N = 299)
    ax.axvline(yeojohnson(regdata.iloc[:,i])[1], color='black', ls='--', lw=0.8)
plt.show()
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