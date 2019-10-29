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
#fig.savefig('./Figures/FeaturesSeries.pdf', bbox_inches = 'tight', pad_inches = 0)
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
#reg1plot.savefig('./Figures/reg1plot.pdf')
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
#reg2plot.savefig('./Figures/reg2plot.pdf')
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
#reg3plot.savefig('./Figures/reg3plot.pdf')
plt.show()
##############################################################################
# Scatter plots
plt.scatter(Pdata2.iloc[:,1],dfallPC.iloc[:,2])
plt.scatter(Pdata2.iloc[:,1],dfallPC.iloc[:,2])
plt.scatter(Pdata2.iloc[:,1],dfallPC.iloc[:,2])
plt.scatter(Pdata2.iloc[:,1],dfallPC.iloc[:,2])
# %% VAR results ##############################################################
# Plot conclusion of Granger causality tests
plt.figure()
plt.bar(range(0,len(var1yesno)),np.array(var1yesno), alpha=.5, label='Feat -> Resp', color='r', align='edge')
plt.bar(range(0,len(var11yesno)), np.array(var11yesno), alpha=.5, label= 'Resp -> Feat', color='b', align='edge')
plt.xticks(ticks=np.arange(0,len(var1yesno),step=6), labels=list_of_responses, fontsize=8)
plt.xlim(0,len(var1yesno))
plt.legend()
plt.grid(b=None,axis='x')
##############################################################################
# ! PLOTTING MUST BE DONE BEFORE DATAFRAMING THE RESULTS !
## Plot test AIC results, LinearRegression with and without features/PCs
plt.bar(np.arange(0,len(linresultsWO),step=1), height=linresultsWO, 
                  align='edge', alpha=0.5, label='Without Features', zorder=2)
plt.bar(np.arange(0,len(linresultsW),step=1), height=linresultsW, 
                  align='edge', alpha=0.5, label='With Features', zorder=2)
plt.bar(np.arange(0,len(linresultsWPCA),step=1), height=linresultsWPCA, 
                  align='edge', alpha=0.5, label='With PCs', zorder=2)
plt.xticks(ticks=np.arange(0,len(linresultsWO),step=6))
plt.legend()
plt.grid(b=None,axis='x')
##############################################################################
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
    axs[0].set_xlim([0,len(axLinWO[idx])])
    axs[pred+1].set_xlim([0,len(axLinWO[idx])])
fig2.savefig('./Figures/LinRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

## Plot some random draws of Random Forest predictions (orange) vs reality (blue)
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
    axs[0].set_xlim([0,len(axLinWO[idx])])
    axs[pred+1].set_xlim([0,len(axLinWO[idx])])
fig2.savefig('./Figures/LinRegPred.pdf', bbox_inches = 'tight', pad_inches = 0)
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
    
## Export tables VARpvalues1 and VARpvalues11 to latex
with open('var1pvals.tex','w') as tf:
    tf.write(var1pvals.to_latex())
with open('var11pvals.tex','w') as tf:
    tf.write(var11pvals.to_latex())
"""
##############################################################################