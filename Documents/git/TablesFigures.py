# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:09:13 2019

@author: yeeya
"""
#from sklearn.tree.export import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# %% ## Printing section of data
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
#
### GLS results ##############################################################
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
#
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
#
### VAR results ##############################################################
plt.figure()
plt.bar(range(0,len(var1pvals)),np.array(var1yesno), alpha=.5, label='With feature', color='r', align='edge')
plt.bar(range(0,len(var11pvals)), np.array(var11yesno), alpha=.5, label= 'Without', color='b', align='edge')
plt.xticks(ticks=np.arange(0,len(var1pvals),step=6), labels=list_of_responses, fontsize=8)
plt.xlim(0,len(var1pvals))
plt.legend()
plt.grid(b=None,axis='x')
##############################################################################

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
    
## Export tables to latex
with open('params1na.tex','w') as tf:
    tf.write(params1na.to_latex(float_format='%.4f'))
"""
##############################################################################