# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:20:27 2019

@author: yeeya
"""

# %% LOGIT ####################################################################

# Transform abnormal returns into logical: 1 iff \abs(response) > \sigma^2
Pdata.columns     = list_of_responses
Pdata3            = pd.DataFrame(Pdata[1::])
Pdata3.index      = regdata.index
retpos            = Pdata3.abs() > pd.core.window.Expanding(Pdata3, min_periods=12, axis=0).std().bfill()
retpos            = retpos.astype(int)
retpos            = pd.DataFrame(retpos)
retpos.columns    = list_of_responses
regdata2          = pd.concat([retpos,dfall],axis=1,ignore_index=False)

# X defined as [ones, cefd, ret]
# One regression per stock portfolio
regL1  = []
tvalsL1 = []
for resp in list_of_responses:
    for exog in notff3:
        formula = resp + " ~ " + exog
        regL1.append(sm.Logit.from_formula(formula, data=regdata2).fit())
        print(regL1[len(regL1)-1].summary2())
        print(regL1[len(regL1)-1].get_margeff(at='overall').summary())
        tvalsL1.append(regL1[len(regL1)-1].get_margeff(at='overall').tvalues)
tvalsL1copy = np.array(tvalsL1).reshape(-6,6)
print('Mean of pvalues mfx =', np.abs(tvalsL1copy).mean(axis=0))
sns.regplot(regdata2[['cefd']],regdata2[[list_of_responses[3]]], logistic=1)

# X defined as [ones, [dfCOT]]
# One regression per stock portfolio
regL2  = []
tvalsL2 = []
for resp in list_of_responses:
    formula = resp + " ~ NONPNL + CPNL + NCPNL + OI"
    regL2.append(sm.Logit.from_formula(formula, data=regdata2).fit())
    print(regL2[len(regL2)-1].summary2())
    print(regL2[len(regL2)-1].get_margeff(at='overall').summary())
    tvalsL2.append(regL2[len(regL2)-1].get_margeff(at='overall').tvalues)
tvalsL2copy = np.array(tvalsL2).reshape(-6,2)
print('Mean of pvalues mfx =', np.abs(tvalsL2copy).mean(axis=0))

# X defined as [ones, cefd, dfCOT, vixret, ret]
# One regression per stock portfolio
regL3  = []
tvalsL3 = []
for resp in list_of_responses:
    formula = resp + " ~ cefd + CPNL + NCPNL + vixret "
    regL3.append(sm.Logit.from_formula(formula, data=regdata2).fit())
    print(regL3[len(regL3)-1].summary2())
    print(regL3[len(regL3)-1].get_margeff(at='overall').summary())
    tvalsL3.append(regL3[len(regL3)-1].get_margeff(at='overall').tvalues)
tvalsL3copy = np.array(tvalsL3).reshape(-6,4)
print('Mean of pvalues mfx =', np.abs(tvalsL3copy).mean(axis=0))
"""
plt.rc('figure', figsize=(7, 5))
plt.text(0.01, 0.05, str(logitres2.summary2(yname='retpos', 
                         xname=['const','cefd','CPNL','NCPNL','OI','TOTPNL','vixret'])), 
{'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logitres2.png')
"""
"""
plt.rc('figure', figsize=(7, 3))
plt.text(0.01, 0.05, str(logit2mfx.summary()), {'fontsize': 10}, 
         fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('logit2mfx.png')
"""
##############################################################################