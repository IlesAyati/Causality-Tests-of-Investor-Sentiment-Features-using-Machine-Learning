# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:55:12 2019

@author: yeeya
"""

def mahalanobis(X=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of design matrix X and the distribution of the data.  
    X   : Design matrix with n rows and p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of X is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = X - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

# Test
df_x = regdata[list_of_responses].loc[:,'cefd']
df_x['mahala'] = mahalanobis(X=df_x, data=regdata[list_of_responses])

