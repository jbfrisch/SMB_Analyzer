# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 17:19:42 2017

@author: jb.frisch
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from PortfolioOptimization.utils import timing as chrono

'''
##############################################################################
                            UZAWA ALGORITHM
##############################################################################
'''
def uz_Xk(lmbda, mu, Sig_mat, R_arr):
    '''Function computing the x* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint 
    and weights sum equal to 1.
    
    `lmbda` is a scalar \n
    `mu` is a n*1 ndarray/matrix of inequalities constraints \n
    `Sig_mat` is a n*n inverted var cov matrix \n
    
    Return a ndarray of size n*1
    '''
    p_size = Sig_mat.shape[0]

    return - np.dot(np.repeat(lmbda[0], p_size) + (lmbda[1] * R_arr) + mu[:,0] - mu[:,1], Sig_mat).T

def uz_lk(lmbda, rho, R_arr, r_target, xk):
    '''Function computing the lambda* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint
    and weights sum equal to 1.

    `lmbda` is a scalar \n
    `rho` is a scalar \n
    `xk` is a n*1 ndarray \n
    
    Return a scalar (only one equality constraint)
    '''
    lmbda_k = lmbda
    lmbda_k[0] = lmbda_k[0] + rho * (xk.sum() - np.float64(1.0))
    lmbda_k[1] = lmbda_k[1] + rho * (np.dot(R_arr, xk) - r_target)
    
    return lmbda_k

def uz_muk(mu, rho, xk, lw_bnd, up_bnd):
    '''Function computing the mu* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint
    and weights sum equal to 1.

    `mu` is a n*1 ndarray/matrix
    `rho` is a scalar \n
    `xk` is a n*1 ndarray \n
    
    Return a ndarray of size n*1 (n inequality constraints)
    '''
    muk = mu 
    muk[:,0] = muk[:,0] + rho * (xk.T - lw_bnd)
    muk[:,1] = muk[:,1] + rho * (up_bnd - xk.T)
    muk[np.where(muk > 0.0)] = np.float64(0.0)
    return muk

def uz_Scheme(Sig_mat, lmbda, mu, rho, R_arr, r_target, lw_bnd, up_bnd):
    '''Function executing Uzawa algorithm. It computes first de x* value at
    the k-th step and next the mu* and lambda*
    
    Return the x_k*, mu_k*, lambda_k*
    '''
    x_k = uz_Xk(lmbda, mu, Sig_mat, R_arr)
    mu_k = uz_muk(mu, rho, x_k, lw_bnd, up_bnd)
    lambda_k = uz_lk(lmbda, rho, R_arr,r_target, x_k)
    
    return [x_k, lambda_k, mu_k]

def d_Lagrangian(x_k, sigma, r, lmbda_k, mu_k):
    return x_k.T * sigma + np.repeat(lmbda_k[0],sigma.shape[0]) + lmbda_k[1] * r + mu_k[:,0] - mu_k[:,1]

@chrono.timing
def Uzawa_MeanVar(Sig_mat, Ret_mat, r_Obj, bounds, lmbda_init, mu_init, rho, Err_Threshold=1.0e-10, iterMax=5000):
    '''Function executing the Uzawa optimization algorithm adapted for a Mean Variance portfolio
    with "`no short selling`" constraint.
    It takes into parameters:
        `Sig_mat` as a n*n inverted var cov matrix \n
        `lmbda_init` as a scalar \n
        `mu_init` as a n*1 ndarray/matrix \n
        `rho` as a scalar \n
        `Err_Threshold` (Optional) as a float value. Default value 1e-10 \n
        `iterMax` (Optional) as an Integer. Default value 5000 \n
        
    The function returns a list with the solution found with Lagrangian parameters computed (lambda and mu)
    and a tab of error at each step to study the convergence.
    '''
    err_k = 100.0
    err_tab = []
    inv_Sig = inv(Sig_mat)
    
    x_k, lambda_k, mu_k = uz_Scheme(inv_Sig, lmbda_init, mu_init, rho, Ret_mat, r_Obj,bounds[0], bounds[1])
    k = 0
    while err_k > Err_Threshold and k < iterMax:
        x_kk, lambda_kk, mu_kk = uz_Scheme(inv_Sig, lambda_k, mu_k, rho, Ret_mat, r_Obj,bounds[0], bounds[1])
        #err_k = np.abs((x_kk - x_k).sum())
        err_k = norm(d_Lagrangian(x_kk, Sig_mat, Ret_mat, lambda_kk, mu_kk))
        err_tab.append(err_k)
        
        lambda_k = lambda_kk
        mu_k = mu_kk
        #x_k = x_kk
        #print '%.d  ----- Weigths: %.2f | %.2f | %.2f | %.2f ---- Error : %.10f' % (k, x_kk[0], x_kk[1], x_kk[2], x_kk[3],(x_kk - x_k).sum())
        k += 1
    
    print 'Terminates in %.d iter  --- Weigths: %.2f | %.2f | %.2f | %.2f --- Error : %.5f' % (k, x_kk[0], x_kk[1], x_kk[2], x_kk[3], err_k)    
    return [x_kk, lambda_k, mu_k, err_tab, k]

'''
##############################################################################
'''

def main():
    lmbda_init = np.ones([2,1]) * 1.0
    mu_init = np.ones([4,2]) * -1.0
    Returns = np.array([5.0, 9.0, 7.0, 6.0]) / 100.0
    test_Corr_Matrix = np.matrix([[100.0, 10.0, 40.0, -10.0],
                                  [10.0, 100.0, 20.0, -10.0],
                                  [40.0, 20.0, 100.0, -20.0],
                                  [-10.0, -10.0, -20.0, 100.0]]) / 100.0

    test_Std_Vect = np.array([4.0, 15.0, 5.0, 10.0]) / 100.0
    D = np.diag(test_Std_Vect)

    test_VCov_Mat = D * test_Corr_Matrix * D

    ''' 
    ------------- Test
    ''' 
    Uzawa_MeanVar(inv(test_VCov_Mat), Returns, 0.06, [0.0, 0.4], lmbda_init, mu_init, 1.0e-5, 1.0e-10, 500000)

if __name__ == '__main__':
    main()