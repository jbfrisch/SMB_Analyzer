# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:36:35 2017

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
def uz_Xk(lmbda, mu, Sig_mat):
    '''Function computing the x* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint 
    and weights sum equal to 1.
    
    `lmbda` is a scalar \n
    `mu` is a n*1 ndarray/matrix of inequalities constraints \n
    `Sig_mat` is a n*n inverted var cov matrix \n
    
    Return a ndarray of size n*1
    '''
    p_size = Sig_mat.shape[0]
    return -np.dot(np.repeat(lmbda, p_size) + mu.T, Sig_mat).T

def uz_lk(lmbda, rho, xk):
    '''Function computing the lambda* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint
    and weights sum equal to 1.

    `lmbda` is a scalar \n
    `rho` is a scalar \n
    `xk` is a n*1 ndarray \n
    
    Return a scalar (only one equality constraint)
    '''
    return lmbda + rho * (xk.sum() - 1.0)

def uz_muk(mu, rho, xk):
    '''Function computing the mu* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint
    and weights sum equal to 1.

    `mu` is a n*1 ndarray/matrix
    `rho` is a scalar \n
    `xk` is a n*1 ndarray \n
    
    Return a ndarray of size n*1 (n inequality constraints)
    '''
    muk = mu + rho * xk
    muk[np.where(muk > 0.0)] = 0.0
    return muk

def uz_Scheme(Sig_mat, lmbda, mu, rho):
    '''Function executing Uzawa algorithm. It computes first de x* value at
    the k-th step and next the mu* and lambda*
    
    Return the x_k*, mu_k*, lambda_k*
    '''
    x_k = uz_Xk(lmbda, mu, Sig_mat)
    mu_k = uz_muk(mu, rho, x_k)
    lambda_k = uz_lk(lmbda, rho, x_k)
    
    return [x_k, lambda_k, mu_k]

def d_Lagrangian(x_k, sigma, lmbda_k, mu_k):
    return x_k.T * sigma + np.repeat(lmbda_k,sigma.shape[0]) + mu_k.T
    

@chrono.timing
def Uzawa_MinVar(Sig_mat, lmbda_init, mu_init, rho, Err_Threshold=1.0e-10, iterMax=5000, Inv_Mat = None):
    """
    Function executing the Uzawa optimization algorithm adapted for a MinVar portfolio
    with "`no short selling`" constraint.
    
    Parameters
    ----------
    `Sig_mat` : as a n*n inverted var cov matrix \n
    `lmbda_init` : as a scalar \n
    `mu_init` : as a n*1 ndarray/matrix \n
    `rho` : as a scalar \n
    `Err_Threshold` : (Optional) as a float value. Default value 1e-10 \n
    `iterMax` : (Optional) as an Integer. Default value 5000 \n
    
    Returns
    -------    
    The function returns a list with the solution found with Lagrangian parameters computed (lambda and mu)
    and a tab of error at each step to study the convergence.
    """
    err_k = 100.0
    err_tab = []
    
    if Inv_Mat is None:
        inv_Sig = inv(Sig_mat)
    else:
        inv_Sig = Inv_Mat
    
    x_k, lambda_k, mu_k = uz_Scheme(inv_Sig, lmbda_init, mu_init, rho)
    k = 0
    while err_k > Err_Threshold and k < iterMax:
        x_kk, lambda_kk, mu_kk = uz_Scheme(inv_Sig, lambda_k, mu_k, rho)
        #err_k = np.abs((x_kk - x_k).sum())
        #err_k = np.abs(x_kk.T * mu_kk)
        err_k = norm(d_Lagrangian(x_kk, Sig_mat, lambda_kk, mu_kk))
        err_tab.append(np.float(err_k))
        
        lambda_k = lambda_kk
        mu_k = mu_kk
        k += 1
    
    print 'Terminates in %.d iter  --- Weigths: %.2f | %.2f | %.2f | %.2f --- Error : %.5f' % (k, x_kk[0], x_kk[1], x_kk[2], x_kk[3], err_k)    
    return [x_kk, lambda_k, mu_k, err_tab, k]

'''
##############################################################################
'''

@chrono.timing
def min_Var_Optimization(vcov):
    ''' Compute the portfolio's weights given an estimated covariance Matrix.
    
    Weights are computed as the Lagrange solution of the Global Min-Var optimization problem
    
    `Est_Vcov_Matrix` is a n*n ndarray/matrix
    
    Return a 1*n ndarray
    '''
    Inv_VCov_Matrix = inv(vcov)

    Sigma_Ones = Inv_VCov_Matrix.sum(axis=0)
    One_Sigma_One = Sigma_Ones.sum()

    return Sigma_Ones / One_Sigma_One

'''
##############################################################################
'''

def main():
    lmbda_k = 1.0
    mu_k = np.ones([4,1])*1.0

    test_Corr_Matrix = np.matrix([[100.0,10.0,40.0,50.0],
                                  [10.0,100.0,70.0,40.0],
                                  [40.0,70.0,100.0,80.0],
                                  [50.0,40.0,80.0,100.0]]) / 100.0

    test_Std_Vect = np.array([15.0,20.0,25.0,30.0]) / 100.0
    D = np.diag(test_Std_Vect)

    test_VCov_Mat = D * test_Corr_Matrix * D
    ''' 
    ------------- Test
    ''' 
    min_Var_Optimization(test_VCov_Mat)

    Uzawa_MinVar(inv(test_VCov_Mat), lmbda_k, mu_k, 0.001, 1.0e-10, 50000)

    
if __name__ == '__main__':
    main()