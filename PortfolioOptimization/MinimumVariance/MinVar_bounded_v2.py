# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 17:19:42 2017

@author: jb.frisch
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import multi_dot

from numpy import dot

from PortfolioOptimization.utils import timing as chrono

'''
##############################################################################
                            UZAWA ALGORITHM
##############################################################################
'''
def uz_Xk(mu, Sig_mat):
    '''Function computing the x* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with no-short selling constraint 
    and weights sum equal to 1.
    
    `mu` is a n*1 ndarray/matrix of inequalities constraints \n
    `Sig_mat` is a n*n inverted var cov matrix \n
    
    Return a ndarray of size n*1
    '''
    #u = np.ones([1, Sig_mat.shape[0]])
    #return -dot(u*mu[0][:,0] - u*mu[0][:,1] + mu[1][:,0] - mu[1][:,1], Sig_mat).T
    return -dot(mu[0][:,0] - mu[0][:,1] + mu[1][:,0] - mu[1][:,1], Sig_mat).T

def uz_muk(mu, rho, xk, lw_bnd, up_bnd):
    '''Function computing the mu* value at the k-th step in Uzawa algorithm
    This function is adapted to MinVar portfolio with weight constraints [l-;l+]
    and weights sum equal to 1.

    `mu` is a n*1 ndarray/matrix
    `rho` is a scalar \n
    `xk` is a n*1 ndarray \n
    
    Return a ndarray of size n*1 (n inequality constraints)
    '''
    mu_s_k = np.copy(mu[0])
    path = rho * (xk.sum() - 1.0)
    mu_s_k[:,0] = mu_s_k[:,0] + path
    mu_s_k[:,1] = mu_s_k[:,1] - path
    mu_s_k[np.where(mu_s_k > 0.0)] = 0.0
    
    muk = np.copy(mu[1]) 
    muk[:,0] = muk[:,0] + rho * (xk.T - lw_bnd)
    muk[:,1] = muk[:,1] + rho * (up_bnd - xk.T)
    muk[np.where(muk > 0.0)] = 0.0
    
    return [mu_s_k, muk]

def d_Lagrangian(x_k, sigma, mu_k):
    #u = np.ones([1, sigma.shape[0]])
    #return x_k.T * sigma + u*mu_k[0][:,0] - u*mu_k[0][:,1] + mu_k[1][:,0] - mu_k[1][:,1]
    return dot(x_k.T, sigma) + mu_k[0][:,0] - mu_k[0][:,1] + mu_k[1][:,0] - mu_k[1][:,1]

def uz_Scheme(Sig_mat, mu, rho, lw_bnd, up_bnd):
    '''Function executing Uzawa algorithm. It computes first de x* value at
    the k-th step and next the mu* and lambda*
    
    Return the x_k*, mu_k*
    '''
    x_k = uz_Xk(mu, Sig_mat)
    mu_k = uz_muk(mu, rho, x_k, lw_bnd, up_bnd)
    
    return [x_k, mu_k]

#@chrono.timing
def Optim_Path_Wolfe(Sig_mat, x_k, d_k, m1=1e-8, m2=0.5):
    ''' Function performing the optimal path research following the Wolfe's criterias
    The algorithm is suited to the Minimum Variance optimization problem
    '''
    rho = 0.8
    alpha = rho
    rho_p = 100.0
    rho_m = 0.0
    i=0
    # Init Conditions
    wolfe_Cond_1 = False
    wolfe_Cond_2 = False
    while i < 10:
        wolfe_Cond_1 = multi_dot([(x_k - rho * d_k).T , Sig_mat , (x_k - rho * d_k)]) <= multi_dot([x_k.T , Sig_mat , x_k]) + m1 * rho * multi_dot([x_k.T , Sig_mat , d_k])#(x_k.T * Sig_mat).T)
        wolfe_Cond_2 = multi_dot([(x_k - rho * d_k).T , Sig_mat , d_k]) >= -m2 * multi_dot([x_k.T , Sig_mat , d_k])#(x_k.T * Sig_mat).T
        if not wolfe_Cond_1:
            rho_p = rho
            rho = (rho_p + rho_m) / 2.0
        else: 
            if not wolfe_Cond_2:
                #print('not Cond 2')
                rho_m = rho
                if rho_p == 100.0:
                    rho = 3.0 * rho_m
                elif rho_p < 100.0:
                    rho = (rho_p + rho_m) / 2.0
            else:
                break
        i+=1
    alpha = rho
    #print (alpha)
    return alpha
        
def AU_Scheme(x_k, Sig_mat, mu, rho, lw_bnd, up_bnd):
    d_k = d_Lagrangian(x_k, Sig_mat, mu).T
    alpha_k = Optim_Path_Wolfe(Sig_mat, x_k, d_k)
    x_kk = x_k - alpha_k * d_k
    mu_kk = uz_muk(mu, rho, x_kk, lw_bnd, up_bnd)
    
    return [x_kk, mu_kk]
    
@chrono.timing
def Uzawa_MinVar(Sig_mat, bounds, mu_init, rho, Err_Threshold=1.0e-10, iterMax=5000):
    '''Function executing the Uzawa optimization algorithm adapted for a MinVar portfolio
    with "`no short selling`" constraint.
    It takes into parameters:
        `Sig_mat` as a n*n inverted var cov matrix \n
        `mu_init` as a n*1 ndarray/matrix \n
        `rho` as a scalar \n
        `Err_Threshold` (Optional) as a float value. Default value 1e-10 \n
        `iterMax` (Optional) as an Integer. Default value 5000 \n
        
    The function returns a list with the solution found with Lagrangian parameters computed (lambda and mu)
    and a tab of error at each step to study the convergence.
    '''
    err_tab = []
    err_d_k = 100.0
    inv_Sig = inv(Sig_mat)
    
    x_k, mu_k = uz_Scheme(inv_Sig, mu_init, rho, bounds[0], bounds[1])
    err_k = norm(d_Lagrangian(x_k, Sig_mat, mu_k))
    k = 0
    
    while err_d_k > Err_Threshold and k < iterMax:
        x_kk, mu_kk = uz_Scheme(inv_Sig, mu_k, rho, bounds[0], bounds[1])
        err_kk = norm(d_Lagrangian(x_kk, Sig_mat, mu_kk))
        err_d_k = np.abs(err_kk - err_k)
        err_tab.append(err_d_k)
        
        mu_k = mu_kk
        err_k = err_kk
        k += 1
    
    print 'Terminates in %.d iter  --- Weigths: %.2f | %.2f | %.2f | %.2f ... --- Error : %.10f' % (k, x_kk[0], x_kk[1], x_kk[2], x_kk[3], err_d_k)    
    return [x_kk, mu_k, err_tab, k]
             
@chrono.timing
def ArrowUrwicsz_MinVar(Sig_mat, bounds, rho, Err_Threshold=1.0e-10, iterMax=5000):
    '''Function executing the Arrow-Urwicz optimization algorithm adapted for a MinVar portfolio
    with "`no short selling`" constraint.
    It takes into parameters:
        `Sig_mat` as a n*n inverted var cov matrix \n
        `mu_init` as a n*1 ndarray/matrix \n
        `rho` as a scalar \n
        `Err_Threshold` (Optional) as a float value. Default value 1e-10 \n
        `iterMax` (Optional) as an Integer. Default value 5000 \n
        
    The function returns a list with the solution found with Lagrangian parameters computed (lambda and mu)
    and a tab of error at each step to study the convergence.
    '''
    #err_tab = []
    
    # Initialisation k=0:
    x_k = np.zeros((Sig_mat.shape[0],)) + 1.0 / Sig_mat.shape[0]
    mu_k = [np.ones([1,2]), np.ones([Sig_mat.shape[0], 2])]
    err_k = 1.0
    k = 0

    while err_k > Err_Threshold and k < iterMax:
        x_kk, mu_kk = AU_Scheme(x_k, Sig_mat, mu_k, rho, bounds[0], bounds[1])
        #err_k = norm(d_Lagrangian(x_kk, Sig_mat, mu_kk))
        err_k = np.abs((x_kk - x_k).sum())
        #err_tab.append(err_k)
        
        x_k = x_kk
        mu_k = mu_kk
        k += 1
    
    print 'Terminates in %.d iter  ---  Error : %.18f' % (k, err_k)       
    return [x_kk, mu_k, mu_k, k]

@chrono.timing
def Fast_Uz_MinVar(Sig_mat, bounds, rho, Err_Threshold=1.0e-10, iterMax=5000):
    '''Function executing the Uzawa optimization algorithm adapted for a MinVar portfolio
    with "`no short selling`" constraint.
    It takes into parameters:
        `Sig_mat` as a n*n inverted var cov matrix \n
        `rho` as a scalar \n
        `Err_Threshold` (Optional) as a float value. Default value 1e-10 \n
        `iterMax` (Optional) as an Integer. Default value 5000 \n
        
    The function returns a list with the solution found with Lagrangian parameters computed (lambda and mu)
    and a tab of error at each step to study the convergence.
    '''
    err_d_k = 100.0
    inv_Sig = inv(Sig_mat)
    
    x_k, mu_k = uz_Scheme(inv_Sig, [np.ones([1,2]), np.ones([Sig_mat.shape[0], 2])], rho, bounds[0], bounds[1])
    err_k = norm(d_Lagrangian(x_k, Sig_mat, mu_k))
    k = 0
    
    while err_d_k > Err_Threshold and k < iterMax:
        x_kk, mu_kk = uz_Scheme(inv_Sig, mu_k, rho, bounds[0], bounds[1])
        err_kk = norm(d_Lagrangian(x_kk, Sig_mat, mu_kk))
        err_d_k = np.abs(err_kk - err_k)
        
        mu_k = mu_kk
        err_k = err_kk
        k += 1
    
    print 'Terminates in %.d iter  ---  Error : %.18f' % (k, err_d_k)    
    return [x_kk, mu_k, k]
'''
##############################################################################
'''

def main():
    mu_k = [np.ones([1,2]), np.ones([4,2])]

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
    w, mu, r, k = Uzawa_MinVar(test_VCov_Mat, [0.05,0.3], mu_k, 0.001, 1.0e-10, 500000)


    
if __name__ == '__main__':
    main()
