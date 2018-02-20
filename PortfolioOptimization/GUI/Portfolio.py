#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:36:44 2017

@author: wasp
"""
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import matplotlib as mpl

#import datetime
#from pandas.tseries.offsets import *
from pandas.tseries.offsets import BMonthBegin, BMonthEnd, MonthEnd, BDay

#import PortfolioOptimization.utils.po_theme as th
import PortfolioOptimization.MinimumVariance.MinVar_bounded_v2 as mvb2
import Benchmark as bench

import numpy as np
import pandas as pd

class Portfolio():
    
    def __init__(self, Universe = 'Stoxx 50', Dates = ['01-01-2015','01-01-2017'], Frequency = 1, Histo_Length = 6, Wght_Const = [0.0, 10.0]):
        self._Universe = Universe
        self._Benchmark = bench.Benchmark()
        self._Dates = Dates
        self._Frequency = Frequency
        self._Histo_Length = Histo_Length
        self._Wght_Constraint = Wght_Const
        self._Wght_Histo = pd.DataFrame()
        self._Perf = pd.Series()
        self._Weights = None
        self._Opt_Strat = 'EW'

    @property
    def Universe(self):
        return self._Universe
    
    @Universe.setter
    def Universe(self, Universe):
        self._Universe = Universe
    
    @property
    def Benchmark(self):
        return self._Benchmark
    
    @Benchmark.setter
    def Benchmark(self, Benchmark):
        self._Benchmark = Benchmark
        
    @property
    def Dates(self):
        return self._Dates

    @Dates.setter
    def Dates(self, Dates):
        self._Dates = Dates

    @property
    def Frequency(self):
        return self._Frequency
    
    @Frequency.setter
    def Frequency(self, Frequency):
        self._Frequency = Frequency

    @property
    def Histo_Length(self):
        return self._Histo_Length
    
    @Histo_Length.setter
    def Histo_Length(self, Histo_Length):
        self._Histo_Length = Histo_Length
    
    @property
    def Wght_Constraint(self):
        return self._Wght_Constraint
    
    @Wght_Constraint.setter
    def Wght_Constraint(self, Wght_Constraint):
        self._Wght_Constraint = Wght_Constraint    

    @property
    def Wght_Histo(self):
        return self._Wght_Histo
    
    @Wght_Histo.setter
    def Wght_Histo(self, Wght_Histo):
        self._Wght_Histo = Wght_Histo

    @property
    def Perf(self):
        return self._Perf
    
    @Perf.setter
    def Perf(self, Perf):
        self._Perf = Perf
        
    @property
    def Weights(self):
        return self._Perf
    
    @Weights.setter
    def Weights(self, Weights):
        self._Weights = Weights
    
    @property
    def Opt_Strat(self):
        return self._Opt_Strat
    
    @Opt_Strat.setter
    def Weights(self, Opt_Strat):
        self._Opt_Strat = Opt_Strat
    
    def Equal_Weighted(self, n):
        self.Weights = np.repeat(1.0 / n, n)
        
    def MinVariance(self, VCov_Matrix, path=0.01):
        #self.Weights = mvb2.Fast_Uz_MinVar(VCov_Matrix, self.Wght_Constraint, path, 1.0e-15, 100000)[0]
        self.Weights = mvb2.ArrowUrwicsz_MinVar(VCov_Matrix, self.Wght_Constraint, path, 1.0e-10, 100000)[0]
    
    def RB_Weighted_1_risk(self, risk):
        '''
        Function to Compute an ERC portfolio. Risks can be the volatility
        as a (1/Sigma) portfolio or a (1/Beta) portfolio
        '''
        w = 1.0 / risk
        self.Weights = w / np.sum(w)
    
    def Compute_Performance(self, data_bckTest):
        fl_Wght = self.Weights * np.cumprod(1.0 + data_bckTest.shift(1).fillna(0)) 
        fl_Wght = fl_Wght.multiply(1.0 / fl_Wght.sum(axis=1), axis=0)
        
        self.Perf = self.Perf.append(np.sum(fl_Wght * data_bckTest, axis=1))
    
    def BackTest_MinVar(self):
        d = self.Dates[0]
        for i in xrange(int(self.Periods_Diff(self.Dates[0], self.Dates[1]) / self.Frequency)):
            d_f = d - BMonthBegin() * self.Histo_Length
            vcov, assets = self.Benchmark.Extract_VCov_Matrix([d_f, d], d - MonthEnd())
            self.MinVariance(vcov, 0.001)
            bckData = self.Benchmark.Extract_Returns(
                    [d + BDay(), (d + BDay()) + BMonthEnd() * self.Frequency], 
                     d - MonthEnd()).loc[:, assets]
            self.Compute_Performance(bckData)
            self.Wght_Histo[d] = pd.DataFrame(self.Weights, index=assets)            
            d = (d + BMonthEnd() * self.Frequency) + BDay()

    def BackTest_EqualWeight(self):
        d = self.Dates[0]
        for i in xrange(int(self.Periods_Diff(self.Dates[0], self.Dates[1]) / self.Frequency)):
            assets = self.Benchmark.Extract_Compo(d - MonthEnd())
            self.Equal_Weighted(len(assets))
            bckData = self.Benchmark.Extract_Returns(
                    [d + BDay(), (d + BDay()) + BMonthEnd() * self.Frequency], 
                     d - MonthEnd()).loc[:, assets]
            self.Compute_Performance(bckData)
            self.Wght_Histo[d] = pd.DataFrame(self.Weights, index=assets)  
            d = (d + BMonthEnd() * self.Frequency) + BDay()
    
    def BackTest_1_Beta(self):
        d = self.Dates[0]
        for i in xrange(int(self.Periods_Diff(self.Dates[0], self.Dates[1]) / self.Frequency)):
            # Compute the date for historical data extract
            d_f = d - BMonthBegin() * self.Histo_Length
            # Extract the Benchmark index returns
            bnch = self.Benchmark.Extract_Index_Returns([d_f, d])
            returns = self.Benchmark.Extract_Returns([d_f,d], d - MonthEnd())
            assets = self.Benchmark.Extract_Compo(d - MonthEnd())
            std_Index = np.std(bnch)
            beta = np.empty(len(assets))
            for i in xrange(len(beta)):
                # Delete missing value in the estimation
                beta[i] = bnch.iloc[:,0].cov(returns.iloc[:,i]) / std_Index
            
            self.RB_Weighted_1_risk(beta)
            
            bckData = self.Benchmark.Extract_Returns(
                    [d + BDay(), (d + BDay()) + BMonthEnd() * self.Frequency], 
                     d - MonthEnd()).loc[:, assets]
            self.Compute_Performance(bckData)
            self.Wght_Histo[d] = pd.DataFrame(self.Weights, index=assets)  
            d = (d + BMonthEnd() * self.Frequency) + BDay()    

    def BackTest_1_Sigma(self):
        d = self.Dates[0]
        for i in xrange(int(self.Periods_Diff(self.Dates[0], self.Dates[1]) / self.Frequency)):
            d_f = d - BMonthBegin() * self.Histo_Length
            Std, assets = self.Benchmark.Extract_Sigma([d_f, d], d - MonthEnd())
            self.RB_Weighted_1_risk(Std)
            
            bckData = self.Benchmark.Extract_Returns(
                    [d + BDay(), (d + BDay()) + BMonthEnd() * self.Frequency], 
                     d - MonthEnd()).loc[:, assets]
            self.Compute_Performance(bckData)
            self.Wght_Histo[d] = pd.DataFrame(self.Weights, index=assets)  
            d = (d + BMonthEnd() * self.Frequency) + BDay()  
    '''
    Fonction à déplacer dans utils.py
    '''
    def Periods_Diff(self, Datetime1, Datetime2):
        return (Datetime2.year - Datetime1.year) * 12 + (Datetime2.month - Datetime1.month)
        
    def Rebase(self, base=100.0):
        return base * np.cumprod(1.0 + self.Perf,axis=0)
    
    def Plot(self, base=100.0):
        fig, ax1 = plt.subplots()
        ax1.plot(self.Rebase())
        plt.show()
        