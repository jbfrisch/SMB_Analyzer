#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:24:57 2017

@author: wasp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import multi_dot

#try:
#    import PortfolioOptimization.utils.blpapiwrapper as myblp
#except ImportError:
#    raise ImportError('No Bloomberg Python Api Installed')


class Benchmark():
    
    def __init__(self, name='Stoxx 50', Ticker='SX5E Index', compo_historic=None, data_historic=None, data_Index=None):
        self.Name = name
        self.Ticker = Ticker
        self.Historic_Composition = compo_historic
        self.Historic_Data = data_historic
        self.Historic_Index = data_Index
        if data_historic is not None:
            self.Returns_Matrix()
        
    def Returns_Matrix(self):
        self.Historic_Returns = (self.Historic_Data / self.Historic_Data.shift(1) - 1.0)
        self.Historic_Returns.drop(self.Historic_Returns.index[0], inplace=True)
    
    def Clean_Data(self):
        self.Historic_Data.fillna(method='ffill', inplace=True)
        self.Historic_Data.columns = self.Historic_Data.columns.droplevel(1)
    
    def Import_Data(self, data):
        self.Historic_Data = data
        self.Clean_Data()
        self.Returns_Matrix()
    
    def Extract_Compo(self, date_extr):
        return (self.Historic_Composition[date_extr][pd.notnull(self.Historic_Composition[date_extr])] + ' Equity').tolist()
    
    def Extract_Data(self, date_extr, date_Compo):
        return self.Historic_Data[date_extr[0] : date_extr[1]].loc[:, self.Extract_Compo(date_Compo)]
    
    def Extract_Returns(self, date_extr, date_Compo):
        return self.Historic_Returns[date_extr[0] : date_extr[1]].loc[:, self.Extract_Compo(date_Compo)]
    
    def Extract_Index_Returns(self, date_extr):
        ext = self.Historic_Index[date_extr[0] : date_extr[1]]
        ext.columns = ext.columns.droplevel(1)
        return (ext / ext.shift(1) - 1.0)

    def Extract_Sigma(self, date_extr, date_Compo):
        ret = self.Extract_Returns(date_extr, date_Compo)
        ret.dropna(axis=1, inplace=True)
        
        return [ret.std(axis=0), ret.columns.values]
    
    def Extract_VCov_Matrix(self, date_extr, date_Compo):
        ret = self.Extract_Returns(date_extr, date_Compo)
        ret.dropna(axis=1, inplace=True)
        
        Est_Std = ret.std(axis=0) * np.sqrt(250)
        D = np.diag(Est_Std)
        Est_Corr = np.corrcoef(ret.T)
        
        return [multi_dot([D,Est_Corr,D]), Est_Std.index.values]

    def Extract_VCov_Matrix_PCA(self, date_extr, date_Compo):
        ret = self.Extract_Returns(date_extr, date_Compo)
        ret.dropna(axis=1, inplace=True)
        
        # Center and Reduction
        Est_Std = ret.std(axis=0)
        ret = (ret - ret.mean(axis=0)) / Est_Std
        
        D = np.diag(Est_Std)
        Est_Corr = np.corrcoef(ret.T)
        
        return [multi_dot([D,Est_Corr,D]), Est_Std.index.values]
    
    def Rebase(self, date_extr, base=100.0):
        return base * np.cumprod(1.0 + self.Extract_Index_Returns(date_extr), axis=0)
    
    def plot(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.Historic_Index)
        plt.show()
